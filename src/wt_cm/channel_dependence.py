import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import paths as P
import protocol as PR
import detect_final as D
import baseline_rule as B
import headline as H
K = 3.0
CH = {'filter': 'filter_traj_online', 'GBM': 'resid_gbm_adaptive', 'thermal': 'thermal_traj_online__VA'}
PAIRS = [('filter', 'thermal'), ('GBM', 'thermal'), ('filter', 'GBM')]

def zed(case):
    return {k: B.rolling_z_v2(P.load_trajectory(f), scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days) for k, f in CH.items()}

def corr(a, b):
    a, b = (a.dropna(), b.dropna())
    j = a.index.intersection(b.index)
    return (np.corrcoef(a[j], b[j])[0, 1], len(j)) if len(j) > 8 else (np.nan, len(j))

def control_corr(Z, turbines, pair):
    per = {}
    for t in turbines:
        if t in Z[pair[0]] and t in Z[pair[1]]:
            r, n = corr(Z[pair[0]][t], Z[pair[1]][t])
            if np.isfinite(r):
                per[t] = r
    return per

def cluster_bootstrap(per, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array(list(per.values()))
    if len(v) < 2:
        return (np.nan, np.nan)
    d = [np.median(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
if __name__ == '__main__':
    rows, perrows = ([], [])
    for case in ['MF07', 'MF12']:
        Z = zed(case)
        c = PR.CASES[case]
        ev = pd.Timestamp(c['event'])
        for pair in PAIRS:
            per = control_corr(Z, D.HEALTHY, pair)
            lo, hi = cluster_bootstrap(per)
            med = float(np.median(list(per.values()))) if per else np.nan
            xs = pd.concat([Z[pair[0]][t].dropna() for t in D.HEALTHY])
            ys = pd.concat([Z[pair[1]][t].dropna() for t in D.HEALTHY])
            j = xs.index.intersection(ys.index)
            pooled = np.corrcoef(xs[j], ys[j])[0, 1] if len(j) > 8 else np.nan
            fr, fn = corr(Z[pair[0]][case].loc[c['monitor_from']:ev], Z[pair[1]][case].loc[c['monitor_from']:ev])
            rows.append(dict(case=case, pair=f'{pair[0]}-{pair[1]}', control_median_r=round(med, 3), control_ci_lo=round(lo, 3), control_ci_hi=round(hi, 3), control_pooled_r=round(pooled, 3), n_control_turbines=len(per), pre_event_r=round(fr, 3), n_pre_event=fn))
            for t, r in per.items():
                perrows.append(dict(case=case, pair=f'{pair[0]}-{pair[1]}', turbine=t, r=round(r, 3)))
        for wch in ['filter', 'GBM']:
            for label, turbines in [('control', D.HEALTHY), ('fault', [case])]:
                hi_w = hi_both = 0
                for t in turbines:
                    if t not in Z[wch]:
                        continue
                    sl = slice(c['monitor_from'], ev if label == 'fault' else None)
                    w = Z[wch][t].loc[sl]
                    p = Z['thermal'][t].loc[sl]
                    j = w.dropna().index.intersection(p.dropna().index)
                    hi_w += int((w[j] > K).sum())
                    hi_both += int(((w[j] > K) & (p[j] > K)).sum())
                rows.append(dict(case=case, pair=f'cooccur {wch}>{K:.0f} & thermal>{K:.0f}', control_median_r=np.nan, control_ci_lo=np.nan, control_ci_hi=np.nan, control_pooled_r=np.nan, n_control_turbines=len(turbines), pre_event_r=round(hi_both / hi_w, 3) if hi_w else np.nan, n_pre_event=hi_w) if label == 'fault' else dict(case=case, pair=f'cooccur {wch}>{K:.0f} & thermal>{K:.0f} [control]', control_median_r=round(hi_both / hi_w, 3) if hi_w else np.nan, control_ci_lo=np.nan, control_ci_hi=np.nan, control_pooled_r=np.nan, n_control_turbines=len(turbines), pre_event_r=np.nan, n_pre_event=hi_w))
    out = pd.DataFrame(rows)
    out.to_csv(P.RESULTS / 'channel_dependence.csv', index=False)
    pd.DataFrame(perrows).to_csv(P.RESULTS / 'channel_dependence_per_turbine.csv', index=False)
    corrs = out[~out.pair.str.startswith('cooccur')]
    print('=== channel correlation: healthy controls vs pre-event ===')
    print(corrs[['case', 'pair', 'control_median_r', 'control_ci_lo', 'control_ci_hi', 'control_pooled_r', 'pre_event_r', 'n_pre_event']].to_string(index=False))
    print('\n=== co-occurrence of elevated warning and elevated confirmation ===')
    print(out[out.pair.str.startswith('cooccur')][['case', 'pair', 'control_median_r', 'pre_event_r', 'n_pre_event']].to_string(index=False))
