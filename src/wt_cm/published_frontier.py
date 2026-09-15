import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, protocol as PR, detect_final as D
FILES = {'P1 Zaher ANN-NBM': 'P1', 'P2 Schlechtingen linear FSRC': 'P2', 'P3 GBM-NBM + EWMA': 'P3', 'P5 multivariate CUSUM': 'P5'}
MERGE = pd.Timedelta('45D')
TARGETS = [0.221, 0.743]

def episodes(s, thr):
    hi = s[s > thr]
    if not len(hi):
        return []
    out, start, last = ([], hi.index[0], hi.index[0])
    for t in hi.index[1:]:
        if t - last > MERGE:
            out.append(start)
            start = t
        last = t
    out.append(start)
    return out
rows = []
for name, code in FILES.items():
    for case in ['MF07', 'MF12']:
        M = P.load_trajectory(f'published_{case}__{code}')
        c = PR.CASES[case]
        ev = pd.Timestamp(c['event'])
        ctl = {t: M[t].loc[c['monitor_from']:].dropna() for t in D.HEALTHY if t in M and M[t].notna().sum() > 10}
        yrs = sum(((x.index[-1] - x.index[0]).days / 365.25 for x in ctl.values()))
        s = M[case].loc[c['monitor_from']:ev].dropna()
        if not len(s) or not yrs:
            continue
        allv = np.concatenate([x.values for x in ctl.values()])
        grid = np.unique(np.quantile(allv, np.linspace(0.5, 0.99999, 400)))
        for target in TARGETS:
            best = None
            for thr in grid:
                n = sum((len(episodes(x, thr)) for x in ctl.values()))
                rate = n / yrs
                if rate <= target:
                    ep = episodes(s, thr)
                    lead = (ev - ep[0]).days if ep else np.nan
                    span = (ev - pd.Timestamp(c['monitor_from'])).days
                    near = bool(np.isfinite(lead) and lead >= span - 45)
                    best = dict(method=name, case=case, target_burden=target, threshold=round(float(thr), 4), control_episodes=n, control_yrs=round(yrs, 1), achieved_burden=round(rate, 3), lead_days=lead, monitoring_span_days=span, near_record_boundary=near)
                    break
            if best:
                rows.append(best)
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'published_frontier.csv', index=False)
for target in TARGETS:
    print(f'\n=== matched control burden <= {target} episodes/turbine-year ===')
    sub = d[d.target_burden == target]
    print(sub[['method', 'case', 'achieved_burden', 'lead_days', 'control_episodes', 'control_yrs']].to_string(index=False))
print('\nCAPE-WT: MF12 199 d warning / 139 d confirmation; MF07 >=294 / 219.')
print('The 0.221 target is the MF12 filter-only warning burden; 0.743 is the mature matched parallel burden.')
print('P4 is excluded from strict-online ranking because its event labels are used in training by design.')
