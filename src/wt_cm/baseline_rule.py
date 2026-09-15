import numpy as np
import pandas as pd
import paths as P
import detect_final as D
import normalise as N
import protocol as PR
import state_machine as SM
MIN_VALID_BASE_WINDOWS = 12
MIN_HISTORY_WINDOWS = 25
MIN_HISTORY_DAYS = 360

def rolling_z_v2(M, start=None, scale_end=None, min_valid=MIN_VALID_BASE_WINDOWS, min_history=MIN_HISTORY_WINDOWS, min_history_days=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    end = scale_end or PR.DEV_END
    dev = M.loc[start:end] if start else M.loc[:end]
    s_pool = N.pooled_scale(dev, dev.median())
    if not np.isfinite(s_pool) or s_pool <= 0:
        return M * np.nan
    out = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    for t in M.index:
        lo, hi = (t - N.ROLL_EMBARGO - N.ROLL_WINDOW, t - N.ROLL_EMBARGO)
        base = M.loc[lo:hi]
        if len(base) < min_history:
            continue
        if min_history_days is not None and (not len(base) or (base.index[-1] - base.index[0]).days < min_history_days):
            continue
        counts = base.notna().sum()
        med = base.median()
        ok = counts >= min_valid
        idx = ok[ok].index
        out.loc[t, idx] = (M.loc[t, idx] - med[idx]) / s_pool
    return out

def evaluate(Ft, Tt, case, k=3.0, h=0.5, horizon=180, min_history=25):
    c = PR.CASES[case]
    event = pd.Timestamp(c['event'])
    Zf = rolling_z_v2(Ft, scale_end=PR.SCALE_END[case], min_history=min_history)
    Zt = rolling_z_v2(Tt, scale_end=PR.SCALE_END[case], min_history=min_history)
    sl = slice(c['monitor_from'], event)
    slc = slice(c['monitor_from'], None)
    eps = SM.run_machine(Zf[case].loc[sl], Zt[case].loc[sl], k, h, h, horizon)
    ce = SM.first_confirmed_episode(eps)
    nw = nc = 0
    expo = 0.0
    for t in D.HEALTHY:
        zf, zt = (Zf[t].loc[slc], Zt[t].loc[slc])
        he = SM.run_machine(zf, zt, k, h, h, horizon, latch=False)
        nw += len(SM.warnings_of(he))
        nc += len(SM.confirmations_of(he))
        expo += (zf.index[-1] - zf.index[0]).days / 365.25
    return dict(first_warning=eps[0]['warning'].date() if eps else None, warn_lead=(event - eps[0]['warning']).days if eps else None, confirmation=ce['confirmation'].date() if ce else None, conf_lead=(event - ce['confirmation']).days if ce else None, fp_warn=nw, fp_conf=nc, exposure=round(expo, 1), far_warn=round(nw / expo, 3) if expo else np.nan, z_at_first=round(float(Zf[case].loc[eps[0]['warning']]), 2) if eps else None)
if __name__ == '__main__':
    Tt = P.load('causal_traj')['VA']
    F_arch = P.load('filter_traj')
    F_online = P.load_trajectory('filter_traj_online')
    print('=== history-requirement sweep, ARCHIVE filter preprocessing ===')
    rows = []
    for case in ['MF07', 'MF12']:
        for mh in [12, 23, 24, 25, 26]:
            r = evaluate(F_arch, Tt, case, min_history=mh)
            rows.append(dict(case=case, pipeline='archive', min_history=mh, **r))
    a = pd.DataFrame(rows)
    print(a.to_string(index=False))
    print('\n=== COMBINED: strict-online filter + full-history baseline ===')
    rows2 = []
    for case in ['MF07', 'MF12']:
        for lab, FF in [('archive screen', F_arch), ('strict online', F_online)]:
            for mh in [12, 25]:
                r = evaluate(FF, Tt, case, min_history=mh)
                rows2.append(dict(case=case, filter_pipeline=lab, min_history=mh, **r))
    b = pd.DataFrame(rows2)
    print(b.to_string(index=False))
    a.to_csv(P.RESULTS / 'history_sweep.csv', index=False)
    b.to_csv(P.RESULTS / 'headline_combined.csv', index=False)
    print('\nwritten: results/history_sweep.csv, results/headline_combined.csv')
