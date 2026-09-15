import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, protocol as PR, detect_final as D
import baseline_rule as B, headline as H, state_machine as SM
K, HTH, HOR = (3.0, 0.5, 180)
LIT_CHANNELS = {'P1 Zaher ANN-NBM': 'P1', 'P2 Schlechtingen linear FSRC': 'P2', 'P3 GBM-NBM + EWMA': 'P3', 'P5 multivariate CUSUM': 'P5'}
PROPOSED_CHANNELS = {'filter (proposed)': 'filter_traj_online', 'adaptive GBM (proposed)': 'resid_gbm_adaptive'}
GRID = P.load_trajectory('filter_traj_online').index
THE = P.load_trajectory('thermal_traj_online__VA')

def to_grid(M, preserve_availability=False):
    if M.index.equals(GRID):
        return M
    if preserve_availability:
        return M.reindex(GRID)
    out = {}
    for c in M.columns:
        s = M[c].dropna()
        out[c] = pd.Series([s.loc[t - pd.Timedelta('45D'):t].mean() if len(s.loc[t - pd.Timedelta('45D'):t]) else np.nan for t in GRID], index=GRID)
    return pd.DataFrame(out)
rows = []
for name in list(LIT_CHANNELS) + list(PROPOSED_CHANNELS):
    for case in ['MF07', 'MF12']:
        if name in LIT_CHANNELS:
            Mraw = P.load_trajectory(f'published_{case}__{LIT_CHANNELS[name]}')
        else:
            Mraw = P.load_trajectory(PROPOSED_CHANNELS[name])
        c = PR.CASES[case]
        ev = pd.Timestamp(c['event'])
        kw = dict(scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
        if name == 'adaptive GBM (proposed)':
            Zw = B.rolling_z_v2(Mraw, **kw).reindex(GRID)
        else:
            Zw = B.rolling_z_v2(to_grid(Mraw), **kw)
        Zt = B.rolling_z_v2(THE, **kw)
        if case not in Zw or Zw[case].loc[c['monitor_from']:ev].dropna().empty:
            rows.append(dict(channel=name, case=case, evaluable='no'))
            continue
        eps = SM.run_machine(Zw[case].loc[c['monitor_from']:ev], Zt[case].loc[c['monitor_from']:ev], K, HTH, HTH, HOR)
        ce = next((e for e in eps if e['status'] == 'confirmed'), None)
        nw = nc = 0
        expo = 0.0
        for t in D.HEALTHY:
            if t not in Zw:
                continue
            zw, zt = (Zw[t].loc[c['monitor_from']:], Zt[t].loc[c['monitor_from']:])
            he = SM.run_machine(zw, zt, K, HTH, HTH, HOR, latch=False)
            nw += len(he)
            nc += sum((e['status'] == 'confirmed' for e in he))
            expo += H.evaluable_exposure(zw, H.RECORD_END)
        rows.append(dict(channel=name, case=case, evaluable='yes', warn_lead=(ev - eps[0]['warning']).days if eps else None, conf_lead=(ev - ce['confirmation']).days if ce else None, warn_eps=nw, confirmed=nc, expo=round(expo, 2), far_warn=round(nw / expo, 3) if expo else np.nan, far_conf=round(nc / expo, 3) if expo else np.nan))
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'published_twostage.csv', index=False)
for case in ['MF07', 'MF12']:
    print(f'\n=== {case}: published warning channel + proposed confirmation stage ===')
    print(d[d.case == case].drop(columns='case').to_string(index=False))
