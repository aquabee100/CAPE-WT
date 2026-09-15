import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
K, HTH, HOR = (3.0, 0.5, 180)

def zof(M, case):
    return B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
CHANNELS = {'filter dp (deployed)': lambda: P.load_trajectory('filter_traj_online'), 'GBM adaptive residual': lambda: P.load_trajectory('resid_gbm_adaptive'), 'physics adaptive residual': lambda: pd.read_parquet(P.RESULTS / 'resid_phys_adaptive.parquet')}
THERMAL = P.load_trajectory('thermal_traj_online__VA')
rows = []
for case in ['MF07', 'MF12']:
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    Zt = zof(THERMAL, case)
    for name, load in CHANNELS.items():
        Zw = zof(load(), case)
        if case not in Zw or Zw[case].loc[c['monitor_from']:ev].dropna().empty:
            rows.append(dict(case=case, warn_channel=name, evaluable='NO - warm-up exceeds pre-event record'))
            continue
        sl = slice(c['monitor_from'], ev)
        slc = slice(c['monitor_from'], None)
        eps = SM.run_machine(Zw[case].loc[sl], Zt[case].loc[sl], K, HTH, HTH, HOR)
        ce = SM.first_confirmed_episode(eps)
        nw = nc = 0
        expo = 0.0
        for t in D.HEALTHY:
            zw, zt = (Zw[t].loc[slc], Zt[t].loc[slc])
            he = SM.run_machine(zw, zt, K, HTH, HTH, HOR, latch=False)
            nw += len(SM.warnings_of(he))
            nc += len(SM.confirmations_of(he))
            expo += H.evaluable_exposure(zw, H.RECORD_END)
        rows.append(dict(case=case, warn_channel=name, evaluable='yes', warn_lead=(ev - eps[0]['warning']).days if eps else None, conf_lead=(ev - ce['confirmation']).days if ce else None, warn_eps=nw, conf_alarms=nc, exposure_yr=round(expo, 2), far_warn=round(nw / expo, 3) if expo else np.nan, far_conf=round(nc / expo, 3) if expo else np.nan, far_conf_upper95=round(3.0 / expo, 3) if expo and nc == 0 else None))
r = pd.DataFrame(rows)
r.to_csv(P.RESULTS / 'architecture_comparison.csv', index=False)
print(r.to_string(index=False))
