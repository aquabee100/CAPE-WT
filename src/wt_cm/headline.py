import numpy as np
import pandas as pd
import paths as P
import causal as C
import detect_final as D
import normalise as N
import protocol as PR
import state_machine as SM
import baseline_rule as B
BASELINE = N.ROLL_WINDOW
STEP = C.STEP
FULL_GRID_SPAN = BASELINE // STEP * STEP
MIN_VALID_BASE_WINDOWS = 12
RECORD_END = pd.Timestamp('2022-08-01 23:50:00')

def evaluable_exposure(z, followup_end, step=STEP):
    idx = z.index
    if len(idx) < 2:
        return 0.0
    t = pd.Series(idx, index=idx)
    nxt = t.shift(-1)
    end = pd.Timestamp(followup_end)
    lim = pd.concat([nxt, t + step, pd.Series(end, index=idx)], axis=1).min(axis=1)
    dt = (lim - t).clip(lower=pd.Timedelta(0))
    return float(dt[z.notna()].sum().total_seconds() / (365.25 * 24 * 3600))

def evaluate(Ft, Tt, case, k=3.0, h=0.5, horizon=180):
    c = PR.CASES[case]
    event = pd.Timestamp(c['event'])
    kw = dict(scale_end=PR.SCALE_END[case], min_history=1, min_history_days=FULL_GRID_SPAN.days, min_valid=MIN_VALID_BASE_WINDOWS)
    Zf, Zt = (B.rolling_z_v2(Ft, **kw), B.rolling_z_v2(Tt, **kw))
    sl, slc = (slice(c['monitor_from'], event), slice(c['monitor_from'], None))
    eps = SM.run_machine(Zf[case].loc[sl], Zt[case].loc[sl], k, h, h, horizon)
    ce = SM.first_confirmed_episode(eps)
    nw = nc = 0
    expo_f = expo_t = 0.0
    for t in D.HEALTHY:
        zf, zt = (Zf[t].loc[slc], Zt[t].loc[slc])
        he = SM.run_machine(zf, zt, k, h, h, horizon, latch=False)
        nw += len(SM.warnings_of(he))
        nc += len(SM.confirmations_of(he))
        expo_f += evaluable_exposure(zf, RECORD_END)
        expo_t += evaluable_exposure(zt, RECORD_END)
    return dict(case=case, first_window=Zf[case].loc[sl].dropna().index[0].date(), warning=eps[0]['warning'].date() if eps else None, warn_lead=(event - eps[0]['warning']).days if eps else None, z_at_warning=round(float(Zf[case].loc[eps[0]['warning']]), 2) if eps else None, confirmation=ce['confirmation'].date() if ce else None, conf_lead=(event - ce['confirmation']).days if ce else None, warn_episodes=nw, confirmed_alarms=nc, filter_exposure_yr=round(expo_f, 2), thermal_exposure_yr=round(expo_t, 2), far_warn=round(nw / expo_f, 3) if expo_f else np.nan, far_conf=round(nc / expo_f, 3) if expo_f else np.nan)
if __name__ == '__main__':
    print(f'BASELINE {BASELINE.days}d // STEP {STEP.days}d -> FULL_GRID_SPAN = {FULL_GRID_SPAN.days}d (derived)\n')
    Tt = P.load_trajectory('thermal_traj_online__VA')
    F_on = P.load_trajectory('filter_traj_online')
    print('=== TRACK A - PRIMARY, strict online ===')
    prim = pd.DataFrame([evaluate(F_on, Tt, c) for c in ['MF07', 'MF12']])
    print(prim.to_string(index=False))
    prim.to_csv(P.RESULTS / 'headline_online.csv', index=False)
    print('\nwritten: results/headline_online.csv')
