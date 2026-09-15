import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
import parallel_arch as PA
START = pd.Timestamp(PR.CASES['MF07']['monitor_from'])
SCALE = PR.SCALE_END['MF07']
STEP = H.STEP
FIL = P.load_trajectory('filter_traj_online')
GBM = P.load_trajectory('resid_gbm_adaptive')
THE = P.load_trajectory('thermal_traj_online__VA')
z = lambda M: B.rolling_z_v2(M, scale_end=SCALE, min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
Zf, Zg, Zt = (z(FIL), z(GBM), z(THE))

def expo_of(zs, mask):
    idx = zs.index
    t = pd.Series(idx, index=idx)
    nxt = t.shift(-1)
    lim = pd.concat([nxt, t + STEP, pd.Series(H.RECORD_END, index=idx)], axis=1).min(axis=1)
    dt = (lim - t).clip(lower=pd.Timedelta(0))
    sel = zs.notna() & mask
    return float(dt[sel].sum().total_seconds() / (365.25 * 24 * 3600))
rows = []
per_turb = []
for t in D.HEALTHY:
    zf, zg, zt = (Zf[t].loc[START:], Zg[t].reindex(Zf.index).loc[START:], Zt[t].loc[START:])
    availB = zg.notna()
    eA = expo_of(zf, ~availB)
    eB = expo_of(zf, availB)
    eps = PA.run_parallel(zf, zg, zt, latch=False)
    cA = cB = 0
    srcB = {'filter': 0, 'GBM': 0, 'both': 0}
    conf = 0
    for e in eps:
        inB = bool(availB.get(e['warning'], False))
        if inB:
            cB += 1
            srcB[e['source']] = srcB.get(e['source'], 0) + 1
        else:
            cA += 1
        conf += e['status'] == 'confirmed'
    per_turb.append(dict(turbine=t, expo_A=eA, warn_A=cA, expo_B=eB, warn_B=cB, confirmed=conf, **{f'B_{k}': v for k, v in srcB.items()}))
pt = pd.DataFrame(per_turb)
pt.to_csv(P.RESULTS / 'availability_per_turbine.csv', index=False)
print('=== per control turbine, unique fleet timeline from %s ===' % START.date())
print(pt.to_string(index=False))
EA, EB = (pt.expo_A.sum(), pt.expo_B.sum())
WA, WB = (pt.warn_A.sum(), pt.warn_B.sum())
summ = pd.DataFrame([dict(regime='A: GBM unavailable (warm-up)', channels='filter only', exposure_yr=round(EA, 2), warn_episodes=int(WA), far_warn=round(WA / EA, 3) if EA else np.nan, filter_only=int(WA), gbm_only=0, both=0), dict(regime='B: GBM available', channels='filter OR GBM', exposure_yr=round(EB, 2), warn_episodes=int(WB), far_warn=round(WB / EB, 3) if EB else np.nan, filter_only=int(pt.B_filter.sum()), gbm_only=int(pt.B_GBM.sum()), both=int(pt.B_both.sum()))])
summ.to_csv(P.RESULTS / 'availability_stratified.csv', index=False)
print('\n=== FINAL: warning burden stratified by GBM availability ===')
print(summ.to_string(index=False))
tot_conf = int(pt.confirmed.sum())
T = EA + EB
print(f'\nconfirmed control alarms: {tot_conf} over {T:.2f} turbine-years   one-sided 95% upper rate < {3.0 / T:.3f}/turbine-year')
