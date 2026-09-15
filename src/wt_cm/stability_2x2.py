import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
START, END = ('2020-01-01', '2022-07-19')
K, HTH = (3.0, 0.5)
CELLS = {('physics', 'frozen'): 'resid_phys_frozen', ('physics', 'adaptive'): 'resid_phys_adaptive', ('GBM', 'frozen'): 'resid_gbm_frozen', ('GBM', 'adaptive'): 'resid_gbm_adaptive'}

def stats(z):
    zz = z.loc[START:END]
    v = zz.dropna()
    if len(v) < 8:
        return None
    yrs = (v.index - v.index[0]).days / 365.25
    half = len(v) // 2
    ep = SM.run_machine(zz, zz * 0 - 99, K, HTH, HTH, 180, latch=False)
    return dict(evaluable_yr=round(H.evaluable_exposure(zz, pd.Timestamp(END)), 2), early=round(float(v.iloc[:half].median()), 3), late=round(float(v.iloc[half:].median()), 3), shift=round(float(v.iloc[half:].median() - v.iloc[:half].median()), 3), drift_per_yr=round(float(np.polyfit(yrs, v.values, 1)[0]), 3), sd=round(float(v.std()), 3), frac_z_gt3=round(float((v > 3).mean()), 3), warn_episodes=len(SM.warnings_of(ep)))
rows = []
for (struct, adapt), name in CELLS.items():
    M = pd.read_parquet(P.RESULTS / f'{name}.parquet')
    Z = B.rolling_z_v2(M, scale_end=PR.SCALE_END['MF12'], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
    for t in D.HEALTHY:
        if t not in Z:
            continue
        st = stats(Z[t])
        if st:
            rows.append(dict(structure=struct, adaptation=adapt, turbine=t, **st))
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'stability_2x2.csv', index=False)
print(d.to_string(index=False))
g = d.groupby(['structure', 'adaptation']).agg(n=('turbine', 'count'), exposure_yr=('evaluable_yr', 'sum'), med_abs_drift=('drift_per_yr', lambda s: round(s.abs().median(), 3)), up=('drift_per_yr', lambda s: int((s > 0).sum())), down=('drift_per_yr', lambda s: int((s < 0).sum())), med_abs_shift=('shift', lambda s: round(s.abs().median(), 3)), med_sd=('sd', 'median'), med_frac_z3=('frac_z_gt3', 'median'), warn=('warn_episodes', 'sum'))
g['warn_per_turb_yr'] = (g['warn'] / g['exposure_yr']).round(3)
print('\n=== fleet summary, 7 reference turbines, %s..%s ===' % (START, END))
print(g.to_string())
g.to_csv(P.RESULTS / 'stability_2x2_summary.csv')
