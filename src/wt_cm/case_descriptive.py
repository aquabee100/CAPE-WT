import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
CELLS = {'physics frozen': 'resid_phys_frozen', 'physics adaptive': 'resid_phys_adaptive', 'GBM frozen': 'resid_gbm_frozen', 'GBM adaptive': 'resid_gbm_adaptive'}
K, HTH, HORIZON = (3.0, 0.5, 180)
PRE = 365

def control_far(Z, case):
    c = PR.CASES[case]
    nw = 0.0
    expo = 0.0
    for t in D.HEALTHY:
        z = Z[t].loc[c['monitor_from']:]
        ep = SM.run_machine(z, z * 0 - 99, K, HTH, HTH, HORIZON, latch=False)
        nw += len(SM.warnings_of(ep))
        expo += H.evaluable_exposure(z, H.RECORD_END)
    return (nw, expo)
rows = []
for case in ['MF07', 'MF12']:
    c = PR.CASES[case]
    event = pd.Timestamp(c['event'])
    for label, art in CELLS.items():
        M = pd.read_parquet(P.RESULTS / f'{art}.parquet')
        Z = B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
        z = Z[case].loc[c['monitor_from']:event]
        v = z.dropna()
        if not len(v):
            first_ev = Z[case].dropna()
            rows.append(dict(case=case, model=label, evaluable='NO - first window ' + str(first_ev.index[0].date()) + ' is after the event', first_warning=None, lead_days=None, first_z_gt3=None, lead_z_gt3=None, peak_z=None, auc_pre365=None, control_warn=None, control_yrs=None, control_far=None))
            continue
        first3 = v[v > 3].index[0] if (v > 3).any() else None
        ep = SM.run_machine(z, z * 0 - 99, K, HTH, HTH, HORIZON, latch=True)
        w = SM.warnings_of(ep)
        seg = v.loc[event - pd.Timedelta(days=PRE):event]
        if len(seg) > 1:
            dtd = np.diff(seg.index.values).astype('timedelta64[D]').astype(float)
            auc = float((np.clip(seg.values[:-1], 0, None) * dtd).sum())
        else:
            auc = 0.0
        nw, expo = control_far(Z, case)
        rows.append(dict(case=case, model=label, evaluable='yes', first_warning=w[0].date() if w else None, lead_days=(event - pd.Timestamp(w[0])).days if w else None, first_evaluable=v.index[0].date(), censored=bool(len(v) and v.iloc[0] > K), first_z_gt3=first3.date() if first3 is not None else None, lead_z_gt3=(event - first3).days if first3 is not None else None, peak_z=round(float(v.max()), 2), auc_pre365=round(auc, 1), control_warn=int(nw), control_yrs=round(expo, 2), control_far=round(nw / expo, 3) if expo else np.nan))
d = pd.DataFrame(rows)
for case in ['MF07', 'MF12']:
    print(f"\n=== {case} (event {PR.CASES[case]['event']}) ===")
    print(d[d.case == case].drop(columns='case').to_string(index=False))
d.to_csv(P.RESULTS / 'case_descriptive.csv', index=False)
print('\nwritten: results/case_descriptive.csv')
