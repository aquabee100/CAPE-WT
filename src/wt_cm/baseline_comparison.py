import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
START, END = ('2020-01-01', '2022-07-19')
K, HTH, HOR = (3.0, 0.5, 180)
ARMS = {'physics frozen': 'resid_phys_frozen', 'physics adaptive': 'resid_phys_adaptive', 'GBM frozen': 'resid_gbm_frozen', 'GBM adaptive': 'resid_gbm_adaptive', 'linear frozen': 'resid_lin_frozen', 'linear adaptive': 'resid_lin_adaptive', 'MLP frozen': 'resid_mlp_frozen', 'MLP adaptive': 'resid_mlp_adaptive', 'PCA frozen': 'resid_pca_frozen', 'PCA adaptive': 'resid_pca_adaptive', 'Mahalanobis frozen': 'resid_maha_frozen', 'Mahalanobis adaptive': 'resid_maha_adaptive', 'IsolationForest frozen': 'resid_ifor_frozen', 'IsolationForest adaptive': 'resid_ifor_adaptive', 'autoencoder frozen': 'resid_ae_frozen', 'autoencoder adaptive': 'resid_ae_adaptive'}
MILANI = [dict(arm='Milani et al. 2025 (published)', case='MF07', evaluable='literature', warn_lead=222, conf_lead=None, note='blind test turbine'), dict(arm='Milani et al. 2025 (published)', case='MF12', evaluable='literature', warn_lead=178, conf_lead=None, note='IN-SAMPLE: event used in training')]
THE = P.load_trajectory('thermal_traj_online__VA')

def zof(M, case):
    return B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
if __name__ == '__main__':
    ctrl, case_rows = ([], [])
    for name, art in ARMS.items():
        f = P.RESULTS / f'{art}.parquet'
        if not f.exists():
            print(f'MISSING {art}')
            continue
        M = pd.read_parquet(f)
        Z = zof(M, 'MF12')
        dr, sd, wep, expo = ([], [], 0, 0.0)
        for t in D.HEALTHY:
            if t not in Z:
                continue
            z = Z[t].loc[START:END]
            v = z.dropna()
            if len(v) < 8:
                continue
            yrs = (v.index - v.index[0]).days / 365.25
            dr.append(float(np.polyfit(yrs, v.values, 1)[0]))
            sd.append(float(v.std()))
            wep += len(SM.warnings_of(SM.run_machine(z, z * 0 - 99, K, HTH, HTH, HOR, latch=False)))
            expo += H.evaluable_exposure(z, pd.Timestamp(END))
        ctrl.append(dict(arm=name, n=len(dr), exposure_yr=round(expo, 2), med_abs_drift=round(float(np.median(np.abs(dr))), 3), up=int(sum((d > 0 for d in dr))), down=int(sum((d < 0 for d in dr))), med_sd=round(float(np.median(sd)), 3), warn_episodes=wep, warn_per_turb_yr=round(wep / expo, 3) if expo else np.nan))
        for case in ['MF07', 'MF12']:
            Zc = zof(M, case)
            Zt = zof(THE, case)
            c = PR.CASES[case]
            ev = pd.Timestamp(c['event'])
            z = Zc[case].loc[c['monitor_from']:ev]
            v = z.dropna()
            if not len(v):
                case_rows.append(dict(arm=name, case=case, evaluable='no - warm-up exceeds pre-event record'))
                continue
            eps = SM.run_machine(z, Zt[case].loc[c['monitor_from']:ev], K, HTH, HTH, HOR, latch=True)
            ce = next((e for e in eps if e['status'] == 'confirmed'), None)
            case_rows.append(dict(arm=name, case=case, evaluable='yes', first_evaluable=v.index[0].date(), censored=bool(v.iloc[0] > K), warn_lead=(ev - eps[0]['warning']).days if eps else None, conf_lead=(ev - ce['confirmation']).days if ce else None, peak_z=round(float(v.max()), 2)))
    C = pd.DataFrame(ctrl)
    C.to_csv(P.RESULTS / 'baseline_controls.csv', index=False)
    case_rows += MILANI
    R = pd.DataFrame(case_rows)
    R.to_csv(P.RESULTS / 'baseline_cases.csv', index=False)
    print('=== CONTROL FLEET (7 turbines, %s..%s) — the informative comparison ===' % (START, END))
    print(C.sort_values('warn_per_turb_yr').to_string(index=False))
    print('\n=== FIELD CASES — descriptive only, NOT ranked ===')
    print(R.to_string(index=False))
