from __future__ import annotations
import numpy as np
import pandas as pd
import paths as P
import protocol as PR
import detect_final as D
import parallel_arch as PA
import headline as H

def main():
    F = P.load_trajectory('filter_traj_online')
    G = P.load_trajectory('resid_gbm_adaptive')
    T = P.load_trajectory('thermal_traj_online__VA')
    rows = []
    for case in ['MF07', 'MF12']:
        c = PR.CASES[case]
        ev = pd.Timestamp(c['event'])
        Zf, Zg, Zt = (PA.zof(F, case), PA.zof(G, case), PA.zof(T, case))
        for turb in [case, *D.HEALTHY]:
            role = 'case' if turb == case else 'control'
            stop = ev if role == 'case' else H.RECORD_END
            zf = Zf[turb].loc[c['monitor_from']:stop]
            zg = Zg[turb].reindex(zf.index)
            zt = Zt[turb].reindex(zf.index)
            eps = PA.run_parallel(zf, zg, zt, latch=role == 'case')
            for j, e in enumerate(eps, 1):
                w = pd.Timestamp(e['warning'])
                conf = e.get('confirmation')
                conf = pd.Timestamp(conf) if conf is not None else pd.NaT
                age = (conf - w).days if pd.notna(conf) else np.nan
                assert pd.isna(conf) or conf >= w
                assert pd.isna(age) or age <= PA.HOR
                rows.append(dict(case=case, turbine=turb, role=role, episode=j, warning=w, status=e['status'], confirmation=conf, warning_source=e.get('source', 'filter'), gbm_available_at_warning=bool(np.isfinite(zg.get(w, np.nan))), warning_lead_days=(ev - w).days if role == 'case' else np.nan, confirmation_lead_days=(ev - conf).days if role == 'case' and pd.notna(conf) else np.nan, confirm_delay_days=age))
    L = pd.DataFrame(rows)
    L.to_csv(P.RESULTS / 'online_episode_ledger.csv', index=False)
    sr = []
    for case in ['MF07', 'MF12']:
        ev = pd.Timestamp(PR.CASES[case]['event'])
        q = L[(L.case == case) & (L.role == 'case')].sort_values('warning')
        ce = q[q.status == 'confirmed'].head(1)
        sr.append(dict(case=case, first_warning=q.warning.iloc[0] if len(q) else pd.NaT, first_warning_lead=(ev - q.warning.iloc[0]).days if len(q) else np.nan, first_warning_source=q.warning_source.iloc[0] if len(q) else None, confirmed_episode_warning=ce.warning.iloc[0] if len(ce) else pd.NaT, confirmed_episode_warning_lead=(ev - ce.warning.iloc[0]).days if len(ce) else np.nan, confirmed_episode_source=ce.warning_source.iloc[0] if len(ce) else None, confirmation=ce.confirmation.iloc[0] if len(ce) else pd.NaT, confirmation_lead=(ev - ce.confirmation.iloc[0]).days if len(ce) else np.nan, warning_to_confirmation_delay=ce.confirm_delay_days.iloc[0] if len(ce) else np.nan, n_case_episodes=len(q)))
    S = pd.DataFrame(sr)
    S.to_csv(P.RESULTS / 'online_case_episode_summary.csv', index=False)
    print('=== ONLINE CASE EPISODE SUMMARY ===')
    print(S.to_string(index=False))
    print('\nwritten: results/online_episode_ledger.csv, results/online_case_episode_summary.csv')
if __name__ == '__main__':
    main()
