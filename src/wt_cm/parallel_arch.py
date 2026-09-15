import numpy as np
import pandas as pd
import paths as P
import detect_final as D
import protocol as PR
import baseline_rule as B
import headline as H
import state_machine as SM
K, HTH, HOR = (3.0, 0.5, 180)

def run_parallel(zf, zg, zt, k=K, h=HTH, horizon_days=HOR, latch=True):
    state = 'NORMAL'
    sf = sg = st = 0.0
    warn_t, prov = (None, None)
    episodes = []
    for t in zf.index:
        if state == 'WARNING' and (t - warn_t).days > horizon_days:
            episodes.append(dict(warning=warn_t, status='expired', confirmation=None, source=prov))
            state, sf, sg, st, warn_t, prov = ('NORMAL', 0.0, 0.0, 0.0, None, None)
        if state == 'WARNING':
            vt = zt.get(t, np.nan)
            if np.isfinite(vt):
                st = max(0.0, st + vt - k)
                if st > h:
                    episodes.append(dict(warning=warn_t, status='confirmed', confirmation=t, source=prov))
                    if latch:
                        return episodes
                    state, sf, sg, st, warn_t, prov = ('NORMAL', 0.0, 0.0, 0.0, None, None)
            continue
        hit = []
        vf = zf.get(t, np.nan)
        if np.isfinite(vf):
            sf = max(0.0, sf + vf - k)
            if sf > h:
                hit.append('filter')
        vg = zg.get(t, np.nan) if zg is not None else np.nan
        if np.isfinite(vg):
            sg = max(0.0, sg + vg - k)
            if sg > h:
                hit.append('GBM')
        if hit:
            prov = 'both' if len(hit) == 2 else hit[0]
            warn_t, state, sf, sg, st = (t, 'WARNING', 0.0, 0.0, 0.0)
    if warn_t is not None:
        episodes.append(dict(warning=warn_t, status='pending', confirmation=None, source=prov))
    return episodes

def zof(M, case):
    return B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
if __name__ == '__main__':
    FIL = P.load_trajectory('filter_traj_online')
    GBM = P.load_trajectory('resid_gbm_adaptive')
    THE = P.load_trajectory('thermal_traj_online__VA')
    rows = []
    for case in ['MF07', 'MF12']:
        c = PR.CASES[case]
        ev = pd.Timestamp(c['event'])
        Zf, Zg, Zt = (zof(FIL, case), zof(GBM, case), zof(THE, case))
        sl, slc = (slice(c['monitor_from'], ev), slice(c['monitor_from'], None))
        for arch in ['filter only', 'parallel filter OR GBM']:
            par = arch.startswith('parallel')
            zg_case = Zg[case].loc[sl] if par else None
            if par:
                eps = run_parallel(Zf[case].loc[sl], zg_case, Zt[case].loc[sl])
            else:
                eps = SM.run_machine(Zf[case].loc[sl], Zt[case].loc[sl], K, HTH, HTH, HOR)
            ce = next((e for e in eps if e['status'] == 'confirmed'), None)
            nw = nc = 0
            expo = 0.0
            for t in D.HEALTHY:
                zf, zt = (Zf[t].loc[slc], Zt[t].loc[slc])
                zg = Zg[t].loc[slc] if par else None
                he = run_parallel(zf, zg, zt, latch=False) if par else SM.run_machine(zf, zt, K, HTH, HTH, HOR, latch=False)
                nw += len([e for e in he])
                nc += len([e for e in he if e['status'] == 'confirmed'])
                expo += H.evaluable_exposure(zf, H.RECORD_END)
            gbm_avail = int(Zg[case].loc[sl].notna().sum()) if par else None
            rows.append(dict(case=case, architecture=arch, warn_lead=(ev - eps[0]['warning']).days if eps else None, first_warning_source=eps[0].get('source', 'filter') if eps else None, confirmed_episode_warn_lead=(ev - ce['warning']).days if ce else None, confirmed_episode_source=ce.get('source') if ce else None, conf_lead=(ev - ce['confirmation']).days if ce else None, confirm_delay=(ce['confirmation'] - ce['warning']).days if ce else None, warn_episodes=nw, confirmed_alarms=nc, exposure_yr=round(expo, 2), far_warn=round(nw / expo, 3) if expo else np.nan, far_conf=round(nc / expo, 3) if expo else np.nan, far_conf_upper95=round(3.0 / expo, 3) if expo and nc == 0 else None, gbm_windows_available_pre_event=gbm_avail))
    r = pd.DataFrame(rows)
    r.to_csv(P.RESULTS / 'parallel_architecture.csv', index=False)
    print(r.to_string(index=False))
    print('\n=== control warning provenance, parallel architecture ===')
    for case in ['MF07', 'MF12']:
        c = PR.CASES[case]
        Zf, Zg, Zt = (zof(FIL, case), zof(GBM, case), zof(THE, case))
        src = []
        for t in D.HEALTHY:
            for e in run_parallel(Zf[t].loc[c['monitor_from']:], Zg[t].loc[c['monitor_from']:], Zt[t].loc[c['monitor_from']:], latch=False):
                src.append(e['source'])
        s = pd.Series(src).value_counts()
        print(f'  {case}: ' + (', '.join((f'{k} {v}' for k, v in s.items())) if len(s) else 'none'))
