import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H
import frontier2 as F2
KW, HW, HP, HOR = (3.0, 0.5, 0.5, 180)
FIL = P.load_trajectory('filter_traj_online')
GBM = P.load_trajectory('resid_gbm_adaptive')
THE = P.load_trajectory('thermal_traj_online__VA')

def par(zf, zg, zp, kP, latch):
    state = 'NORMAL'
    sf = sg = sp = 0.0
    warn = None
    src = None
    out = []
    for t in zf.index:
        if state == 'WARNING' and (t - warn).days > HOR:
            out.append(dict(warning=warn, status='expired', confirmation=None, source=src))
            state, sf, sg, sp, warn, src = ('NORMAL', 0.0, 0.0, 0.0, None, None)
        if state == 'WARNING':
            v = zp.get(t, np.nan)
            if np.isfinite(v):
                sp = max(0.0, sp + v - kP)
                if sp > HP:
                    out.append(dict(warning=warn, status='confirmed', confirmation=t, source=src))
                    if latch:
                        return out
                    state, sf, sg, sp, warn, src = ('NORMAL', 0.0, 0.0, 0.0, None, None)
            continue
        hit = []
        vf = zf.get(t, np.nan)
        if np.isfinite(vf):
            sf = max(0.0, sf + vf - KW)
            if sf > HW:
                hit.append('filter')
        vg = zg.get(t, np.nan) if zg is not None else np.nan
        if np.isfinite(vg):
            sg = max(0.0, sg + vg - KW)
            if sg > HW:
                hit.append('GBM')
        if hit:
            src = 'both' if len(hit) == 2 else hit[0]
            warn, state, sf, sg, sp = (t, 'WARNING', 0.0, 0.0, 0.0)
    if warn is not None:
        out.append(dict(warning=warn, status='pending', confirmation=None, source=src))
    return out
rows = []
for case in ['MF07', 'MF12']:
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    z = lambda M: B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
    Zf, Zg, Zp = (z(FIL), z(GBM), z(THE))
    for kP, label in [(1.0, 'k_P=1.0'), (1.5, 'k_P=1.5 control-calibrated'), (2.0, 'k_P=2.0'), (2.5, 'k_P=2.5 rule-selected'), (3.0, 'k_P=3.0 frozen reference')]:
        eps = par(Zf[case].loc[c['monitor_from']:ev], Zg[case].loc[c['monitor_from']:ev], Zp[case].loc[c['monitor_from']:ev], kP, True)
        ce = next((e for e in eps if e['status'] == 'confirmed'), None)
        nw = nc = 0
        expo = 0.0
        for t in D.HEALTHY:
            zf, zg, zp = (Zf[t].loc[c['monitor_from']:], Zg[t].loc[c['monitor_from']:], Zp[t].loc[c['monitor_from']:])
            he = par(zf, zg, zp, kP, False)
            nw += len(he)
            nc += sum((e['status'] == 'confirmed' for e in he))
            expo += H.evaluable_exposure(zf, H.RECORD_END)
        rows.append(dict(case=case, operating_point=label, warn_lead=(ev - eps[0]['warning']).days if eps else None, source=eps[0]['source'] if eps else None, conf_lead=(ev - ce['confirmation']).days if ce else None, warn_eps=nw, conf_eps=nc, exposure=round(expo, 2), far_warn=round(nw / expo, 3), far_conf=round(nc / expo, 3), far_conf_ub95=round(3.0 / expo, 3) if nc == 0 else None))
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'kp_propagation.csv', index=False)
print(d.to_string(index=False))
