import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM
import baseline_comparison as BC
CASE, HW, HP, HOR = ('MF12', 0.5, 0.5, 180)
c = PR.CASES[CASE]
EV = pd.Timestamp(c['event'])
zof = lambda M: B.rolling_z_v2(M, scale_end=PR.SCALE_END[CASE], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
Zt = zof(P.load_trajectory('thermal_traj_online__VA'))

def machine(zw, zp, kW, kP, latch):
    state = 'NORMAL'
    sw = sp = 0.0
    warn = None
    out = []
    for t in zw.index:
        if state == 'WARNING' and (t - warn).days > HOR:
            out.append(dict(warning=warn, status='expired', confirmation=None))
            state, sw, sp, warn = ('NORMAL', 0.0, 0.0, None)
        if state == 'WARNING':
            v = zp.get(t, np.nan)
            if np.isfinite(v):
                sp = max(0.0, sp + v - kP)
                if sp > HP:
                    out.append(dict(warning=warn, status='confirmed', confirmation=t))
                    if latch:
                        return out
                    state, sw, sp, warn = ('NORMAL', 0.0, 0.0, None)
            continue
        v = zw.get(t, np.nan)
        if np.isfinite(v):
            sw = max(0.0, sw + v - kW)
            if sw > HW:
                warn, state, sw, sp = (t, 'WARNING', 0.0, 0.0)
    if warn is not None:
        out.append(dict(warning=warn, status='pending', confirmation=None))
    return out

def score(Z, kW, kP):
    zw = Z[CASE].loc[c['monitor_from']:EV]
    eps = machine(zw, Zt[CASE].loc[c['monitor_from']:EV], kW, kP, True)
    ce = next((e for e in eps if e['status'] == 'confirmed'), None)
    nw = nc = 0
    expo = 0.0
    for t in D.HEALTHY:
        if t not in Z:
            continue
        z = Z[t].loc[c['monitor_from']:]
        he = machine(z, Zt[t].loc[c['monitor_from']:], kW, kP, False)
        nw += len(he)
        nc += sum((e['status'] == 'confirmed' for e in he))
        expo += H.evaluable_exposure(z, H.RECORD_END)
    return dict(warn_date=eps[0]['warning'].date() if eps else None, warn_lead=(EV - eps[0]['warning']).days if eps else np.nan, conf_lead=(EV - ce['confirmation']).days if ce else np.nan, warn_eps=nw, conf_eps=nc, expo=round(expo, 2), far_warn=round(nw / expo, 3) if expo else np.nan, far_conf=round(nc / expo, 3) if expo else np.nan, far_conf_ub95=round(3.0 / expo, 3) if expo and nc == 0 else None)
if __name__ == '__main__':
    KS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    A = []
    for name, art in BC.ARMS.items():
        f = P.RESULTS / f'{art}.parquet'
        if not f.exists():
            continue
        Z = zof(pd.read_parquet(f))
        if CASE not in Z or Z[CASE].loc[c['monitor_from']:EV].dropna().empty:
            continue
        for kW in KS:
            A.append(dict(arm=name, k_W=kW, k_P=3.0, **score(Z, kW, 3.0)))
    A = pd.DataFrame(A)
    A.to_csv(P.RESULTS / 'frontierA_warning.csv', index=False)
    g = A[A.arm == 'GBM adaptive']
    print('=== DIAGNOSTIC: k_P frozen at 3.0, GBM adaptive ===')
    print(g[['k_W', 'warn_date', 'warn_lead', 'conf_lead', 'far_warn', 'far_conf']].to_string(index=False))
    same = g.groupby('warn_date')['conf_lead'].nunique()
    print('\nwarning dates with >1 distinct confirmation lead:', list(same[same > 1].index) or 'NONE  (stages are separated correctly)')
    print('\n=== Experiment B: confirmation frontier, GBM warning frozen k_W=3.0 ===')
    Zg = zof(P.load_trajectory('resid_gbm_adaptive'))
    Bo = [dict(k_P=kP, **score(Zg, 3.0, kP)) for kP in KS]
    Bo = pd.DataFrame(Bo)
    Bo.to_csv(P.RESULTS / 'frontierB_confirmation.csv', index=False)
    print(Bo[['k_P', 'warn_lead', 'conf_lead', 'conf_eps', 'far_conf', 'far_conf_ub95']].to_string(index=False))
