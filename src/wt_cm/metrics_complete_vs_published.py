import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, protocol as PR, detect_final as D, headline as H
import parallel_arch as PA
import complete_vs_published as CV
EARLY = 90
CASES = ['MF07', 'MF12']
F = P.load_trajectory('filter_traj_online')
G = P.load_trajectory('resid_gbm_adaptive')
T = P.load_trajectory('thermal_traj_online__VA')

def run_parallel_k(zf, zg, zt, kw=3.0, kt=3.0, h=0.5, horizon=180, latch=True):
    state, sf, sg, st, warn_t, prov = ('NORMAL', 0.0, 0.0, 0.0, None, None)
    eps = []
    for t in zf.index:
        if state == 'WARNING' and (t - warn_t).days > horizon:
            eps.append(dict(warning=warn_t, status='expired', confirmation=None, end=warn_t + pd.Timedelta(days=horizon), source=prov))
            state, sf, sg, st, warn_t = ('NORMAL', 0.0, 0.0, 0.0, None)
        if state == 'WARNING':
            vt = zt.get(t, np.nan)
            if np.isfinite(vt):
                st = max(0.0, st + vt - kt)
                if st > h:
                    eps.append(dict(warning=warn_t, status='confirmed', confirmation=t, end=t, source=prov))
                    if latch:
                        return eps
                    state, sf, sg, st, warn_t = ('NORMAL', 0.0, 0.0, 0.0, None)
            continue
        hit = []
        vf = zf.get(t, np.nan)
        if np.isfinite(vf):
            sf = max(0.0, sf + vf - kw)
            if sf > h:
                hit.append('filter')
        vg = zg.get(t, np.nan)
        if np.isfinite(vg):
            sg = max(0.0, sg + vg - kw)
            if sg > h:
                hit.append('GBM')
        if hit:
            prov = 'both' if len(hit) == 2 else hit[0]
            warn_t, state, sf, sg, st = (t, 'WARNING', 0.0, 0.0, 0.0)
    if warn_t is not None:
        eps.append(dict(warning=warn_t, status='pending', confirmation=None, end=min(warn_t + pd.Timedelta(days=horizon), H.RECORD_END), source=prov))
    return eps

def cape_block(case, kt=3.0):
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    mf = pd.Timestamp(c['monitor_from'])
    Zf, Zg, Zt = (PA.zof(F, case), PA.zof(G, case).reindex(PA.zof(F, case).index), PA.zof(T, case))
    eps = run_parallel_k(Zf[case].loc[mf:ev], Zg[case].loc[mf:ev], Zt[case].loc[mf:ev], kt=kt)
    ce = next((e for e in eps if e['status'] == 'confirmed'), None)
    n_conf, expo, warn_days, conf_days, early = (0, 0.0, 0.0, 0.0, 0)
    for t in D.HEALTHY:
        zf = Zf[t].loc[mf:]
        he = run_parallel_k(zf, Zg[t].loc[mf:], Zt[t].loc[mf:], kt=kt, latch=False)
        n_conf += sum((e['status'] == 'confirmed' for e in he))
        expo += H.evaluable_exposure(zf, H.RECORD_END)
        warn_days += sum(((min(e['end'], H.RECORD_END) - e['warning']).days for e in he))
        conf = [e['confirmation'] for e in he if e['status'] == 'confirmed']
        if conf and (conf[0] - mf).days < EARLY:
            early += 1
    return dict(lead=(ev - ce['confirmation']).days if ce else np.nan, warn_lead=(ev - eps[0]['warning']).days if eps else np.nan, warn_to_conf=(ce['confirmation'] - ce['warning']).days if ce else np.nan, n=n_conf, expo=expo, alarm_pct=0.0 if n_conf == 0 else np.nan, warn_pct=100 * warn_days / 365.25 / expo, early=early, censored=False)

def base_block(U, case, thr):
    c = PR.CASES[case]
    mf = pd.Timestamp(c['monitor_from'])
    r = CV.baseline_eval(U, case, thr)
    above = days = 0
    early = 0
    for t in D.HEALTHY:
        if t not in U:
            continue
        s = U[t].loc[mf:H.RECORD_END].dropna()
        above += int((s > thr).sum())
        days += len(s)
        ep = CV.bounded_episodes(s, thr)
        if ep and (ep[0] - mf).days < EARLY:
            early += 1
    return dict(lead=r['lead'], censored=r['censored'], n=r['n_ref'], expo=r['expo'], alarm_pct=100 * above / days if days else np.nan, early=early)

def fmt_mtbfa(n, expo):
    if n == 0:
        return f'>{1 / CV.poisson_ci(0, expo)[2]:.1f}'
    return f'{expo / n:.2f}'
rows = []
systems = [('CAPE-WT complete', 'fixed (Table 3)', None)]
for code, name in CV.LIT.items():
    systems += [(name, 'native', code), (name, 'causal calibration', code)]
for name, op, code in systems:
    blocks, rob = ({}, {})
    for case in CASES:
        if code is None:
            b = cape_block(case)
            alt = [cape_block(case, kt=k)['lead'] for k in (2.5, 3.5)] + [b['lead']]
        else:
            M = P.load_trajectory(f'published_{case}__{code}')
            U = CV.standardised_stat(code, M, case)
            if op == 'native':
                thr = CV.NATIVE_THR[code]
            else:
                lo, hi, _ = CV.CAL_WINDOW[case]
                thr = float(pd.concat([U[t].loc[lo:hi] for t in D.HEALTHY if t in U]).dropna().max())
            b = base_block(U, case, thr)
            alt = [CV.baseline_eval(U, case, thr * f)['lead'] for f in (0.83, 1.17)] + [b['lead']]
        blocks[case] = b
        alt = [a for a in alt if np.isfinite(a)]
        rob[case] = f'{int(min(alt))}-{int(max(alt))}' if alt else 'no alarm'
    det = [np.isfinite(blocks[c]['lead']) for c in CASES]
    deployable = all((np.isfinite(blocks[c]['lead']) and (not blocks[c]['censored']) and (blocks[c]['n'] / blocks[c]['expo'] <= 0.2) for c in CASES))
    row = dict(system=name, operating_point=op, detected=f'{sum(det)}/2' + (' (censored)' if any((blocks[c]['censored'] for c in CASES)) else ''), deployable='yes' if deployable else 'no')
    for c in CASES:
        b, L = (blocks[c], CV.LABEL[c])
        row.update({f'{L}_lead_d': ('>=' if b['censored'] else '') + (str(int(b['lead'])) if np.isfinite(b['lead']) else 'none'), f'{L}_ref_alarms': b['n'], f'{L}_expo_ty': round(b['expo'], 2), f'{L}_MTBFA_ty': fmt_mtbfa(b['n'], b['expo']), f'{L}_alarms_per_det': b['n'] if np.isfinite(b['lead']) else 'n/a', f'{L}_time_in_alarm_pct': round(b['alarm_pct'], 1), f'{L}_warning_time_pct': round(b['warn_pct'], 1) if 'warn_pct' in b else '', f'{L}_early_alarm_share': f"{b['early']}/7", f'{L}_robust_lead_range_d': rob[c], f'{L}_warn_to_conf_d': int(b['warn_to_conf']) if 'warn_to_conf' in b and np.isfinite(b['warn_to_conf']) else ''})
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'metrics_complete_vs_published.csv', index=False)
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', 50)
print(d.T.to_string())
