import itertools
import pickle
import paths as P
import numpy as np
import pandas as pd
import causal as C
import detect_final as D
import models as MD
import normalise as N
import protocol as PR
import state_machine as SM
RESULTS = P.RESULTS
KS = [2.0, 2.5, 3.0, 3.5, 4.0]
HS = [0.5, 1.0, 2.0, 4.0, 8.0]
HORIZONS = [30, 60, 90, 120, 180, 270, 365]

def detect(Ft, Tt, case, k, h, horizon):
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    Zf = N.rolling_z(Ft, scale_end=PR.SCALE_END[case])
    Zt = N.rolling_z(Tt, scale_end=PR.SCALE_END[case])
    sl = slice(c['monitor_from'], ev)
    sl_ctrl = slice(c['monitor_from'], None)
    eps = SM.run_machine(Zf[case].loc[sl], Zt[case].loc[sl], k, h, h, horizon)
    conf = SM.first_confirmed_episode(eps)
    nw = nc = 0
    exp = 0.0
    for hh in D.HEALTHY:
        zf, zt = (Zf[hh].loc[sl_ctrl], Zt[hh].loc[sl_ctrl])
        he = SM.run_machine(zf, zt, k, h, h, horizon, latch=False)
        nw += len(he)
        nc += sum((1 for e in he if e['status'] == 'confirmed'))
        exp += (zf.index[-1] - zf.index[0]).days / 365.25
    return dict(case=case, k=k, h=h, horizon=horizon, warn_lead=(ev - eps[0]['warning']).days if eps else None, conf_warn_lead=(ev - conf['warning']).days if conf else None, conf_lead=(ev - conf['confirmation']).days if conf else None, conf_delay_days=(conf['confirmation'] - conf['warning']).days if conf else None, fp_warn=nw, fp_conf=nc, exposure_yrs=round(exp, 1), far_warn=round(nw / exp, 3) if exp else np.nan, far_conf=round(nc / exp, 3) if exp else np.nan)

def filter_index(frames, flat_n, dp_lo, agg, normalise=True):
    M = {}
    for t, tf in frames.items():
        dp = tf['p_before'] - tf['p_after']
        om = tf['rpm'] * 2 * np.pi / 60.0
        ok = tf['P'].between(500, 2100) & (tf['rpm'] > 8) & tf['T_oil'].between(20, 90) & (tf['run'] > 540) & (tf['alarm'] < 60)
        ok &= dp.between(dp_lo, 20) if dp_lo is not None else dp.between(-5, 20)
        for c in ['p_before', 'p_after', 'T_oil', 'rpm']:
            ok &= ~C.interpolated_mask(tf[c])
            if flat_n:
                ok &= ~C.causal_flatline_mask(tf[c], flat_n)
        R = (dp / (MD.phi(tf['T_oil']) * om) if normalise else dp).where(ok).dropna()
        rows = [{'t': e, 'idx': float(getattr(R.loc[e - C.WINDOW:e], agg)())} for e in C.trailing_grid(tf.index) if len(R.loc[e - C.WINDOW:e]) >= 200]
        M[t] = pd.DataFrame(rows).set_index('t')['idx'] if rows else pd.Series(dtype=float)
    return pd.DataFrame(M)
if __name__ == '__main__':
    import thermal_id as ti
    import filter_ch as F
    MD.GAMMA = 0.03
    Ft = P.load('filter_traj')
    Tt = P.load('causal_traj')['VA']
    surf = pd.DataFrame([detect(Ft, Tt, c, k, h, 180) for c, k, h in itertools.product(['MF07', 'MF12'], KS, HS)])
    surf.to_csv(RESULTS / 'operating_surface.csv', index=False)
    both = 0
    for k, h in itertools.product(KS, HS):
        rs = [surf[(surf.case == c) & (surf.k == k) & (surf.h == h)].iloc[0] for c in ['MF07', 'MF12']]
        if all((pd.notna(r.conf_lead) and r.fp_conf == 0 for r in rs)):
            both += 1
    for c in ['MF07', 'MF12']:
        s = surf[surf.case == c]
        print(f'{c}: confirmed in {s.conf_lead.notna().sum()}/25 cells; zero confirmed FA in {(s.fp_conf == 0).sum()}/25')
    print(f'BOTH faults confirmed AND zero confirmed FA: {both}/25 cells\n')
    hz = pd.DataFrame([detect(Ft, Tt, c, 3.0, 0.5, z) for c, z in itertools.product(['MF07', 'MF12'], HORIZONS)])
    hz.to_csv(RESULTS / 'horizon_sweep.csv', index=False)
    print('horizon sweep (k=3.0, h=0.5): conf lead')
    for z in HORIZONS:
        a = hz[(hz.horizon == z) & (hz.case == 'MF07')].iloc[0]
        b = hz[(hz.horizon == z) & (hz.case == 'MF12')].iloc[0]
        f = lambda v: 'none' if pd.isna(v) else f'{int(v)}d'
        print(f'  {z:>4} d: MF07 {f(a.conf_lead):>6s} (FA {int(a.fp_conf)})  MF12 {f(b.conf_lead):>6s} (FA {int(b.fp_conf)})')
    df = ti.load_raw()
    frames = {t: F.frame(df, t) for t in ti.TURBINES}
    del df
    variants = [('flat6, dp>0.01 (nominal)', 6, 0.01, 'median'), ('no flatline, dp>0.01', None, 0.01, 'median'), ('flat12, dp>0.01', 12, 0.01, 'median'), ('flat36, dp>0.01', 36, 0.01, 'median'), ('flat6, no dp censoring', 6, None, 'median'), ('no flatline, no censoring', None, None, 'median')]
    rows = []
    print('\npreprocessing sensitivity (k=3.0, h=0.5, 180 d):')
    for lab, fn, lo, agg in variants:
        Mf = filter_index(frames, fn, lo, agg)
        out = {}
        for case in ['MF07', 'MF12']:
            r = detect(Mf, Tt, case, 3.0, 0.5, 180)
            r['variant'] = lab
            rows.append(r)
            out[case] = r
        f = lambda v: 'none' if v is None else f'{v}d'
        print(f"  {lab:28s} MF07 {f(out['MF07']['warn_lead'])}/{f(out['MF07']['conf_lead'])}  MF12 {f(out['MF12']['warn_lead'])}/{f(out['MF12']['conf_lead'])}  FAR_conf {out['MF12']['far_conf']}")
    pd.DataFrame(rows).to_csv(RESULTS / 'preprocessing_sensitivity.csv', index=False)
    print('\nwritten: operating_surface.csv, horizon_sweep.csv, preprocessing_sensitivity.csv')
