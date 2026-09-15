import pickle
import numpy as np
import pandas as pd
import causal as C
import causal_phys as CP
import paths as P
import detect_final as D
import models as MD
import thermal_id as ti
REF = dict(P=1500.0, rpm=14.0, T_amb=10.0, T_nac=25.0)

def coefficient_trajectory(tf, tc='T_gbb', model='VA'):
    ok = C.causal_clean_v2(tf, tc)
    X, y = CP.design(tf, tc, model)
    X, y = (X[ok], y[ok])
    rows = []
    for end in C.trailing_grid(tf.index):
        Xw, yw = (C.window_slice(X, end), C.window_slice(y, end))
        c = MD.fit(Xw, yw)
        if not c:
            continue
        m = Xw.notna().all(axis=1) & yw.notna()
        A = Xw[m].values
        cond = np.linalg.cond(A / np.abs(A).mean(axis=0)) if m.sum() > 10 else np.nan
        rows.append({'t': end, **c, 'cond': cond})
    return pd.DataFrame(rows).set_index('t')

def steady_state_temp(c):
    a = c.get('sink_nac', 0.0)
    if a <= 0:
        return np.nan
    w2 = (REF['rpm'] * 2 * np.pi / 60.0) ** 2
    drive0 = c['P'] * REF['P'] + c['const'] + c.get('Tamb', 0.0) * REF['T_amb']
    T = 55.0
    for _ in range(60):
        Tn = REF['T_nac'] + (drive0 + c.get('visc', 0.0) * MD.phi(T) * w2) / a
        if not np.isfinite(Tn) or abs(Tn) > 500:
            return np.nan
        if abs(Tn - T) < 1e-06:
            return Tn
        T = 0.5 * T + 0.5 * Tn
    return T
if __name__ == '__main__':
    MD.GAMMA = 0.03
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    store, rows = ({}, [])
    for t in ti.TURBINES:
        tf = frames[t]
        co = coefficient_trajectory(tf)
        store[t] = co
        pos = co['sink_nac'] > 0
        tau = 1.0 / co.loc[pos, 'sink_nac'] / 3600.0
        Tss = co[pos].apply(lambda r: steady_state_temp(dict(r)), axis=1)
        ok = C.causal_clean_v2(tf, 'T_gbb')
        near = ok & tf['P'].between(1400, 1600) & tf['rpm'].between(13, 15) & tf['T_amb'].between(5, 15) & tf['T_nac'].between(20, 30)
        obs = tf.loc[near, 'T_gbb'].median()
        n_near = int(near.sum())
        rows.append(dict(turbine=t, n_windows=len(co), frac_a_pos=round(pos.mean(), 3), tau_h_median=round(tau.median(), 2), tau_h_p10=round(tau.quantile(0.1), 2), tau_h_p90=round(tau.quantile(0.9), 2), T_pred_rated=round(Tss.median(), 1), T_obs_rated=round(float(obs), 1), n_matched=n_near, frac_bP_pos=round((co['P'] > 0).mean(), 3), frac_visc_pos=round((co['visc'] > 0).mean(), 3), cond_median=round(co['cond'].median(), 1), corr_a_bP=round(co[['sink_nac', 'P']].corr().iloc[0, 1], 2)))
    P.save(store, 'thermal_coefficients')
    res = pd.DataFrame(rows)
    res.to_csv(P.RESULTS / 'thermal_validation.csv', index=False)
    print(res.to_string(index=False))
    print(f'\nfleet median tau {res.tau_h_median.median():.2f} h | a > 0 in {res.frac_a_pos.min():.1%}-{res.frac_a_pos.max():.1%} of windows')
    print(f'predicted vs observed at rated: {res.T_pred_rated.min():.1f}-{res.T_pred_rated.max():.1f} C predicted, {res.T_obs_rated.min():.1f}-{res.T_obs_rated.max():.1f} C observed')
    print(f'corr(a, b_P) {res.corr_a_bP.min():.2f}-{res.corr_a_bP.max():.2f} -> individual coefficients are NOT independently identified')
