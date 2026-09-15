import numpy as np
import pandas as pd
import pickle
import thermal_id as ti
HEALTHY = ['MF02', 'MF03', 'MF04', 'MF05', 'MF06', 'MF09', 'MF13']
GAMMA = 0.03
T_VISC_REF = 50.0
REF = dict(P=1500.0, rpm=14.0, T_amb=10.0, T_nac=25.0)

def phi(T):
    return np.exp(-GAMMA * (T - T_VISC_REF))

def design(tf, tc, model):
    T = tf[tc]
    y = ti.derivative(T)
    w2 = (tf['rpm'] * 2 * np.pi / 60.0) ** 2
    X = {'P': tf['P'], 'const': 1.0}
    if 'V' in model:
        X['visc'] = phi(T) * w2
    if 'A' in model:
        X['Tamb'] = tf['T_amb']
    X['sink_nac'] = -(T - tf['T_nac'])
    if 'S' in model:
        X['sink_amb'] = -(T - tf['T_amb'])
    return (pd.DataFrame(X, index=tf.index), y)
MODELS = ['base', 'A', 'V', 'VA', 'VS', 'VAS']

def fit(X, y, ridge=1e-08, min_n=400):
    m = X.notna().all(axis=1) & y.notna()
    Xw, yw = (X[m].values, y[m].values)
    if len(yw) < min_n:
        return None
    sc = np.abs(Xw).mean(axis=0)
    sc[sc == 0] = 1.0
    Xs = Xw / sc
    A = Xs.T @ Xs + ridge * len(yw) * np.eye(Xs.shape[1])
    try:
        coef = np.linalg.solve(A, Xs.T @ yw) / sc
    except np.linalg.LinAlgError:
        return None
    return dict(zip(X.columns, coef))

def index_from_coef(c):
    a_n = c.get('sink_nac', 0.0)
    a_a = c.get('sink_amb', 0.0)
    a = a_n + a_a
    if a <= 1e-09:
        return np.nan
    w2 = (REF['rpm'] * 2 * np.pi / 60.0) ** 2
    drive0 = c['P'] * REF['P'] + c['const'] + c.get('Tamb', 0.0) * REF['T_amb']
    sink0 = a_n * REF['T_nac'] + a_a * REF['T_amb']
    T = 55.0
    for _ in range(60):
        drive = drive0 + c.get('visc', 0.0) * phi(T) * w2
        Tn = (drive + sink0) / a
        if not np.isfinite(Tn) or abs(Tn) > 500:
            return np.nan
        if abs(Tn - T) < 1e-06:
            T = Tn
            break
        T = 0.5 * T + 0.5 * Tn
    return T - REF['T_nac']

def trajectory(tf, tc, model, window='45D', step='15D'):
    ok = ti.clean(tf, tc)
    X, y = design(tf, tc, model)
    X, y = (X[ok], y[ok])
    rows = []
    for s in pd.date_range(tf.index[0], tf.index[-1], freq=step):
        c = fit(X.loc[s:s + pd.Timedelta(window)], y.loc[s:s + pd.Timedelta(window)])
        if c:
            rows.append({'t': s + pd.Timedelta(window) / 2, 'idx': index_from_coef(c)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index('t')['idx']

def seasonal_amplitude(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 40:
        return (np.nan, np.nan)
    d = s.index.dayofyear.values
    X = np.c_[np.ones(len(s)), np.cos(2 * np.pi * d / 365.25), np.sin(2 * np.pi * d / 365.25), np.cos(4 * np.pi * d / 365.25), np.sin(4 * np.pi * d / 365.25)]
    c, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    fitted = X @ c
    return (float(fitted.max() - fitted.min()), float((s.values - fitted).std()))
if __name__ == '__main__':
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    store = {}
    for tc in ['T_gbb', 'T_mbre']:
        print(f'\n===== node {tc} : model selection on healthy turbines only =====')
        print(f"{'model':6s} {'seasonal p-p [K]':>17s} {'resid sd [K]':>13s} {'ratio':>7s}")
        for model in MODELS:
            traj = {t: trajectory(frames[t], tc, model) for t in ti.TURBINES}
            store[tc, model] = pd.DataFrame(traj)
            amps, sds = ([], [])
            for t in HEALTHY:
                a, sd = seasonal_amplitude(traj[t])
                amps.append(a)
                sds.append(sd)
            ma, ms = (np.nanmedian(amps), np.nanmedian(sds))
            print(f'{model:6s} {ma:17.2f} {ms:13.2f} {ma / ms:7.2f}')
    pickle.dump(store, open('model_traj.pkl', 'wb'))
