import numpy as np
import paths as P
import pandas as pd
import pickle
import thermal_id as ti
import models as MD
import causal as C
import detect_final as D
import protocol as PR

def design(tf, tc, model):
    T = tf[tc]
    y = C.causal_derivative(T)
    w2 = (tf['rpm'] * 2 * np.pi / 60.0) ** 2
    X = {'P': tf['P'], 'const': 1.0}
    if 'V' in model:
        X['visc'] = MD.phi(T) * w2
    if 'A' in model:
        X['Tamb'] = tf['T_amb']
    X['sink_nac'] = -(T - tf['T_nac'])
    if 'S' in model:
        X['sink_amb'] = -(T - tf['T_amb'])
    return (C.align_regressors(pd.DataFrame(X, index=tf.index)), y)

def trajectory(tf, tc, model):
    ok = C.causal_clean_v2(tf, tc)
    X, y = design(tf, tc, model)
    X, y = (X[ok], y[ok])
    rows = []
    for end in C.trailing_grid(tf.index):
        c = MD.fit(C.window_slice(X, end), C.window_slice(y, end))
        if c:
            rows.append({'t': end, 'idx': MD.index_from_coef(c)})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index('t')['idx']
if __name__ == '__main__':
    MD.GAMMA = 0.03
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    store = {}
    print('=== causal stationarity gate (healthy turbines only) ===')
    print(f"{'model':6s} {'season p-p [K]':>15s} {'resid sd [K]':>13s} {'ratio':>7s}")
    for model in MD.MODELS:
        traj = {t: trajectory(frames[t], 'T_gbb', model) for t in ti.TURBINES}
        store[model] = pd.DataFrame(traj)
        amps, sds = ([], [])
        for t in D.HEALTHY:
            a, sd = MD.seasonal_amplitude(traj[t].loc[:PR.DEV_END])
            amps.append(a)
            sds.append(sd)
        ma, ms = (np.nanmedian(amps), np.nanmedian(sds))
        print(f'{model:6s} {ma:15.2f} {ms:13.2f} {ma / ms:7.2f}')
    P.save(store, 'causal_traj')
    print('\n=== nested structure selection (leave-one-healthy-turbine-out) ===')
    for held in D.HEALTHY:
        others = [t for t in D.HEALTHY if t != held]
        scores = {}
        for model in MD.MODELS:
            amps = [MD.seasonal_amplitude(store[model][t].loc[:PR.DEV_END])[0] for t in others]
            scores[model] = np.nanmedian(amps)
        best = min(scores, key=scores.get)
        print(f'  holding out {held}: selects {best} ({scores[best]:.2f} K)')
