import re
import numpy as np
import pandas as pd
PKL = 'data/MANF2017-2022.pkl'
TURBINES = ['MF02', 'MF03', 'MF04', 'MF05', 'MF06', 'MF07', 'MF09', 'MF12', 'MF13']
DT = 600.0

def load_raw():
    df = pd.read_pickle(PKL)
    df.index = pd.to_datetime(df.index)
    return df

def _find(cols, turbine, signal):
    m = [c for c in cols if c.startswith(turbine + '_') and signal in c]
    if not m:
        raise KeyError(f'{turbine} / {signal}')
    return m[0]
SIGNALS = {'P': 'Grid Production Power Avg', 'rpm': 'Rotor RPM Avg', 'pitch': 'Blades PitchAngle Avg', 'v': 'Ambient WindSpeed Avg', 'T_amb': 'Ambient Temp. Avg', 'T_nac': 'Nacelle Temp. Avg', 'T_oil': 'Gear Oil Temp. Avg', 'T_gbb': 'Gear Bearing Temp. Avg', 'T_mbre': 'Gear_MainBear_TempRE_Avg', 'T_mbnre': 'Gear_MainBear_TempNRE_Avg', 'alarm': 'HourCounters Average AlarmActive Avg', 'run': 'HourCounters Average Run Avg'}

def turbine_frame(df, turbine):
    cols = [c for c in df.columns if re.match('MF\\d+_', str(c))]
    out = pd.DataFrame(index=df.index)
    for name, sig in SIGNALS.items():
        out[name] = df[_find(cols, turbine, sig)].astype('float32')
    return out

def flatline_mask(s, n=6):
    same = s.diff() == 0
    grp = (~same).cumsum()
    return same.groupby(grp).transform('size').where(same, 0) >= n

def clean(tf, temp_col='T_oil'):
    ok = tf['P'].between(50, 2100) & (tf['rpm'] > 3) & tf[temp_col].between(5, 90) & tf['T_nac'].between(-10, 60) & tf['T_amb'].between(-20, 50) & (tf['run'] > 540) & (tf['alarm'] < 60) & ~flatline_mask(tf[temp_col]) & ~flatline_mask(tf['P'])
    return ok

def derivative(s, half=3):
    return (s.shift(-half) - s.shift(half)) / (2 * half * DT)

def build_design(tf, temp_col='T_oil'):
    T = tf[temp_col]
    dTdt = derivative(T)
    X = pd.DataFrame({'b1_P': tf['P'], 'b2_w2': (tf['rpm'] * 2 * np.pi / 60.0) ** 2, 'b0': 1.0, 'a_dT': -(T - tf['T_nac'])}, index=tf.index)
    return (X, dTdt)

def fit_window(X, y, ridge=1e-08, min_n=400):
    m = X.notna().all(axis=1) & y.notna()
    Xw, yw = (X[m].values, y[m].values)
    if len(yw) < min_n:
        return None
    sc = np.abs(Xw).mean(axis=0)
    sc[sc == 0] = 1.0
    Xs = Xw / sc
    A = Xs.T @ Xs + ridge * len(yw) * np.eye(Xs.shape[1])
    coef = np.linalg.solve(A, Xs.T @ yw) / sc
    resid = yw - Xw @ coef
    out = {name.split('_')[0]: c for name, c in zip(X.columns, coef)}
    out['n'] = len(yw)
    out['rmse'] = float(np.sqrt((resid ** 2).mean()))
    return out

def identify_trajectory(tf, ok, temp_col='T_oil', window='30D', step='15D'):
    X, y = build_design(tf, temp_col)
    X, y = (X[ok], y[ok])
    starts = pd.date_range(tf.index[0], tf.index[-1], freq=step)
    rows = []
    for s in starts:
        e = s + pd.Timedelta(window)
        r = fit_window(X.loc[s:e], y.loc[s:e])
        if r:
            r['t'] = s + pd.Timedelta(window) / 2
            rows.append(r)
    out = pd.DataFrame(rows).set_index('t')
    out['tau_h'] = 1.0 / out['a'] / 3600.0
    out['gain'] = out['b1'] / out['a']
    return out
