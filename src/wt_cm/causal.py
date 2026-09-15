import numpy as np
import pandas as pd
import thermal_id as ti
DT = 600.0
NSTENCIL = 7
WINDOW = pd.Timedelta('45D')
STEP = pd.Timedelta('15D')
_k = np.arange(NSTENCIL)
_tk = -_k * DT
_W = (_tk - _tk.mean()) / ((_tk - _tk.mean()) ** 2).sum()

def causal_derivative(s):
    out = None
    for k in range(NSTENCIL):
        term = _W[k] * s.shift(k)
        out = term if out is None else out + term
    return out

def causal_clean(tf, temp_col):
    ok = ti.clean(tf, temp_col)
    full = ok.copy()
    for k in range(1, NSTENCIL):
        full &= ok.shift(k, fill_value=False).astype(bool)
    return full

def trailing_grid(index):
    return pd.date_range(index[0] + WINDOW, index[-1], freq=STEP)

def window_slice(s, end):
    return s.loc[end - WINDOW:end]

def align_regressors(X):
    Xa = X.rolling(NSTENCIL).mean()
    if 'const' in Xa.columns:
        Xa['const'] = 1.0
    return Xa

def causal_flatline_mask(s, n=6):
    same = s.diff() == 0
    out = same.copy()
    for k in range(1, n - 1):
        out &= same.shift(k, fill_value=False).astype(bool)
    return out.fillna(False).astype(bool)

def interpolated_mask(x, minrun=5):
    d1 = x.diff()
    d2 = d1.diff()
    lin = (d2.abs() < 1e-09) & (d1.abs() > 1e-09)
    run = (~lin).cumsum()
    L = lin.groupby(run).transform('size').where(lin, 0)
    return (L >= minrun).fillna(False).astype(bool)
AUDIT_COLS = ['T_gbb', 'T_oil', 'P', 'rpm', 'v', 'T_amb', 'T_nac', 'pitch']

def causal_clean_v2(tf, temp_col):
    ok = tf['P'].between(50, 2100) & (tf['rpm'] > 3) & tf[temp_col].between(5, 90) & tf['T_nac'].between(-10, 60) & tf['T_amb'].between(-20, 50) & (tf['run'] > 540) & (tf['alarm'] < 60) & ~causal_flatline_mask(tf[temp_col]) & ~causal_flatline_mask(tf['P'])
    for c in AUDIT_COLS:
        if c in tf.columns:
            ok &= ~interpolated_mask(tf[c])
    full = ok.copy()
    for k in range(1, NSTENCIL):
        full &= ok.shift(k, fill_value=False).astype(bool)
    return full
