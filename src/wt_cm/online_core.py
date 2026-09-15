from __future__ import annotations
import numpy as np
import pandas as pd
import causal as C
import models as MD

def online_clean(tf: pd.DataFrame, temp_col: str) -> pd.Series:
    ok = tf['P'].between(50, 2100) & (tf['rpm'] > 3) & tf[temp_col].between(5, 90) & tf['T_nac'].between(-10, 60) & tf['T_amb'].between(-20, 50) & (tf['run'] > 540) & (tf['alarm'] < 60) & ~C.causal_flatline_mask(tf[temp_col]) & ~C.causal_flatline_mask(tf['P'])
    full = ok.copy()
    for k in range(1, C.NSTENCIL):
        full &= ok.shift(k, fill_value=False).astype(bool)
    return full.astype(bool)

def filter_resistance(tf: pd.DataFrame) -> pd.Series:
    dp = tf['p_before'] - tf['p_after']
    ok = tf['P'].between(500, 2100) & (tf['rpm'] > 8) & tf['T_oil'].between(20, 90) & dp.between(0.01, 20) & (tf['run'] > 540) & (tf['alarm'] < 60)
    for c in ['p_before', 'p_after', 'T_oil', 'rpm']:
        ok &= ~C.causal_flatline_mask(tf[c])
    omega = tf['rpm'] * 2 * np.pi / 60.0
    return (dp / (MD.phi(tf['T_oil']) * omega)).where(ok)

def filter_trajectory(tf: pd.DataFrame, min_samples: int=200, end_limit=None) -> pd.Series:
    r = filter_resistance(tf).dropna()
    rows = []
    lim = pd.Timestamp(end_limit) if end_limit is not None else None
    for end in C.trailing_grid(tf.index):
        if lim is not None and end > lim:
            break
        seg = r.loc[end - C.WINDOW:end]
        if len(seg) >= min_samples:
            rows.append({'t': end, 'idx': float(seg.median())})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index('t')['idx']
