import re
import numpy as np
import paths as P
import pandas as pd
import causal as C
import models as MD
import thermal_id as ti
SIGNALS = {'P': 'Grid Production Power Avg', 'rpm': 'Rotor RPM Avg', 'T_oil': 'Gear Oil Temp. Avg', 'p_before': 'Gear Filter BeforeInlinePressureOil', 'p_after': 'Gear Filter AfterInlinePressureOil', 'T_amb': 'Ambient Temp. Avg', 'T_nac': 'Nacelle Temp. Avg', 'v': 'Ambient WindSpeed Avg', 'pitch': 'Blades PitchAngle Avg', 'alarm': 'HourCounters Average AlarmActive Avg', 'run': 'HourCounters Average Run Avg'}

def frame(df, turbine):
    cols = [c for c in df.columns if re.match('MF\\d+_', str(c))]

    def get(sig):
        m = [c for c in cols if c.startswith(turbine + '_') and sig in c]
        if not m:
            raise KeyError(f'{turbine} / {sig}')
        return df[m[0]].astype('float32')
    out = pd.DataFrame(index=df.index)
    for k, s in SIGNALS.items():
        out[k] = get(s)
    out['T_gbb'] = out['T_oil']
    return out

def resistance(tf):
    dp = tf['p_before'] - tf['p_after']
    ok = tf['P'].between(500, 2100) & (tf['rpm'] > 8) & tf['T_oil'].between(20, 90) & dp.between(0.01, 20) & (tf['run'] > 540) & (tf['alarm'] < 60)
    for c in ['p_before', 'p_after', 'T_oil', 'rpm']:
        ok &= ~C.interpolated_mask(tf[c]) & ~C.causal_flatline_mask(tf[c])
    omega = tf['rpm'] * 2 * np.pi / 60.0
    return (dp / (MD.phi(tf['T_oil']) * omega)).where(ok)

def trajectory(tf):
    R = resistance(tf).dropna()
    rows = []
    for end in C.trailing_grid(tf.index):
        seg = R.loc[end - C.WINDOW:end]
        if len(seg) >= 200:
            rows.append({'t': end, 'idx': float(seg.median())})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index('t')['idx']
if __name__ == '__main__':
    import pickle
    import detect_final as D
    import normalise as N
    import protocol as PR
    MD.GAMMA = 0.03
    df = ti.load_raw()
    frames = {t: frame(df, t) for t in ti.TURBINES}
    del df
    M = pd.DataFrame({t: trajectory(frames[t]) for t in ti.TURBINES})
    P.save(M, 'filter_traj')
    amps = [MD.seasonal_amplitude(M[t].loc[:PR.DEV_END]) for t in D.HEALTHY]
    print(f'DEV seasonal p-p {np.nanmedian([a[0] for a in amps]):.3f} | noise {np.nanmedian([a[1] for a in amps]):.3f}')
