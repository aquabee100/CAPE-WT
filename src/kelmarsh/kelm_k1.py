import re
from pathlib import Path
import numpy as np
import pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
HEADER_ROW = 9
DT = pd.Timedelta('10min')
WINDOW = pd.Timedelta('45D')
STEP = pd.Timedelta('15D')
NSTENCIL = 2
GAMMA = 0.03
T_VISC_REF = 50.0
MIN_N = 400
MAP = {'T': 'Gear oil temperature (°C)', 'P': 'Power (kW)', 'rpm': 'Rotor speed (RPM)', 'T_nac': 'Nacelle temperature (°C)', 'T_amb': 'Nacelle ambient temperature (°C)'}

def phi(T):
    return np.exp(-GAMMA * (T - T_VISC_REF))

def load_block(turb, period_glob):
    f = list(RAW.glob(f'Turbine_Data_Kelmarsh_{turb}_{period_glob}*.csv'))[0]
    df = pd.read_csv(f, skiprows=HEADER_ROW, low_memory=False)
    idx = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    out = pd.DataFrame(index=idx)
    for k, col in MAP.items():
        out[k] = pd.to_numeric(df[col], errors='coerce').values
    return out[out.index.notna()].sort_index()

def causal_derivative(s):
    return (s - s.shift(NSTENCIL)) / (NSTENCIL * DT.total_seconds())

def align(df):
    return df.rolling(NSTENCIL + 1, min_periods=NSTENCIL + 1).mean()

def causal_flatline(s, n=6):
    d = s.diff().abs()
    return d.rolling(n, min_periods=n).max() < 1e-09

def clean(f):
    ok = f['P'].between(50, 2200) & (f['rpm'] > 3) & f['T'].between(-5, 120) & f['T_nac'].between(-20, 70) & f['T_amb'].between(-25, 55) & ~causal_flatline(f['T']).fillna(False) & ~causal_flatline(f['P']).fillna(False)
    full = ok.copy()
    for k in range(1, NSTENCIL + 1):
        full &= ok.shift(k, fill_value=False).astype(bool)
    return full

def design(f):
    y = causal_derivative(f['T'])
    a = align(f)
    X = pd.DataFrame(index=f.index)
    X['P'] = a['P']
    X['const'] = 1.0
    X['visc'] = phi(a['T']) * (a['rpm'] * 2 * np.pi / 60.0) ** 2
    X['T_amb'] = a['T_amb']
    X['sink_nac'] = -(a['T'] - a['T_nac'])
    return (X, y)

def fit_windows(f):
    ok = clean(f)
    X, y = design(f)
    m = ok & X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    rows = []
    if not len(X):
        return pd.DataFrame()
    grid = pd.date_range(f.index[0] + WINDOW, f.index[-1], freq=STEP)
    for end in grid:
        Xs, ys = (X.loc[end - WINDOW:end], y.loc[end - WINDOW:end])
        if len(Xs) < MIN_N:
            continue
        A = Xs.values
        c = np.linalg.lstsq(A.T @ A + 1e-08 * np.eye(A.shape[1]), A.T @ ys.values, rcond=None)[0]
        resid = ys.values - A @ c
        co = dict(zip(Xs.columns, c))
        a_n = co['sink_nac']
        rows.append(dict(t=end, rmse=float(np.sqrt((resid ** 2).mean())), b_P=co['P'], b_V=co['visc'], b_A=co['T_amb'], a_nac=a_n, n=len(Xs), tau_h=1.0 / a_n / 3600 if a_n > 0 else np.nan, index=(co['P'] * 1500 + co['const'] + co['visc'] * phi(60) * (14 * 2 * np.pi / 60) ** 2 + co['T_amb'] * 10) / a_n if a_n > 0 else np.nan))
    return pd.DataFrame(rows)
if __name__ == '__main__':
    BLOCKS = {'2019': '2019-01-01_-_2020-01-01', '2021': '2021-01-01_-_2021-07-01'}
    summary, allcoef = ([], [])
    for turb in range(1, 7):
        for blab, bglob in BLOCKS.items():
            f = load_block(turb, bglob)
            ok = clean(f)
            r = fit_windows(f)
            if len(r):
                r['turbine'], r['period'] = (f'KWF{turb}', blab)
                allcoef.append(r)
            summary.append(dict(turbine=f'KWF{turb}', period=blab, n_raw=len(f), n_valid=int(ok.sum()), valid_fraction=round(float(ok.mean()), 3), n_fitted_windows=len(r), median_rmse=round(float(r['rmse'].median()), 6) if len(r) else np.nan, median_b_P=round(float(r['b_P'].median()), 9) if len(r) else np.nan, median_a_nac=round(float(r['a_nac'].median()), 8) if len(r) else np.nan, frac_a_nac_positive=round(float((r['a_nac'] > 0).mean()), 3) if len(r) else np.nan, frac_b_P_positive=round(float((r['b_P'] > 0).mean()), 3) if len(r) else np.nan, median_tau_h=round(float(r['tau_h'].median()), 2) if len(r) else np.nan, median_index=round(float(r['index'].median()), 2) if len(r) else np.nan, index_sd=round(float(r['index'].std()), 2) if len(r) else np.nan))
            print(f'  KWF{turb} {blab}: {len(r)} windows', flush=True)
    s = pd.DataFrame(summary)
    s.to_csv(OUT / 'kelmarsh_k1_summary.csv', index=False)
    if allcoef:
        pd.concat(allcoef).to_csv(OUT / 'kelmarsh_k1_coefficients.csv', index=False)
    print('\n=== K1 physical-plausibility summary ===')
    print(s.to_string(index=False))
