from pathlib import Path
import numpy as np
import pandas as pd
import kelm_k1 as K
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
BLOCKS = {'2019': '2019-01-01_-_2020-01-01', '2021': '2021-01-01_-_2021-07-01'}
HALF = K.WINDOW / 2

def fit(X, y):
    A = X.values
    c = np.linalg.lstsq(A.T @ A + 1e-08 * np.eye(A.shape[1]), A.T @ y.values, rcond=None)[0]
    co = dict(zip(X.columns, c))
    resid = y.values - A @ c
    a_n = co['sink_nac']
    q = co['const'] + co['P'] * X['P'].values + co['visc'] * X['visc'].values + co['T_amb'] * X['T_amb'].values
    idx = (co['P'] * 1500 + co['const'] + co['visc'] * K.phi(60) * (14 * 2 * np.pi / 60) ** 2 + co['T_amb'] * 10) / a_n if a_n > 0 else np.nan
    return dict(rmse=float(np.sqrt((resid ** 2).mean())), a_nac=a_n, tau_h=1.0 / a_n / 3600 if a_n > 0 else np.nan, q_forcing_mean=float(q.mean()), index=idx, n=len(X))

def legacy_design(f):
    T = f['T']
    y = (T.shift(-3) - T.shift(3)) / (6 * K.DT.total_seconds())
    X = pd.DataFrame(index=f.index)
    X['P'] = f['P']
    X['const'] = 1.0
    X['visc'] = K.phi(f['T']) * (f['rpm'] * 2 * np.pi / 60.0) ** 2
    X['T_amb'] = f['T_amb']
    X['sink_nac'] = -(f['T'] - f['T_nac'])
    return (X, y)

def prepared(f, legacy=False):
    ok = K.clean(f)
    X, y = legacy_design(f) if legacy else K.design(f)
    m = ok & X.notna().all(axis=1) & y.notna()
    return (X[m], y[m])
if __name__ == '__main__':
    a_rows, b_rows = ([], [])
    for turb in range(1, 7):
        for blab, bglob in BLOCKS.items():
            f = K.load_block(turb, bglob)
            Xc, yc = prepared(f)
            Xl, yl = prepared(f, legacy=True)
            grid = pd.date_range(f.index[0] + K.WINDOW, f.index[-1] - HALF, freq=K.STEP)
            for tc in grid:
                lo, hi = (tc - HALF, tc + HALF)
                Xs, ys = (Xc.loc[lo:hi], yc.loc[lo:hi])
                if len(Xs) < K.MIN_N:
                    continue
                cen = fit(Xs, ys)
                Xt, yt = (Xc.loc[tc + HALF - K.WINDOW:tc + HALF], yc.loc[tc + HALF - K.WINDOW:tc + HALF])
                tra = fit(Xt, yt)
                a_rows.append(dict(turbine=f'KWF{turb}', period=blab, t_center=tc, t_available=tc + HALF, lookahead_days=HALF.days + 0.5, n_centered=cen['n'], n_trailing=tra['n'], index_centered=cen['index'], index_trailing=tra['index'], abs_difference=abs(cen['index'] - tra['index']), rmse_centered=cen['rmse'], rmse_trailing=tra['rmse']))
            for end in pd.date_range(f.index[0] + K.WINDOW, f.index[-1], freq=K.STEP):
                Xs, ys = (Xc.loc[end - K.WINDOW:end], yc.loc[end - K.WINDOW:end])
                Xg, yg = (Xl.loc[end - K.WINDOW:end], yl.loc[end - K.WINDOW:end])
                if len(Xs) < K.MIN_N or len(Xg) < K.MIN_N:
                    continue
                c, l = (fit(Xs, ys), fit(Xg, yg))
                b_rows.append(dict(turbine=f'KWF{turb}', period=blab, t=end, rmse_causal=c['rmse'], rmse_legacy=l['rmse'], a_nac_causal=c['a_nac'], a_nac_legacy=l['a_nac'], tau_causal=c['tau_h'], tau_legacy=l['tau_h'], q_causal=c['q_forcing_mean'], q_legacy=l['q_forcing_mean'], index_causal=c['index'], index_legacy=l['index'], index_difference=c['index'] - l['index']))
            print(f'  KWF{turb} {blab} done', flush=True)
    A = pd.DataFrame(a_rows)
    A.to_csv(OUT / 'kelmarsh_k3a_lookahead.csv', index=False)
    B = pd.DataFrame(b_rows)
    B.to_csv(OUT / 'kelmarsh_k3b_derivative.csv', index=False)
    print('\n=== TABLE 1: centred-window availability ===')
    print(f'windows compared        : {len(A)}')
    print(f'nominal timestamp       : t')
    print(f'earliest availability   : t + {HALF.days}.5 d')
    print(f'look-ahead              : {HALF.days}.5 d')
    print(f'same observations?      : n_centered == n_trailing in {int((A.n_centered == A.n_trailing).sum())}/{len(A)} windows')
    print(f'median |index mismatch| : {A.abs_difference.median():.3e}')
    print(f'max    |index mismatch| : {A.abs_difference.max():.3e}')
    print(f'median |rmse mismatch|  : {(A.rmse_centered - A.rmse_trailing).abs().median():.3e}')
    print('\n=== TABLE 2: derivative implementation, fleet ===')
    n = len(B)
    print(f'matched windows           : {n}')
    print(f'median RMSE causal        : {B.rmse_causal.median():.6f}')
    print(f'median RMSE legacy        : {B.rmse_legacy.median():.6f}')
    print(f'legacy RMSE lower in      : {int((B.rmse_legacy < B.rmse_causal).sum())}/{n}')
    print(f'a_nac > 0 causal          : {int((B.a_nac_causal > 0).sum())}/{n}')
    print(f'a_nac > 0 legacy          : {int((B.a_nac_legacy > 0).sum())}/{n}')
    print(f'Q_forcing > 0 causal      : {int((B.q_causal > 0).sum())}/{n}')
    print(f'Q_forcing > 0 legacy      : {int((B.q_legacy > 0).sum())}/{n}')
    print(f'median tau causal (h)     : {B.tau_causal.median():.2f}')
    print(f'median tau legacy (h)     : {B.tau_legacy.median():.2f}')
    print(f'median |index difference| : {B.index_difference.abs().median():.3f}')
    print(f'median index causal       : {B.index_causal.median():.2f}')
    print(f'median index legacy       : {B.index_legacy.median():.2f}')
    print('\nper turbine:')
    g = B.groupby('turbine').agg(n=('t', 'size'), rmse_causal=('rmse_causal', 'median'), rmse_legacy=('rmse_legacy', 'median'), tau_causal=('tau_causal', 'median'), tau_legacy=('tau_legacy', 'median'), idx_causal=('index_causal', 'median'), idx_legacy=('index_legacy', 'median'), med_abs_dI=('index_difference', lambda s: s.abs().median()))
    print(g.round(4).to_string())
