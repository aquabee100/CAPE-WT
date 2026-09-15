import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
import paths as P, causal as C, causal_phys as CP, models as MD
import online_core as OC
import protocol as PR, thermal_id as ti
TC = 'T_gbb'
LOOKBACK = pd.Timedelta('365D')
GBM_PARAMS = dict(max_iter=300, max_depth=6, learning_rate=0.05)
FEATS = ['P', 'rpm', 'v', 'T_amb', 'T_nac', 'pitch']
STRIDE = 3

def online_clean(tf, tc=TC):
    return OC.online_clean(tf, tc)

def _ls(X, y, ridge=1e-08):
    A = X.values
    return np.linalg.lstsq(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y.values, rcond=None)[0]

def phys_coef_index(tf, end_limit=None):
    ok = online_clean(tf)
    X, y = CP.design(tf, TC, 'VA')
    X, y = (X[ok], y[ok])
    rows = []
    lim = pd.Timestamp(end_limit) if end_limit is not None else None
    for end in C.trailing_grid(tf.index):
        if lim is not None and end > lim:
            break
        c = MD.fit(C.window_slice(X, end), C.window_slice(y, end))
        if c:
            rows.append({'t': end, 'idx': MD.index_from_coef(c)})
    return pd.DataFrame(rows).set_index('t')['idx'] if rows else pd.Series(dtype=float)

def phys_residual(tf, adaptive):
    ok = online_clean(tf)
    X, y = CP.design(tf, TC, 'VA')
    m = ok & X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    rows = []
    if not adaptive:
        tr = X.index <= pd.Timestamp(PR.DEV_END)
        if tr.sum() < 2000:
            return pd.Series(dtype=float)
        c = _ls(X[tr], y[tr])
    for end in C.trailing_grid(tf.index):
        if adaptive:
            lo, hi = (end - C.WINDOW - LOOKBACK, end - C.WINDOW)
            Xt, yt = (X.loc[lo:hi], y.loc[lo:hi])
            if len(Xt) < 2000 or hi - Xt.index[0] < LOOKBACK * 0.9:
                continue
            c = _ls(Xt, yt)
        seg = slice(end - C.WINDOW, end)
        Xs, ys = (X.loc[seg], y.loc[seg])
        if len(Xs) < 400:
            continue
        rows.append({'t': end, 'idx': float((ys.values - Xs.values @ c).mean())})
    return pd.DataFrame(rows).set_index('t')['idx'] if rows else pd.Series(dtype=float)

def gbm_residual(tf, adaptive, end_limit=None):
    ok = online_clean(tf)
    X, y = (tf.loc[ok, FEATS], tf.loc[ok, TC])
    m = X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    rows = []
    if not adaptive:
        tr = X.index <= pd.Timestamp(PR.DEV_END)
        if tr.sum() < 2000:
            return pd.Series(dtype=float)
        mdl = HistGradientBoostingRegressor(random_state=0, **GBM_PARAMS).fit(X[tr].iloc[::STRIDE], y[tr].iloc[::STRIDE])
    lim = pd.Timestamp(end_limit) if end_limit is not None else None
    for end in C.trailing_grid(tf.index):
        if lim is not None and end > lim:
            break
        if adaptive:
            lo, hi = (end - C.WINDOW - LOOKBACK, end - C.WINDOW)
            Xt, yt = (X.loc[lo:hi], y.loc[lo:hi])
            if len(Xt) < 2000 or hi - Xt.index[0] < LOOKBACK * 0.9:
                continue
            mdl = HistGradientBoostingRegressor(random_state=0, **GBM_PARAMS).fit(Xt.iloc[::STRIDE], yt.iloc[::STRIDE])
        seg = slice(end - C.WINDOW, end)
        Xs, ys = (X.loc[seg], y.loc[seg])
        if len(Xs) < 400:
            continue
        rows.append({'t': end, 'idx': float((ys.values - mdl.predict(Xs)).mean())})
    return pd.DataFrame(rows).set_index('t')['idx'] if rows else pd.Series(dtype=float)

def gbm_score_at(tf, end):
    end = pd.Timestamp(end)
    ok = online_clean(tf)
    X, y = (tf.loc[ok, FEATS], tf.loc[ok, TC])
    m = X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    lo, hi = (end - C.WINDOW - LOOKBACK, end - C.WINDOW)
    Xt, yt = (X.loc[lo:hi], y.loc[lo:hi])
    if len(Xt) < 2000 or hi - Xt.index[0] < LOOKBACK * 0.9:
        return np.nan
    mdl = HistGradientBoostingRegressor(random_state=0, **GBM_PARAMS).fit(Xt.iloc[::STRIDE], yt.iloc[::STRIDE])
    Xs, ys = (X.loc[end - C.WINDOW:end], y.loc[end - C.WINDOW:end])
    if len(Xs) < 400:
        return np.nan
    return float((ys.values - mdl.predict(Xs)).mean())
BUILDERS = {'thermal_traj_online__VA': phys_coef_index, 'resid_phys_frozen': lambda tf: phys_residual(tf, False), 'resid_phys_adaptive': lambda tf: phys_residual(tf, True), 'resid_gbm_frozen': lambda tf: gbm_residual(tf, False), 'resid_gbm_adaptive': lambda tf: gbm_residual(tf, True)}
if __name__ == '__main__':
    import sys
    only = sys.argv[1:] or list(BUILDERS)
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    for name in only:
        out = {}
        for t, tf in frames.items():
            s = BUILDERS[name](tf)
            if len(s):
                out[t] = s
        M = pd.DataFrame(out)
        P.save_trajectory(M, name)
        print(f'{name}: {M.shape}  windows {M.index[0].date()}..{M.index[-1].date()}', flush=True)
