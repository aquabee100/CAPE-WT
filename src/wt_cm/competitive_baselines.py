import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import paths as P
import causal as C
import protocol as PR
import thermal_id as ti
import online_2x2 as O
TC, FEATS = (O.TC, O.FEATS)
HP = json.load(open(P.RESULTS / 'selected_hyperparameters.json'))
MLP_KW = dict(hidden_layer_sizes=tuple(HP['mlp']['hidden_layer_sizes']), alpha=HP['mlp']['alpha'], max_iter=HP['mlp_max_iter'], early_stopping=True, random_state=0)
NPC = HP['pca_components']
LOOKBACK = O.LOOKBACK
STRIDE = O.STRIDE

def _frames(tf):
    ok = O.online_clean(tf)
    X, y = (tf.loc[ok, FEATS], tf.loc[ok, TC])
    m = X.notna().all(axis=1) & y.notna()
    return (X[m], y[m])

def _score(tf, fit_fn, adaptive):
    X, y = _frames(tf)
    rows = []
    if not adaptive:
        tr = X.index <= pd.Timestamp(PR.DEV_END)
        if tr.sum() < 2000:
            return pd.Series(dtype=float)
        model = fit_fn(X[tr], y[tr])
    for end in C.trailing_grid(tf.index):
        if adaptive:
            lo, hi = (end - C.WINDOW - LOOKBACK, end - C.WINDOW)
            Xt, yt = (X.loc[lo:hi], y.loc[lo:hi])
            if len(Xt) < 2000 or hi - Xt.index[0] < LOOKBACK * 0.9:
                continue
            model = fit_fn(Xt, yt)
        Xs, ys = (X.loc[end - C.WINDOW:end], y.loc[end - C.WINDOW:end])
        if len(Xs) < 400:
            continue
        rows.append({'t': end, 'idx': float(model(Xs, ys))})
    return pd.DataFrame(rows).set_index('t')['idx'] if rows else pd.Series(dtype=float)

def linear_fit(Xt, yt):
    A = np.c_[np.ones(len(Xt)), Xt.values]
    c = np.linalg.lstsq(A.T @ A + 1e-08 * np.eye(A.shape[1]), A.T @ yt.values, rcond=None)[0]
    return lambda Xs, ys: (ys.values - np.c_[np.ones(len(Xs)), Xs.values] @ c).mean()

def mlp_fit(Xt, yt):
    sc = StandardScaler().fit(Xt.values)
    m = MLPRegressor(**MLP_KW).fit(sc.transform(Xt.values[::STRIDE]), yt.values[::STRIDE])
    return lambda Xs, ys: (ys.values - m.predict(sc.transform(Xs.values))).mean()

def pca_fit(Xt, yt):
    Z = np.c_[Xt.values, yt.values]
    sc = StandardScaler().fit(Z)
    p = PCA(n_components=NPC).fit(sc.transform(Z))

    def score(Xs, ys):
        Zs = sc.transform(np.c_[Xs.values, ys.values])
        e = np.sqrt(((Zs - p.inverse_transform(p.transform(Zs))) ** 2).mean(axis=1))
        return float(e.mean())
    return score
ARMS = {'resid_lin_frozen': lambda tf: _score(tf, linear_fit, False), 'resid_lin_adaptive': lambda tf: _score(tf, linear_fit, True), 'resid_pca_frozen': lambda tf: _score(tf, pca_fit, False), 'resid_pca_adaptive': lambda tf: _score(tf, pca_fit, True), 'resid_mlp_frozen': lambda tf: _score(tf, mlp_fit, False), 'resid_mlp_adaptive': lambda tf: _score(tf, mlp_fit, True)}
if __name__ == '__main__':
    import sys
    only = sys.argv[1:] or list(ARMS)
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    for name in only:
        ck = P.RESULTS / f'{name}.partial.parquet'
        out = {}
        if ck.exists():
            prev = pd.read_parquet(ck)
            out = {c: prev[c].dropna() for c in prev.columns}
            print(f'resuming {name}: {len(out)}/9 turbines', flush=True)
        for t, tf in frames.items():
            if t in out:
                continue
            s = ARMS[name](tf)
            if len(s):
                out[t] = s
            pd.DataFrame(out).to_parquet(ck)
            print(f'  {name} {t} [{len(out)}/9]', flush=True)
        M = pd.DataFrame(out)
        M.to_parquet(P.RESULTS / f'{name}.parquet')
        ck.unlink(missing_ok=True)
        print(f'{name}: {M.shape}', flush=True)
