import numpy as np
import paths as P
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import thermal_id as ti
import causal as C
import detect_final as D
import protocol as PR
warnings.filterwarnings('ignore')
TC = 'T_gbb'
FEATS = ['P', 'rpm', 'v', 'T_amb', 'T_nac', 'pitch']
GBM_GRID = [dict(max_iter=300, max_depth=d, learning_rate=0.05) for d in (4, 6, 10)]
MLP_GRID = [dict(hidden_layer_sizes=h, alpha=0.0001) for h in ((32, 16), (64, 32))]
PCA_GRID = [2, 3, 4, 5]

def window_mean(s, grid):
    return pd.Series([s.loc[e - C.WINDOW:e].mean() if len(s.loc[e - C.WINDOW:e]) > 200 else np.nan for e in grid], index=grid)

def grouped_cv(make, X, y, groups, scale=False):
    sc_all = []
    for tr, te in GroupKFold(n_splits=7).split(X, y, groups):
        Xtr, Xte = (X[tr], X[te])
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = (sc.transform(Xtr), sc.transform(Xte))
        m = make()
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        sc_all.append(1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(np.mean(sc_all))

def tune(frames):
    Xs, ys, gs = ([], [], [])
    for i, t in enumerate(D.HEALTHY):
        tf = frames[t]
        ok = C.causal_clean_v2(tf, TC)
        X, y = (tf.loc[ok, FEATS], tf.loc[ok, TC])
        m = X.index <= PR.DEV_END
        Xs.append(X[m].values[::20])
        ys.append(y[m].values[::20])
        gs.append(np.full(len(X[m].values[::20]), i))
    X, y, g = (np.vstack(Xs), np.concatenate(ys), np.concatenate(gs))
    best_g = max(GBM_GRID, key=lambda p: grouped_cv(lambda: HistGradientBoostingRegressor(random_state=0, **p), X, y, g))
    best_m = max(MLP_GRID, key=lambda p: grouped_cv(lambda: MLPRegressor(max_iter=200, early_stopping=True, random_state=0, **p), X, y, g, scale=True))
    XY = np.c_[X, y]
    errs = {}
    for n in PCA_GRID:
        fold = []
        for tr, te in GroupKFold(n_splits=7).split(XY, groups=g):
            sc = StandardScaler().fit(XY[tr])
            Xtr, Xte = (sc.transform(XY[tr]), sc.transform(XY[te]))
            p = PCA(n_components=n).fit(Xtr)
            fold.append(float(np.sqrt(((Xte - p.inverse_transform(p.transform(Xte))) ** 2).mean())))
        errs[n] = float(np.mean(fold))
    best_p = min(PCA_GRID, key=lambda n: errs[n] + 0.02 * n)
    print('  pca held-out recon error:', {k: round(v, 4) for k, v in errs.items()})
    print('grouped-CV tuning (healthy turbines, train period only):')
    print('  gbm:', best_g, '| mlp:', best_m, '| pca:', best_p)
    return (best_g, best_m, best_p)

def static_index(tf, ok, grid):
    dT = (tf[TC] - tf['T_nac'])[ok]
    P = tf['P'][ok]
    out = []
    for e in grid:
        yy, xx = (dT.loc[e - C.WINDOW:e], P.loc[e - C.WINDOW:e])
        m = yy.notna() & xx.notna()
        if m.sum() < 400:
            out.append(np.nan)
            continue
        A = np.c_[np.ones(m.sum()), xx[m].values]
        c, *_ = np.linalg.lstsq(A, yy[m].values, rcond=None)
        out.append(c[0] + c[1] * 1500.0)
    return pd.Series(out, index=grid)

def nbm_index(tf, ok, grid, kind, params):
    X, y = (tf.loc[ok, FEATS], tf.loc[ok, TC])
    tr = X.index <= PR.DEV_END
    if tr.sum() < 2000:
        return pd.Series(np.nan, index=grid)
    if kind == 'gbm':
        m = HistGradientBoostingRegressor(random_state=0, **params).fit(X[tr], y[tr])
        pred = pd.Series(m.predict(X), index=X.index)
    else:
        sc = StandardScaler().fit(X[tr])
        m = MLPRegressor(max_iter=200, early_stopping=True, random_state=0, **params).fit(sc.transform(X[tr]), y[tr])
        pred = pd.Series(m.predict(sc.transform(X)), index=X.index)
    return window_mean(y - pred, grid)

def pca_index(tf, ok, grid, n_comp):
    cols = FEATS + [TC]
    X = tf.loc[ok, cols].dropna()
    tr = X.index <= PR.DEV_END
    if tr.sum() < 2000:
        return pd.Series(np.nan, index=grid)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    p = PCA(n_components=n_comp).fit(sc.transform(X[tr]))
    err = np.sqrt(((Xs - p.inverse_transform(p.transform(Xs))) ** 2).mean(axis=1))
    return window_mean(pd.Series(err, index=X.index), grid)
if __name__ == '__main__':
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    g, mp, npc = tune(frames)
    import json
    json.dump({'gbm': g, 'mlp': {**mp, 'hidden_layer_sizes': list(mp['hidden_layer_sizes'])}, 'mlp_max_iter': 200, 'pca_components': npc}, open(P.RESULTS / 'selected_hyperparameters.json', 'w'), indent=2)
    idx = {n: {} for n in ['static', 'nbm-gbm', 'nbm-mlp', 'pca']}
    for t in ti.TURBINES:
        tf = frames[t]
        ok = C.causal_clean_v2(tf, TC)
        grid = C.trailing_grid(tf.index)
        idx['static'][t] = static_index(tf, ok, grid)
        idx['nbm-gbm'][t] = nbm_index(tf, ok, grid, 'gbm', g)
        idx['nbm-mlp'][t] = nbm_index(tf, ok, grid, 'mlp', mp)
        idx['pca'][t] = pca_index(tf, ok, grid, npc)
        print('done', t, flush=True)
    P.save(idx, 'causal_baselines')
