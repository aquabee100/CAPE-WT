import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import paths as P
import thermal_id as ti
import causal as C
import causal_baselines as CB
warnings.filterwarnings('ignore')
LOOKBACK = pd.Timedelta('365D')
MAXROWS = 10000
import json
GBM = dict(max_iter=300, max_depth=6, learning_rate=0.05)
MLP = dict(hidden_layer_sizes=(64, 32), alpha=0.0001)
MLP_MAX_ITER = 200

def load_hyperparameters(path='selected_hyperparameters.json'):
    global GBM, MLP, MLP_MAX_ITER
    try:
        hp = json.load(open(P.RESULTS / path))
    except FileNotFoundError:
        print(f'[gate_b2] {path} not found; using DEV defaults.')
        return
    GBM = hp['gbm']
    MLP = {'hidden_layer_sizes': tuple(hp['mlp']['hidden_layer_sizes']), 'alpha': hp['mlp']['alpha']}
    MLP_MAX_ITER = hp['mlp_max_iter']

def _sub(X, y):
    if len(X) <= MAXROWS:
        return (X, y)
    step = int(np.ceil(len(X) / MAXROWS))
    return (X[::step], y[::step])

def adaptive_index(tf, ok, grid, kind, seed=0):
    X, y = (tf.loc[ok, CB.FEATS], tf.loc[ok, CB.TC])
    t0 = X.index[0]
    out = []
    for e in grid:
        score_lo = e - C.WINDOW
        if score_lo - LOOKBACK < t0:
            out.append(np.nan)
            continue
        tr = X.loc[score_lo - LOOKBACK:score_lo]
        ytr = y.loc[score_lo - LOOKBACK:score_lo]
        if len(tr) and tr.index[-1] == score_lo:
            tr, ytr = (tr.iloc[:-1], ytr.iloc[:-1])
        sc, ysc = (X.loc[score_lo:e], y.loc[score_lo:e])
        if len(tr) < 3000 or len(sc) < 200:
            out.append(np.nan)
            continue
        Xtr, Ytr = _sub(tr.values, ytr.values)
        if kind == 'gbm':
            m = HistGradientBoostingRegressor(random_state=seed, **GBM).fit(Xtr, Ytr)
            pred = m.predict(sc.values)
        else:
            s = StandardScaler().fit(Xtr)
            m = MLPRegressor(max_iter=MLP_MAX_ITER, early_stopping=True, random_state=seed, **MLP).fit(s.transform(Xtr), Ytr)
            pred = m.predict(s.transform(sc.values))
        out.append(float((ysc.values - pred).mean()))
    return pd.Series(out, index=grid)
if __name__ == '__main__':
    import sys, os
    todo = sys.argv[1:] if len(sys.argv) > 1 else list(ti.TURBINES)
    kinds = os.environ.get('KINDS', 'gbm').split(',')
    seeds = [int(x) for x in os.environ.get('SEEDS', '0').split(',')]
    load_hyperparameters()
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in todo}
    del df
    res = P.load('adaptive_nbm2') if P.exists('adaptive_nbm2') else {}
    for kind in kinds:
        res.setdefault(f'nbm-{kind}-adaptive', {})
    for t in todo:
        tf = frames[t]
        ok = C.causal_clean_v2(tf, CB.TC)
        grid = C.trailing_grid(tf.index)
        for kind in kinds:
            for sd in seeds:
                key = f'nbm-{kind}-adaptive' + ('' if sd == 0 else f'-s{sd}')
                res.setdefault(key, {})
                res[key][t] = adaptive_index(tf, ok, grid, kind, seed=sd)
        P.save(res, 'adaptive_nbm2')
        print('done', t, flush=True)
