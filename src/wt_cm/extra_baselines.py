import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import paths as P, causal as C, protocol as PR, thermal_id as ti, online_2x2 as O
import competitive_baselines as CB
COLS = O.FEATS + [O.TC]

def mahalanobis_fit(Xt, yt):
    Z = np.c_[Xt.values, yt.values]
    sc = StandardScaler().fit(Z)
    cov = EmpiricalCovariance().fit(sc.transform(Z))
    return lambda Xs, ys: float(np.sqrt(np.maximum(cov.mahalanobis(sc.transform(np.c_[Xs.values, ys.values])), 0)).mean())

def iforest_fit(Xt, yt):
    Z = np.c_[Xt.values, yt.values]
    sc = StandardScaler().fit(Z)
    m = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(Z)[::CB.STRIDE])
    return lambda Xs, ys: float(-m.score_samples(sc.transform(np.c_[Xs.values, ys.values])).mean())

def autoencoder_fit(Xt, yt):
    Z = np.c_[Xt.values, yt.values]
    sc = StandardScaler().fit(Z)
    Zs = sc.transform(Z)[::CB.STRIDE]
    m = MLPRegressor(hidden_layer_sizes=(16, 3, 16), alpha=0.0001, max_iter=200, early_stopping=True, random_state=0).fit(Zs, Zs)

    def score(Xs, ys):
        Q = sc.transform(np.c_[Xs.values, ys.values])
        return float(np.sqrt(((Q - m.predict(Q)) ** 2).mean(axis=1)).mean())
    return score
ARMS = {'resid_maha_frozen': lambda tf: CB._score(tf, mahalanobis_fit, False), 'resid_maha_adaptive': lambda tf: CB._score(tf, mahalanobis_fit, True), 'resid_ifor_frozen': lambda tf: CB._score(tf, iforest_fit, False), 'resid_ifor_adaptive': lambda tf: CB._score(tf, iforest_fit, True), 'resid_ae_frozen': lambda tf: CB._score(tf, autoencoder_fit, False), 'resid_ae_adaptive': lambda tf: CB._score(tf, autoencoder_fit, True)}
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
            print(f'resuming {name}: {len(out)}/9', flush=True)
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
