import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import paths as P
import causal as C
import protocol as PR
import detect_final as D
import thermal_id as ti
import online_2x2 as O
TC = O.TC
DEV_END = pd.Timestamp(PR.DEV_END)

def frames(tf):
    ok = O.online_clean(tf)
    X, y = (tf.loc[ok, O.FEATS], tf.loc[ok, TC])
    m = X.notna().all(axis=1) & y.notna()
    return (X[m], y[m])

def p1_zaher(tf, train_end=DEV_END):
    X, y = frames(tf)
    Xl = pd.concat([X, X.shift(1).add_suffix('_l1'), X.shift(6).add_suffix('_l6')], axis=1).dropna()
    yl = y.reindex(Xl.index).dropna()
    Xl = Xl.reindex(yl.index)
    tr = Xl.index <= pd.Timestamp(train_end)
    if tr.sum() < 2000:
        return pd.Series(dtype=float)
    sc = StandardScaler().fit(Xl[tr].values)
    m = MLPRegressor(hidden_layer_sizes=(3,), max_iter=300, early_stopping=True, random_state=0).fit(sc.transform(Xl[tr].values[::O.STRIDE]), yl[tr].values[::O.STRIDE])
    r = pd.Series(yl.values - m.predict(sc.transform(Xl.values)), index=Xl.index)
    return r.resample('D').mean().dropna()

def p2_schlechtingen(tf, train_end=DEV_END):
    X, y = frames(tf)
    X = X[['P', 'T_nac', 'rpm']]
    tr = X.index <= pd.Timestamp(train_end)
    if tr.sum() < 2000:
        return pd.Series(dtype=float)
    A = np.c_[np.ones(int(tr.sum())), X[tr].values]
    c = np.linalg.lstsq(A.T @ A + 1e-08 * np.eye(A.shape[1]), A.T @ y[tr].values, rcond=None)[0]
    r = pd.Series(y.values - np.c_[np.ones(len(X)), X.values] @ c, index=X.index)
    return r.resample('D').mean().dropna()

def p3_gbm_ewma(tf, lam=0.2, train_end=DEV_END):
    X, y = frames(tf)
    tr = X.index <= pd.Timestamp(train_end)
    if tr.sum() < 2000:
        return pd.Series(dtype=float)
    m = HistGradientBoostingRegressor(random_state=0, **O.GBM_PARAMS).fit(X[tr].iloc[::O.STRIDE], y[tr].iloc[::O.STRIDE])
    r = pd.Series(y.values - m.predict(X), index=X.index).resample('D').mean().dropna()
    mu, sd = (r[r.index <= pd.Timestamp(train_end)].mean(), r[r.index <= pd.Timestamp(train_end)].std())
    z = (r - mu) / (sd if sd else 1.0)
    return z.ewm(alpha=lam, adjust=False).mean()

def p4_milani(tf, fault_turbine_frames, event, val_frames):

    def prep(f):
        X, y = frames(f)
        Z = pd.concat([X, y.rename(TC)], axis=1).resample('6h').mean().dropna()
        return Z
    Ztr = prep(fault_turbine_frames)
    ev = pd.Timestamp(event)
    Ztr = Ztr.loc[ev - pd.Timedelta('365D'):ev]
    if len(Ztr) < 200:
        return pd.Series(dtype=float)
    lab = (Ztr.index >= ev - pd.Timedelta('30D')).astype(int)
    idx = np.r_[np.where(lab == 0)[0], np.random.default_rng(0).choice(np.where(lab == 1)[0], size=(lab == 0).sum(), replace=True)]
    sc = StandardScaler().fit(Ztr.values)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8, 4), max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=0).fit(sc.transform(Ztr.values[idx]), lab[idx])
    Zs = prep(tf)
    pred = pd.Series(clf.predict(sc.transform(Zs.values)), index=Zs.index)
    return pred.rolling('7D').mean()

def p5_mcusum(tf, k=0.5, train_end=DEV_END):
    X, y = frames(tf)
    Z = pd.concat([X, y.rename(TC)], axis=1).resample('D').mean().dropna()
    tr = Z.index <= pd.Timestamp(train_end)
    if tr.sum() < 60:
        return pd.Series(dtype=float)
    sc = StandardScaler().fit(Z[tr].values)
    S = sc.transform(Z.values)
    cov = np.cov(S[np.asarray(tr)].T) + 1e-06 * np.eye(S.shape[1])
    inv = np.linalg.inv(cov)
    d = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', S, inv, S), 0))
    mu, sd = (d[np.asarray(tr)].mean(), d[np.asarray(tr)].std())
    dz = (d - mu) / (sd if sd else 1.0)
    out, s = ([], 0.0)
    for v in dz:
        s = max(0.0, s + v - k)
        out.append(s)
    return pd.Series(out, index=Z.index)
METHODS = {'P1 Zaher ANN-NBM': dict(fn=p1_zaher, needs_fault=False), 'P2 Schlechtingen linear FSRC': dict(fn=p2_schlechtingen, needs_fault=False), 'P3 GBM-NBM + EWMA': dict(fn=p3_gbm_ewma, needs_fault=False), 'P4 Milani ANN fault index': dict(fn=p4_milani, needs_fault=True), 'P5 multivariate CUSUM': dict(fn=p5_mcusum, needs_fault=False)}
FIDELITY = {'P1 Zaher ANN-NBM': '3 hidden neurons and lagged inputs per paper; lag set (10 min, 1 h) and threshold rule are OUR choices', 'P2 Schlechtingen linear FSRC': 'inputs and daily-average residual per paper; alarm threshold is OUR choice', 'P3 GBM-NBM + EWMA': 'EWMA chart per the modern NBM literature; lambda=0.2 and L=3 are standard SPC defaults, not from one paper', 'P4 Milani ANN fault index': 'architecture, labelling, weekly MA and 6 h resampling per paper; SMOTE replaced by random oversampling; REQUIRES the fault event to train', 'P5 multivariate CUSUM': 'standard MCUSUM; k=0.5 is conventional, not from a specific paper'}

def _method_key(name):
    return name.split()[0]

def _parse_args():
    ap = argparse.ArgumentParser(description='Regenerate case-specific literature-informed score trajectories.')
    ap.add_argument('--case', choices=['MF07', 'MF12', 'all'], default='all', help='case-specific historical cutoff to run (default: all)')
    ap.add_argument('--method', choices=['P1', 'P2', 'P3', 'P4', 'P5', 'all'], default='all', help='method to run (default: all)')
    ap.add_argument('--resume', action='store_true', help='skip an output trajectory that already exists')
    return ap.parse_args()

def _out_exists(stem):
    return (P.RESULTS / f'{stem}.csv').exists()
if __name__ == '__main__':
    args = _parse_args()
    df = ti.load_raw()
    fr = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    cases = ['MF07', 'MF12'] if args.case == 'all' else [args.case]
    for case in cases:
        train_end = pd.Timestamp(PR.SCALE_END[case])
        print(f'\n=== {case}: literature baselines trained through {train_end.date()} ===', flush=True)
        for name, cfg in METHODS.items():
            key = _method_key(name)
            if cfg['needs_fault'] or (args.method != 'all' and key != args.method):
                continue
            stem = f'published_{case}__{key}'
            if args.resume and _out_exists(stem):
                print(f'{name}: skip existing {stem}.csv', flush=True)
                continue
            out = {}
            for t, tf in fr.items():
                try:
                    ss = cfg['fn'](tf, train_end=train_end)
                    if len(ss):
                        out[t] = ss
                except Exception as e:
                    print(f'  {name} {t}: {type(e).__name__} {e}', flush=True)
            if out:
                M = pd.DataFrame(out)
                P.save_trajectory(M, stem)
                print(f'{name}: {M.shape}', flush=True)
    if args.method in {'all', 'P4'}:
        stem = 'published_context__P4'
        if args.resume and _out_exists(stem):
            print(f'P4 Milani ANN fault index: skip existing {stem}.csv', flush=True)
        else:
            out = {}
            cfg = METHODS['P4 Milani ANN fault index']
            for t, tf in fr.items():
                try:
                    ss = cfg['fn'](tf, fr['MF12'], PR.CASES['MF12']['event'], None)
                    if len(ss):
                        out[t] = ss
                except Exception as e:
                    print(f'  P4 Milani ANN fault index {t}: {type(e).__name__} {e}', flush=True)
            if out:
                M = pd.DataFrame(out)
                P.save_trajectory(M, stem)
                print(f'P4 Milani ANN fault index (contextual, event-trained): {M.shape}', flush=True)
