import itertools, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
import paths as P, causal as C, causal_phys as CP, detect_final as D
import protocol as PR, thermal_id as ti, online_2x2 as O
import signal_preservation as S
_PHYS_COEF, _GBM_MODEL = ({}, {})

def phys_frozen_cached(tf, turb):
    ok = O.online_clean(tf)
    X, y = CP.design(tf, O.TC, 'VA')
    m = ok & X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    if turb not in _PHYS_COEF:
        tr = X.index <= pd.Timestamp(PR.DEV_END)
        _PHYS_COEF[turb] = O._ls(X[tr], y[tr])
    c = _PHYS_COEF[turb]
    rows = []
    for end in C.trailing_grid(tf.index):
        Xs, ys = (X.loc[end - C.WINDOW:end], y.loc[end - C.WINDOW:end])
        if len(Xs) >= 400:
            rows.append({'t': end, 'idx': float((ys.values - Xs.values @ c).mean())})
    return pd.DataFrame(rows).set_index('t')['idx']

def gbm_frozen_cached(tf, turb):
    ok = O.online_clean(tf)
    X, y = (tf.loc[ok, O.FEATS], tf.loc[ok, O.TC])
    m = X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    if turb not in _GBM_MODEL:
        tr = X.index <= pd.Timestamp(PR.DEV_END)
        _GBM_MODEL[turb] = HistGradientBoostingRegressor(random_state=0, **O.GBM_PARAMS).fit(X[tr].iloc[::O.STRIDE], y[tr].iloc[::O.STRIDE])
    mdl = _GBM_MODEL[turb]
    rows = []
    for end in C.trailing_grid(tf.index):
        Xs, ys = (X.loc[end - C.WINDOW:end], y.loc[end - C.WINDOW:end])
        if len(Xs) >= 400:
            rows.append({'t': end, 'idx': float((ys.values - mdl.predict(Xs)).mean())})
    return pd.DataFrame(rows).set_index('t')['idx']
CELLS = {'phys_frozen': lambda tf, t: phys_frozen_cached(tf, t), 'phys_adaptive': lambda tf, t: O.phys_residual(tf, True), 'gbm_frozen': lambda tf, t: gbm_frozen_cached(tf, t), 'gbm_adaptive': lambda tf, t: O.gbm_residual(tf, True)}

def aucp(dr, t0, a, b):
    seg = dr.loc[t0 + pd.Timedelta(days=a):t0 + pd.Timedelta(days=b)].dropna()
    if len(seg) < 2:
        return 0.0
    dtd = np.diff(seg.index.values).astype('timedelta64[D]').astype(float)
    return float((np.clip(seg.values[:-1], 0, None) * dtd).sum())
if __name__ == '__main__':
    import sys
    cells = sys.argv[1:] or list(CELLS)
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    for cell in cells:
        base = pd.read_parquet(P.RESULTS / f'resid_{cell}.parquet')
        ck = P.RESULTS / f'sigpres_raw__{cell}.partial.csv'
        done = set()
        rows = []
        if ck.exists():
            prev = pd.read_csv(ck)
            rows = prev.to_dict('records')
            done = {(r['turbine'], r['amplitude'], r['ramp_days']) for r in rows}
            print(f'resuming {cell}: {len(done)} scenarios already done', flush=True)
        for turb in D.HEALTHY:
            org = base[turb]
            for t0, A, Dd in itertools.product(S.ONSETS, S.AMPLITUDES, S.RAMPS):
                if (turb, A, Dd) in done:
                    continue
                inj = CELLS[cell](S.perturb(frames[turb], t0, A, Dd), turb)
                dr = (inj - org).dropna()
                t0s = pd.Timestamp(t0)
                rows.append(dict(cell=cell, turbine=turb, onset=t0, amplitude=A, ramp_days=Dd, raw_auc_0_45=round(aucp(dr, t0s, 0, 45), 4), raw_auc_45_180=round(aucp(dr, t0s, 45, 180), 4), raw_auc_0_180=round(aucp(dr, t0s, 0, 180), 4), raw_peak=round(float(dr.loc[t0s:t0s + pd.Timedelta(days=180)].max()), 5)))
                pd.DataFrame(rows).to_csv(ck, index=False)
                print(f'  {cell} {turb} A={A} D={Dd}  [{len(rows)}/42]', flush=True)
        pd.DataFrame(rows).to_csv(P.RESULTS / f'sigpres_raw__{cell}.csv', index=False)
        ck.unlink(missing_ok=True)
        print(f'written: results/sigpres_raw__{cell}.csv', flush=True)
