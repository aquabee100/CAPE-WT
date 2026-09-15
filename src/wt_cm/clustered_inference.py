import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D
RNG = np.random.default_rng(0)
B = 10000

def cluster_ci(units, stat, n=B, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(units)
    out = []
    for _ in range(n):
        pick = rng.choice(keys, size=len(keys), replace=True)
        vals = np.concatenate([units[k] for k in pick])
        if len(vals):
            out.append(stat(vals))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.median(out)))
rows = []
K = P.RESULTS / 'kelmarsh_k1_coefficients.csv'
if K.exists():
    k = pd.read_csv(K)
    units = {t: (g.a_nac.values > 0).astype(float) for t, g in k.groupby('turbine')}
    lo, hi, med = cluster_ci(units, np.mean)
    rows.append(dict(quantity='external: fraction of windows with a_nac>0', naive=f'{int((k.a_nac > 0).sum())}/{len(k)} = 1.000', n_clusters=len(units), cluster_median=round(med, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
    K15 = P.RESULTS / 'kelmarsh_k15_identifiability.csv'
    if K15.exists():
        k15 = pd.read_csv(K15)
        u = {t: g.steady_state_rise_K.dropna().values for t, g in k15.groupby('turbine')}
        lo, hi, med = cluster_ci(u, np.median)
        rows.append(dict(quantity='external: steady-state rise (K), median', naive=f'{k15.steady_state_rise_K.median():.2f} K over 192 windows', n_clusters=len(u), cluster_median=round(med, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2)))

def sig_units(kind, cell, col):
    f = P.RESULTS / f"sigpres{('_raw' if kind == 'raw' else '')}__{cell}.csv"
    w = P.RESULTS / f'winter__{cell}.csv'
    ds = [pd.read_csv(f)] + ([pd.read_csv(w)] if w.exists() else [])
    return pd.concat(ds, ignore_index=True)
try:
    rf = sig_units('raw', 'gbm_frozen', None).set_index(['turbine', 'amplitude', 'ramp_days', 'onset'])
    ra = sig_units('raw', 'gbm_adaptive', None).set_index(['turbine', 'amplitude', 'ramp_days', 'onset'])
    zf = sig_units('z', 'gbm_frozen', None).set_index(['turbine', 'amplitude', 'ramp_days', 'onset'])
    za = sig_units('z', 'gbm_adaptive', None).set_index(['turbine', 'amplitude', 'ramp_days', 'onset'])
    idx = rf.index.intersection(ra.index).intersection(zf.index).intersection(za.index)
    Rraw = ra.loc[idx, 'raw_auc_45_180'] / rf.loc[idx, 'raw_auc_45_180']
    Rz = za.loc[idx, 'auc_45_180'] / zf.loc[idx, 'auc_45_180']
    for lab, S, test in [('injection: raw absorption (R_raw<1)', Rraw, lambda v: np.mean(v < 1)), ('injection: standardised gain (R_z>1)', Rz, lambda v: np.mean(v > 1))]:
        u = {}
        for (t, *_), v in S.items():
            u.setdefault(t, []).append(v)
        u = {k2: np.array(v) for k2, v in u.items()}
        lo, hi, med = cluster_ci(u, test)
        naive = f'{int(test(S.values) * len(S))}/{len(S)}'
        rows.append(dict(quantity=lab, naive=naive, n_clusters=len(u), cluster_median=round(med, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
except Exception as e:
    print('injection block skipped:', type(e).__name__, e)
pt = P.RESULTS / 'availability_per_turbine.csv'
if pt.exists():
    a = pd.read_csv(pt)
    if {'warn_B', 'expo_B'} <= set(a.columns):
        u = {r.turbine: np.array([[r.warn_B, r.expo_B]]) for r in a.itertuples()}
        lo, hi, med = cluster_ci(u, lambda v: v[:, 0].sum() / v[:, 1].sum())
        rows.append(dict(quantity='parallel warning burden (episodes/turbine-yr)', naive=f'{a.warn_B.sum():.0f}/{a.expo_B.sum():.2f} = {a.warn_B.sum() / a.expo_B.sum():.3f}', n_clusters=len(u), cluster_median=round(med, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3)))
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'clustered_inference.csv', index=False)
print(d.to_string(index=False))
print(f'\nturbine-cluster bootstrap, {B} resamples; the turbine is the unit.')
