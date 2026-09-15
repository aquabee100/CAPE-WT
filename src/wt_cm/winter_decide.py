import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import paths as P
K = ['turbine', 'amplitude', 'ramp_days']
PAIRS = {'GBM': ('gbm_frozen', 'gbm_adaptive'), 'physics': ('phys_frozen', 'phys_adaptive')}
COLS = {'raw_45_180': 'raw_auc_45_180', 'raw_0_45': 'raw_auc_0_45', 'z_45_180': 'auc_45_180', 'z_0_45': 'auc_0_45'}
rows = []
for season, pre in [('summer 2020-06-01', None), ('winter 2020-12-01', 'winter__')]:
    for st, (fc, ac) in PAIRS.items():
        if pre:
            f = pd.read_csv(P.RESULTS / f'winter__{fc}.csv')
            a = pd.read_csv(P.RESULTS / f'winter__{ac}.csv')
        else:
            f = pd.read_csv(P.RESULTS / f'sigpres_raw__{fc}.csv').merge(pd.read_csv(P.RESULTS / f'sigpres__{fc}.csv')[K + ['auc_45_180', 'auc_0_45']], on=K)
            a = pd.read_csv(P.RESULTS / f'sigpres_raw__{ac}.csv').merge(pd.read_csv(P.RESULTS / f'sigpres__{ac}.csv')[K + ['auc_45_180', 'auc_0_45']], on=K)
        f, a = (f.set_index(K), a.set_index(K))
        for lab, col in COLS.items():
            R = np.where(f[col].abs() > 1e-12, a[col] / f[col], np.nan)
            d = pd.DataFrame({'R': R}, index=f.index).reset_index()
            for (A, Dd), g in d.groupby(['amplitude', 'ramp_days']):
                rows.append(dict(season=season, structure=st, metric=lab, amplitude=A, ramp_days=Dd, n_finite=int(g.R.notna().sum()), median_R=round(float(g.R.median()), 3) if g.R.notna().any() else np.nan, below1=int((g.R < 1).sum())))
t = pd.DataFrame(rows)
t.to_csv(P.RESULTS / 'season_comparison.csv', index=False)
print('=== GBM, raw 45-180 d: absorption criterion ===')
g = t[(t.structure == 'GBM') & (t.metric == 'raw_45_180')]
print(g[['season', 'amplitude', 'ramp_days', 'median_R', 'below1', 'n_finite']].to_string(index=False))
print('\n=== GBM, z 45-180 d: standardised criterion ===')
z = t[(t.structure == 'GBM') & (t.metric == 'z_45_180')]
print(z[['season', 'amplitude', 'ramp_days', 'median_R', 'below1']].to_string(index=False))
print('\n=== physics, raw 45-180 d: heterogeneity criterion ===')
p = t[(t.structure == 'physics') & (t.metric == 'raw_45_180')]
print(p[['season', 'amplitude', 'ramp_days', 'median_R', 'below1']].to_string(index=False))
w = g[g.season.str.startswith('winter')]
crit1 = bool((w.median_R < 1).all() and (w.below1 > w.n_finite / 2).all())
wz = z[z.season.str.startswith('winter')]
crit2 = bool((wz.median_R > 1).all())
pw = p[p.season.str.startswith('winter')]
crit3 = bool(pw.below1.between(1, 6).any() or pw.below1.nunique() > 1)
print(f'\nGBM  R_raw<1 dominant in all 6 winter cells : {crit1}')
print(f'GBM  R_z>1 aggregate in all 6 winter cells  : {crit2}')
print(f'physics heterogeneous (not uniform)        : {crit3}')
print('\nDECISION: ' + ('CLOSE synthetic perturbation work' if crit1 and crit2 and crit3 else 'EXPAND to remaining seasonal onsets'))
