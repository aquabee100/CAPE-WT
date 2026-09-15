import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import paths as P
CELLS = ['phys_frozen', 'phys_adaptive', 'gbm_frozen', 'gbm_adaptive']
STRUCT = {'phys_frozen': ('physics', 'frozen'), 'phys_adaptive': ('physics', 'adaptive'), 'gbm_frozen': ('GBM', 'frozen'), 'gbm_adaptive': ('GBM', 'adaptive')}
d = []
for c in CELLS:
    f = P.RESULTS / f'sigpres__{c}.csv'
    if not f.exists():
        print(f'MISSING {c}')
        continue
    x = pd.read_csv(f)
    x['structure'], x['adaptation'] = STRUCT[c]
    d.append(x)
d = pd.concat(d, ignore_index=True)
d.to_csv(P.RESULTS / 'sigpres_all.csv', index=False)
g = d.groupby(['structure', 'adaptation', 'amplitude', 'ramp_days']).agg(n=('turbine', 'count'), detection_fraction=('warned', 'mean'), median_delay=('delay_days', 'median'), median_peak_dz=('peak_dz', 'median'), median_auc_0_45=('auc_0_45', 'median'), median_auc_45_180=('auc_45_180', 'median'), median_auc_0_180=('auc_0_180', 'median')).round(3).reset_index()
print('=== Milestone 4: signal preservation, 7 reference turbines ===')
print(g.to_string(index=False))
g.to_csv(P.RESULTS / 'sigpres_summary.csv', index=False)
piv = g.pivot_table(index=['structure', 'amplitude', 'ramp_days'], columns='adaptation', values=['median_auc_0_45', 'median_auc_45_180', 'median_auc_0_180', 'detection_fraction'])
print('\n=== adaptation retention R = adaptive / frozen ===')
rows = []
for (st, A, Dd), r in piv.iterrows():

    def rat(k):
        fr, ad = (r[k, 'frozen'], r[k, 'adaptive'])
        return round(ad / fr, 3) if fr and fr > 1e-09 else np.nan
    rows.append(dict(structure=st, amplitude=A, ramp_days=Dd, R_auc_0_45=rat('median_auc_0_45'), R_auc_45_180=rat('median_auc_45_180'), R_auc_0_180=rat('median_auc_0_180'), frozen_auc_0_180=r['median_auc_0_180', 'frozen'], adaptive_auc_0_180=r['median_auc_0_180', 'adaptive'], frozen_det=r['detection_fraction', 'frozen'], adaptive_det=r['detection_fraction', 'adaptive']))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
R.to_csv(P.RESULTS / 'sigpres_retention.csv', index=False)
