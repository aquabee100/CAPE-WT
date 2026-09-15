import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import paths as P
ST = {'phys_frozen': ('physics', 'frozen'), 'phys_adaptive': ('physics', 'adaptive'), 'gbm_frozen': ('GBM', 'frozen'), 'gbm_adaptive': ('GBM', 'adaptive')}
KEY = ['turbine', 'amplitude', 'ramp_days']

def load(prefix, cols):
    out = []
    for c, (s, a) in ST.items():
        f = P.RESULTS / f'{prefix}{c}.csv'
        if not f.exists():
            print('MISSING', f.name)
            continue
        x = pd.read_csv(f)[KEY + cols]
        x['structure'], x['adaptation'] = (s, a)
        out.append(x)
    return pd.concat(out, ignore_index=True)
RAWC = ['raw_auc_0_45', 'raw_auc_45_180', 'raw_auc_0_180']
ZC = ['auc_0_45', 'auc_45_180', 'auc_0_180']
raw = load('sigpres_raw__', RAWC)
z = load('sigpres__', ZC)

def paired(d, cols, tag):
    fr = d[d.adaptation == 'frozen'].set_index(KEY + ['structure'])[cols]
    ad = d[d.adaptation == 'adaptive'].set_index(KEY + ['structure'])[cols]
    idx = fr.index.intersection(ad.index)
    rows = []
    for c in cols:
        f, a = (fr.loc[idx, c], ad.loc[idx, c])
        r = pd.DataFrame({'f': f, 'a': a}).reset_index()
        r['interval'] = c.replace('raw_auc_', '').replace('auc_', '')
        r['R'] = np.where(r['f'].abs() > 1e-12, r['a'] / r['f'], np.nan)
        rows.append(r)
    r = pd.concat(rows, ignore_index=True)
    r['metric'] = tag
    return r
pr = pd.concat([paired(raw, RAWC, 'raw'), paired(z, ZC, 'z')], ignore_index=True)
pr.to_csv(P.RESULTS / 'sigpres_4b_paired_rows.csv', index=False)
g = pr.groupby(['metric', 'structure', 'amplitude', 'ramp_days', 'interval']).agg(n_turbines=('R', 'size'), n_finite=('R', lambda s: int(s.notna().sum())), n_zero_denom=('f', lambda s: int((s.abs() <= 1e-12).sum())), median_R=('R', 'median'), q25=('R', lambda s: s.quantile(0.25)), q75=('R', lambda s: s.quantile(0.75)), min_R=('R', 'min'), max_R=('R', 'max'), n_below_1=('R', lambda s: int((s < 1).sum()))).round(3).reset_index()
g.to_csv(P.RESULTS / 'sigpres_4b_paired.csv', index=False)
for m in ['raw', 'z']:
    for iv in ['45_180', '0_45']:
        sub = g[(g.metric == m) & (g.interval == iv)]
        print(f'\n=== {m.upper()} paired retention, interval {iv} d ===')
        print(sub.drop(columns=['metric', 'interval']).to_string(index=False))
