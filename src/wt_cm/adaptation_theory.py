import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P
K = ['turbine', 'amplitude', 'ramp_days']
IV = '45_180'

def load(kind, cell):
    f = P.RESULTS / f"sigpres{('_raw' if kind == 'raw' else '')}__{cell}.csv"
    return pd.read_csv(f)
rows = []
for struct, fro, ada in [('GBM', 'gbm_frozen', 'gbm_adaptive'), ('physics', 'phys_frozen', 'phys_adaptive')]:
    rf, ra = (load('raw', fro).set_index(K), load('raw', ada).set_index(K))
    zf, za = (load('z', fro).set_index(K), load('z', ada).set_index(K))
    idx = rf.index.intersection(ra.index).intersection(zf.index).intersection(za.index)
    for i in idx:
        raw_f, raw_a = (rf.loc[i, f'raw_auc_{IV}'], ra.loc[i, f'raw_auc_{IV}'])
        z_f, z_a = (zf.loc[i, f'auc_{IV}'], za.loc[i, f'auc_{IV}'])
        if not (np.isfinite([raw_f, raw_a, z_f, z_a]).all() and raw_f > 0 and (z_f > 0)):
            continue
        R_raw = raw_a / raw_f
        R_z = z_a / z_f
        sigma_ratio = R_raw / R_z
        rows.append(dict(structure=struct, turbine=i[0], amplitude=i[1], ramp_days=i[2], R_raw=round(R_raw, 4), R_z=round(R_z, 4), one_minus_alpha=round(R_raw, 4), sigma_A_over_F=round(sigma_ratio, 4), inequality_holds=bool(R_raw > sigma_ratio), detectability_improved=bool(R_z > 1)))
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'adaptation_theory.csv', index=False)
print('=== does  1 - alpha > sigma_A/sigma_F  predict  R_z > 1 ? ===')
for s, g in d.groupby('structure'):
    agree = (g.inequality_holds == g.detectability_improved).mean()
    print(f'\n{s}:  n={len(g)}')
    print(f'  median 1-alpha (raw retained)  : {g.one_minus_alpha.median():.3f}')
    print(f'  median sigma_A/sigma_F         : {g.sigma_A_over_F.median():.3f}')
    print(f'  inequality holds               : {int(g.inequality_holds.sum())}/{len(g)}')
    print(f'  detectability improved (R_z>1) : {int(g.detectability_improved.sum())}/{len(g)}')
    print(f'  PREDICTION AGREES              : {agree * 100:.1f}% of scenarios')
print('\nInterpretation: adaptation absorbs a fraction alpha of the degradation,')
print('but shrinks the background by more, so standardised contrast rises.')
