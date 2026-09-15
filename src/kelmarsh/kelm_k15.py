import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import kelm_k1 as K
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
BLOCKS = {'2019': '2019-01-01_-_2020-01-01', '2021': '2021-01-01_-_2021-07-01'}
coef = pd.read_csv(OUT / 'kelmarsh_k1_coefficients.csv', parse_dates=['t'])
rows = []
for turb in range(1, 7):
    for blab, bglob in BLOCKS.items():
        f = K.load_block(turb, bglob)
        ok = K.clean(f)
        X, y = K.design(f)
        m = ok & X.notna().all(axis=1) & y.notna()
        X = X[m]
        sub = coef[(coef.turbine == f'KWF{turb}') & (coef.period.astype(str) == blab)]
        for _, r in sub.iterrows():
            Xs = X.loc[r.t - K.WINDOW:r.t]
            if len(Xs) < K.MIN_N:
                continue
            P, V = (Xs['P'].values, Xs['visc'].values)
            Z = Xs[['P', 'visc', 'T_amb', 'sink_nac']].values
            Zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
            sv = np.linalg.svd(Zs, compute_uv=False)
            ys = y[m].loc[r.t - K.WINDOW:r.t]
            A = Xs.values
            cc = np.linalg.lstsq(A.T @ A + 1e-08 * np.eye(A.shape[1]), A.T @ ys.values, rcond=None)[0]
            co = dict(zip(Xs.columns, cc))
            q_sup = r.b_P * P + r.b_V * V
            q_for = q_sup + co['const'] + co['T_amb'] * Xs['T_amb'].values
            rc = np.corrcoef(P, V)[0, 1]
            rows.append(dict(turbine=f'KWF{turb}', period=blab, t=r.t, corr_P_visc=round(float(rc), 3), vif_P_visc=round(float(1 / (1 - rc ** 2)), 2), cond_number=round(float(sv[0] / sv[-1]), 1), b_P=r.b_P, b_V=r.b_V, b_0=co['const'], a_nac=r.a_nac, Q_superseded_mean=float(q_sup.mean()), Q_forcing_mean=float(q_for.mean()), steady_state_rise_K=float(q_for.mean() / co['sink_nac']) if co['sink_nac'] > 0 else float('nan')))
d = pd.DataFrame(rows)
d.to_csv(OUT / 'kelmarsh_k15_identifiability.csv', index=False)
print('=== per-turbine identifiability ===')
g = d.groupby('turbine').agg(n=('t', 'size'), med_corr_P_visc=('corr_P_visc', 'median'), med_vif=('vif_P_visc', 'median'), med_cond=('cond_number', 'median'), Q_forcing_med=('Q_forcing_mean', 'median'), rise_med=('steady_state_rise_K', 'median'), rise_q25=('steady_state_rise_K', lambda s: s.quantile(0.25)), rise_q75=('steady_state_rise_K', lambda s: s.quantile(0.75)), Q_forcing_cv=('Q_forcing_mean', lambda s: round(float(s.std() / abs(s.mean())), 3)), Q_superseded_cv=('Q_superseded_mean', lambda s: round(float(s.std() / abs(s.mean())), 3)), frac_bP_pos=('b_P', lambda s: round(float((s > 0).mean()), 3)))
print(g.round(4).to_string())
print('\nQ_forcing  sign: %d pos / %d neg' % ((d.Q_forcing_mean > 0).sum(), (d.Q_forcing_mean < 0).sum()))
print('Q_superseded sign: %d pos / %d neg  <- why the partial sum was wrong' % ((d.Q_superseded_mean > 0).sum(), (d.Q_superseded_mean < 0).sum()))
print('\ncorr(b_P, b_V) across all windows = %.3f' % d[['b_P', 'b_V']].corr().iloc[0, 1])
print('corr(Q_forcing, a_nac)            = %.3f  <- NOT independent evidence' % np.corrcoef(d.Q_forcing_mean, d.a_nac)[0, 1])
print('steady-state rise (K): median %.2f  IQR %.2f..%.2f  n=%d' % (d.steady_state_rise_K.median(), d.steady_state_rise_K.quantile(0.25), d.steady_state_rise_K.quantile(0.75), d.steady_state_rise_K.notna().sum()))
