import numpy as np
import pandas as pd
import pickle
from scipy.stats import chi2
TRAIN_END = '2019-12-31'
MON_START = '2020-01-01'
HEALTHY = ['MF02', 'MF03', 'MF04', 'MF05', 'MF06', 'MF09', 'MF13']
FAIL_MF12 = pd.Timestamp('2022-02-23')

def to_z(M, variant):
    M = M.replace([np.inf, -np.inf], np.nan)
    if variant == 'FLEET':
        loo = pd.DataFrame({t: M.drop(columns=t).median(axis=1) for t in M.columns}, index=M.index)
        M = M - loo
    mu, sd = (M.loc[:TRAIN_END].mean(), M.loc[:TRAIN_END].std())
    return (M - mu) / sd

def cusum(z, k):
    s, out = (0.0, [])
    for v in np.nan_to_num(z.values):
        s = max(0.0, s + v - k)
        out.append(s)
    return pd.Series(out, index=z.index)

def episodes(S, h):
    above = (S > h).astype(int).values
    return int(above[0] + np.sum(np.diff(above) == 1))

def run(Z, h, k):
    n_fa, tyears, first = (0, 0.0, {})
    for t in Z.columns:
        S = cusum(Z[t].loc[MON_START:], k)
        hit = S[S > h]
        first[t] = hit.index[0] if len(hit) else None
        if t in HEALTHY:
            n_fa += episodes(S, h)
            tyears += (S.index[-1] - S.index[0]).days / 365.25
    return (n_fa, tyears, first)

def poisson_ci(n, T):
    lo = chi2.ppf(0.025, 2 * n) / 2 / T if n > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * n + 2) / 2 / T
    return (lo, hi)

def calibrate(Z, k, max_fa=0):
    for h in np.arange(0.5, 200.0, 0.5):
        n_fa, T, _ = run(Z, h, k)
        if n_fa <= max_fa:
            return h
    return None
if __name__ == '__main__':
    store = pickle.load(open('model_traj.pkl', 'rb'))
    M = store['T_gbb', 'VA']
    for variant in ['ABS', 'FLEET']:
        Z = to_z(M, variant)
        print(f'\n===== {variant} (gearbox bearing, model VA) =====')
        print(f"{'k':>4s} {'h':>6s} {'FA':>3s} {'FAR/turb-yr':>12s} {'95% CI':>16s} {'MF12 lead':>10s}")
        for k in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = calibrate(Z, k, max_fa=0)
            if h is None:
                print(f'{k:4.1f}  no threshold with zero false alarms')
                continue
            n_fa, T, first = run(Z, h, k)
            lo, hi = poisson_ci(n_fa, T)
            a = first.get('MF12')
            lead = f'{(FAIL_MF12 - a).days} d' if a is not None else 'no alarm'
            print(f'{k:4.1f} {h:6.1f} {n_fa:3d} {n_fa / T:12.3f} [{lo:.3f}, {hi:.3f}]'.ljust(45) + f'{lead:>10s}')
        print(f'  healthy exposure: {T:.1f} turbine-years')
    Mm = store['T_mbre', 'VAS']
    s = Mm['MF07'].dropna()
    base = s.loc['2019-06':'2021-12']
    pre = s.loc['2017-11':'2018-05']
    print(f'\nMF07 retrospective (main bearing, model VAS): healthy-period {base.mean():.2f}+/-{base.std():.2f} K, pre-outage {pre.mean():.2f} K, z={(pre.mean() - base.mean()) / base.std():+.1f}')
    print('\nper-turbine index residual sd [K], model VA / T_gbb:')
    for t in M.columns:
        s = M[t].replace([np.inf, -np.inf], np.nan).dropna()
        print(f'  {t}: sd {s.std():.2f}  n={len(s)}')
