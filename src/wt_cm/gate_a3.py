import numpy as np
import pandas as pd
import pickle
import paths as P
import detect_final as D
import protocol as PR
import normalise as N
FAIL = pd.Timestamp(PR.FAIL_MF12)
KS = [0.5, 1.0, 1.5, 2.0, 3.0]
ARL_TARGETS = [5.0, 10.0, 20.0]
BLOCK = 3
WIN_PER_YEAR = 365.25 / 15.0
SIM_LEN = 4000
N_SIM = 400
H_GRID = np.arange(0.5, 400.0, 0.5)
CAL_SPANS = {'cal': (PR.CAL_START, PR.CAL_END), 'devcal': (None, PR.CAL_END)}
CAL_MODE = 'devcal'

def cal_pool(Z, turbines, mode=None):
    lo, hi = CAL_SPANS[mode or CAL_MODE]
    out = []
    for t in turbines:
        v = Z[t].loc[lo:hi].values
        v = v[np.isfinite(v)]
        if len(v) >= BLOCK + 1:
            out.append(v)
    return out

def block_matrix(pool, n_sim, n, rng):
    nb = int(np.ceil(n / BLOCK))
    out = np.empty((n_sim, nb * BLOCK))
    who = rng.integers(0, len(pool), size=n_sim)
    for r in range(n_sim):
        s = pool[who[r]]
        st = rng.integers(0, len(s) - BLOCK + 1, size=nb)
        out[r] = s[st[:, None] + np.arange(BLOCK)].reshape(-1)
    return out[:, :n]

def arl_from_matrix(z, k):
    n_sim = z.shape[0]
    s = np.zeros(n_sim)
    cur = np.zeros(n_sim)
    runmax = np.empty((n_sim, SIM_LEN))
    for t in range(SIM_LEN):
        s = np.maximum(0.0, s + z[:, t] - k)
        cur = np.maximum(cur, s)
        runmax[:, t] = cur
    first = np.empty((n_sim, len(H_GRID)))
    for j in range(n_sim):
        idx = np.searchsorted(runmax[j], H_GRID, side='right')
        first[j] = np.where(idx < SIM_LEN, idx + 1, SIM_LEN)
    return first.mean(axis=0) / WIN_PER_YEAR

def h_from_curve(arl, target):
    ok = np.where(arl >= target)[0]
    return float(H_GRID[ok[0]]) if len(ok) else None

def curves_for(Z, turbines, rng, mode=None):
    z = block_matrix(cal_pool(Z, turbines, mode), N_SIM, SIM_LEN, rng)
    return {k: arl_from_matrix(z, k) for k in KS}
FLEET_RELATIVE = False

def fleet_relative(Z):
    return pd.DataFrame({t: Z[t] - Z[[x for x in D.HEALTHY if x != t]].median(axis=1) for t in Z.columns}, index=Z.index)

def evaluate_method(M, name, rows, seed=0, norm=N.canonical_z, mode=None):
    Z = norm(M)
    if FLEET_RELATIVE:
        Z = fleet_relative(Z)
    rng = np.random.default_rng(seed)
    cur_all = curves_for(Z, D.HEALTHY, rng, mode)
    cur_loo = {held: curves_for(Z, [t for t in D.HEALTHY if t != held], rng, mode) for held in D.HEALTHY}
    for k in KS:
        for target in ARL_TARGETS:
            h = h_from_curve(cur_all[k], target)
            if h is None:
                continue
            S = N.cusum_hold(Z['MF12'].loc[PR.EVAL_START:], k)
            hit = S[S > h]
            lead = (FAIL - hit.index[0]).days if len(hit) else None
            fa, episodes, tyears = (0, 0, 0.0)
            for held in D.HEALTHY:
                hl = h_from_curve(cur_loo[held][k], target)
                if hl is None:
                    continue
                Sh = N.cusum_hold(Z[held].loc[PR.EVAL_START:], k)
                above = (Sh > hl).astype(int).values
                n_ep = int(above[0] + np.sum(np.diff(above) == 1))
                episodes += n_ep
                fa += int(n_ep > 0)
                tyears += (Sh.index[-1] - Sh.index[0]).days / 365.25
            rows.append(dict(method=name, cal=mode or CAL_MODE, k=k, arl=target, h=h, lead=lead, fa_turbines=fa, fa_per_turbine_year=episodes / tyears if tyears else np.nan))

def load_methods():
    traj = P.load('causal_traj')
    M = {'phys-VA': traj['VA']}
    for k, v in P.load('causal_baselines').items():
        key = {'nbm-gbm': 'nbm-gbm-frozen', 'nbm-mlp': 'nbm-mlp-frozen'}.get(k, k)
        M[key] = pd.DataFrame(v)
    try:
        for k, v in P.load('adaptive_nbm2').items():
            M[k] = pd.DataFrame(v)
    except Exception:
        pass
    return M
if __name__ == '__main__':
    import itertools
    M = load_methods()
    rows = []
    NORMS = {'fixed': N.canonical_z, 'rolling': N.rolling_z}
    for nm, mode, fr in itertools.product(NORMS, ['cal', 'devcal'], [False, True]):
        globals()['CAL_MODE'] = mode
        globals()['FLEET_RELATIVE'] = fr
        sub = []
        for n, Mi in M.items():
            evaluate_method(Mi, n, sub, norm=NORMS[nm], mode=mode)
        for r in sub:
            r['norm'] = nm
            r['cal_span'] = mode
            r['fleet_relative'] = fr
        rows += sub
        print(f'evaluated norm={nm}, cal_span={mode}, fleet_relative={fr}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(P.RESULTS / 'gate_a3_results.csv', index=False)
    for nm, mode, fr in itertools.product(['fixed', 'rolling'], ['cal', 'devcal'], [False, True]):
        span = '2019 only' if mode == 'cal' else '2017-2019'
        rel = 'fleet-relative' if fr else 'absolute'
        d = df[(df.norm == nm) & (df.cal_span == mode) & (df.fleet_relative == fr) & (df.arl == 10)]
        print(f"\n{'=' * 74}")
        print(f'{nm} baseline | calibration {span} | {rel} index | ARL 10 t-yr')
        print('=' * 74)
        print(f"{'method':20s}" + ''.join((f'{k:>14}' for k in KS)))
        for n in M:
            line = f'{n:20s}'
            for k in KS:
                r = d[(d.method == n) & (d.k == k)]
                if not len(r):
                    line += f"{'--':>14}"
                    continue
                r = r.iloc[0]
                ld = 'none' if pd.isna(r.lead) else f'{int(r.lead)}d'
                line += f"{ld + ', ' + str(int(r.fa_turbines)) + '/7':>14}"
            print(line)
        print(f'  observed FA rate {d.fa_per_turbine_year.min():.3f}-{d.fa_per_turbine_year.max():.3f} /turbine-yr (ARL target implies 0.100) | h at grid floor: {(d.h <= 0.5).mean() * 100:.0f}%')
    print('\n' + '=' * 74)
    print('READ THIS BEFORE QUOTING ANY NUMBER')
    print('=' * 74)
    print('The ARL calibration does not hit its target in any configuration. The\nin-control index drifts upward fleet-wide across 2017-2022 in EVERY method\n(fleet ageing / cooler fouling / oil degradation), so no pre-2020 calibration\nanticipates 2022. Treat the ARL targets as a sensitivity axis and report the\nOBSERVED false-alarm rate as the primary specificity number.\n\nAbsolute vs fleet-relative changes the ranking. Report both. Fleet-relative\ncancels the common drift but needs a homogeneous fleet, so it does not apply\nto a one- or two-turbine site.')
