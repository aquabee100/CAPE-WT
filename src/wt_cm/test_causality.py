import sys
import numpy as np
import pandas as pd
import causal as C
import filter_ch as F
import normalise as N
import paths as P
import protocol as PR
import thermal_id as ti
T0 = pd.Timestamp('2020-06-15')
BIG = 999999.0

def _clean_v2(tf):
    return C.causal_clean_v2(tf, 'T_oil').astype(float)

def _clean_v2_nointerp(tf):
    ok = tf['P'].between(50, 2100) & (tf['rpm'] > 3) & tf['T_oil'].between(5, 90) & tf['T_nac'].between(-10, 60) & tf['T_amb'].between(-20, 50) & (tf['run'] > 540) & (tf['alarm'] < 60) & ~C.causal_flatline_mask(tf['T_oil']) & ~C.causal_flatline_mask(tf['P'])
    full = ok.copy()
    for k in range(1, C.NSTENCIL):
        full &= ok.shift(k).fillna(False).astype(bool)
    return full.astype(float)

def _resistance_online(tf):
    import models as MD
    dp = tf['p_before'] - tf['p_after']
    ok = tf['P'].between(500, 2100) & (tf['rpm'] > 8) & tf['T_oil'].between(20, 90) & dp.between(0.01, 20) & (tf['run'] > 540) & (tf['alarm'] < 60)
    for c in ['p_before', 'p_after', 'T_oil', 'rpm']:
        ok &= ~C.causal_flatline_mask(tf[c])
    omega = tf['rpm'] * 2 * np.pi / 60.0
    return (dp / (MD.phi(tf['T_oil']) * omega)).where(ok)
ONLINE_CHECKS = [('causal_derivative', lambda tf: C.causal_derivative(tf['T_oil'])), ('align_regressors', lambda tf: C.align_regressors(tf[['P', 'rpm']])['P']), ('causal_flatline_mask', lambda tf: C.causal_flatline_mask(tf['T_oil']).astype(float)), ('online_clean_nointerp', _clean_v2_nointerp), ('online_filter_resistance', _resistance_online)]
ARCHIVE_AUDIT = [('interpolated_mask', lambda tf: C.interpolated_mask(tf['T_oil']).astype(float)), ('archive_clean', _clean_v2), ('archive_filter_resistance', lambda tf: F.resistance(tf))]
CHECKS = [(n, f, True) for n, f in ONLINE_CHECKS] + [(n, f, False) for n, f in ARCHIVE_AUDIT]

def corrupt(tf, t0=T0, mode='constant'):
    out = tf.copy()
    fut = out.index > t0
    if not fut.any():
        return out
    n = int(fut.sum())
    if mode == 'constant':
        out.loc[fut, :] = BIG
    elif mode == 'ramp':
        last = out.loc[~fut].iloc[-1]
        slope = out.loc[~fut].diff().iloc[-1].fillna(0.0)
        step = np.arange(1, n + 1)
        for c in out.columns:
            out.loc[fut, c] = (last[c] + slope[c] * step).astype('float64')
    elif mode == 'freeze':
        out.loc[fut, :] = out.loc[~fut].iloc[-1].values
    elif mode == 'noise':
        rng = np.random.default_rng(0)
        out.loc[fut, :] = rng.normal(0, 10000.0, size=(n, out.shape[1]))
    elif mode == 'reverse':
        out.loc[fut, :] = out.loc[~fut].iloc[::-1].reindex(range(n), method=None).ffill().bfill().values[:n] if False else out.loc[fut, :].iloc[::-1].values
    return out
MODES = ['constant', 'ramp', 'freeze', 'noise', 'reverse']

def compare(a, b, t0=T0):
    a, b = (a.loc[:t0], b.loc[:t0])
    same = a.fillna(-12345) == b.fillna(-12345)
    n = int((~same).sum())
    first = same.index[~same][0] if n else None
    return (n, first, (t0 - first).days if first is not None else None)

def adversarial_cuts(tf, n_run=8, n_rand=4, seed=0):
    m = C.interpolated_mask(tf['T_oil'])
    inside = tf.index[m.values]
    rng = np.random.default_rng(seed)
    cuts = []
    if len(inside):
        cuts += list(rng.choice(inside, size=min(n_run, len(inside)), replace=False))
    body = tf.index[(tf.index > tf.index[0] + pd.Timedelta('120D')) & (tf.index < tf.index[-1] - pd.Timedelta('120D'))]
    cuts += list(rng.choice(body, size=n_rand, replace=False))
    return sorted((pd.Timestamp(c) for c in cuts))

def audit_pointwise(tf, cuts=None):
    cuts = cuts if cuts is not None else adversarial_cuts(tf)
    rows = []
    for name, fn, expect in CHECKS:
        base = fn(tf)
        worst = dict(n_changed=0, reach_days=None, modes=set(), ncuts=0)
        for t0 in cuts:
            hit = False
            for m in MODES:
                n, first, reach = compare(base, fn(corrupt(tf, t0=t0, mode=m)), t0=t0)
                if n:
                    hit = True
                    worst['modes'].add(m)
                    worst['n_changed'] = max(worst['n_changed'], n)
                    if reach is not None:
                        worst['reach_days'] = max(worst['reach_days'] or 0, reach)
            worst['ncuts'] += hit
        rows.append(dict(component=name, declared='ONLINE' if expect else 'OFFLINE', cuts_leaking=f"{worst['ncuts']}/{len(cuts)}", n_changed=worst['n_changed'], reach_days=worst['reach_days'], modes=','.join(sorted(worst['modes'])) or '-', verdict='LEAK' if worst['ncuts'] else 'clean'))
    return pd.DataFrame(rows)

def audit_trajectory(tf):
    base = F.trajectory(tf)
    idx = base.index[base.index <= T0]
    tot, first, modes = (0, None, [])
    for m in MODES:
        d = (base.loc[idx] - F.trajectory(corrupt(tf, mode=m)).reindex(idx)).abs()
        n = int((d > 1e-12).sum())
        if n:
            modes.append(m)
            tot = max(tot, n)
            f = d[d > 1e-12].index[0]
            first = f if first is None or f < first else first
    return dict(component='filter trajectory', declared='ONLINE', n_changed=tot, first_bad=first, reach_days=(T0 - first).days if first is not None else None, modes=','.join(modes) or '-', verdict='LEAK' if modes else 'clean')

def audit_rolling_z(M):
    Z = N.rolling_z(M, scale_end=PR.SCALE_END['MF12'])
    idx = Z.index[Z.index <= T0]
    tot, modes = (0, [])
    for m in MODES:
        Zc = N.rolling_z(corrupt(M, mode=m), scale_end=PR.SCALE_END['MF12'])
        n = int(((Z.loc[idx] - Zc.loc[idx]).abs() > 1e-12).sum().sum())
        if n:
            modes.append(m)
            tot = max(tot, n)
    return dict(component='rolling_z', declared='ONLINE', n_changed=tot, first_bad=None, reach_days=None, modes=','.join(modes) or '-', verdict='LEAK' if modes else 'clean')

def baseline_span(M):
    rows = []
    for case in ['MF07', 'MF12']:
        start = pd.Timestamp(PR.CASES[case]['monitor_from'])
        for t in M.index[M.index >= start][:4]:
            lo, hi = (t - N.ROLL_EMBARGO - N.ROLL_WINDOW, t - N.ROLL_EMBARGO)
            base = M.loc[lo:hi]
            counts = base.notna().sum()
            span = (base.index[-1] - base.index[0]).days if len(base) else 0
            rows.append(dict(case=case, window=t.date(), span_days=span, requested_days=365, min_valid=int(counts.min()), max_valid=int(counts.max()), partial=span < 360))
    return pd.DataFrame(rows)
if __name__ == '__main__':
    df = ti.load_raw()
    tf = F.frame(df, 'MF12').astype('float64')
    del df
    cuts = adversarial_cuts(tf)
    print('=== pointwise leakage audit (MF12, %d adversarial cut points, %d corruption modes) ===' % (len(cuts), len(MODES)))
    res = audit_pointwise(tf, cuts)
    res = pd.concat([res, pd.DataFrame([audit_trajectory(tf)])], ignore_index=True)
    print(res.to_string(index=False))
    M = pd.read_parquet(P.RESULTS / 'filter_traj.parquet').astype('float64')
    print()
    print(pd.DataFrame([audit_rolling_z(M)]).to_string(index=False))
    print('\n=== trailing-baseline span at each monitoring start ===')
    print(baseline_span(M).to_string(index=False))
    res.to_csv(P.RESULTS / 'leakage_audit.csv', index=False)
    baseline_span(M).to_csv(P.RESULTS / 'baseline_span.csv', index=False)
    print(f'\nwritten: {P.RESULTS / "leakage_audit.csv"}, {P.RESULTS / "baseline_span.csv"}')
    res['track'] = np.where(res['declared'] == 'ONLINE', 'A/gate', 'B/audit')
    res.loc[res['declared'] == 'OFFLINE', 'declared'] = 'EXPECTED_NONCAUSAL'
    ctrl = res[res['component'] == 'interpolated_mask']
    if len(ctrl) and ctrl['verdict'].iloc[0] != 'LEAK':
        print('\nHARNESS BROKEN: the known-noncausal positive control reported clean. Do not trust any result above.')
        sys.exit(2)
    bad = res[(res['track'] == 'A/gate') & (res['verdict'] == 'LEAK')]
    if len(bad):
        print('\nGATE FAIL: %d strict-online component(s) leak future data:' % len(bad))
        print('  ' + ', '.join(bad['component']))
        sys.exit(1)
    print('\nGATE PASS: every strict-online component is causal. Archive-screen leaks recorded as expected, not failed.')
