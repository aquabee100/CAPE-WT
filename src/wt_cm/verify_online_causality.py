from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import causal as C
import filter_ch as FCH
import online_core as OC
import online_2x2 as O2
import parallel_arch as PA
import paths as P
import protocol as PR
import thermal_id as TI
MODES = ['constant', 'ramp', 'freeze', 'noise', 'reverse']
CASE = 'MF12'
DEFAULT_CUTOFF = pd.Timestamp('2021-10-15')
GBM_CHECK_TIMES = [pd.Timestamp('2021-07-24'), pd.Timestamp('2021-08-08'), pd.Timestamp('2021-10-07')]
DEEP_GBM_CUTOFFS = [pd.Timestamp('2021-08-08'), pd.Timestamp('2021-10-07')]
ATOL = 1e-11

def req(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)
    print('[PASS]', msg, flush=True)

def combined_frame(df: pd.DataFrame, turbine: str) -> pd.DataFrame:
    x = FCH.frame(df, turbine)
    t = TI.turbine_frame(df, turbine)
    x['T_gbb'] = t['T_gbb']
    return x

def corrupt_future(tf: pd.DataFrame, cutoff: pd.Timestamp, mode: str) -> pd.DataFrame:
    out = tf.copy()
    fut = out.index > cutoff
    if not fut.any():
        return out
    cols = list(out.select_dtypes(include=[np.number]).columns)
    n = int(fut.sum())
    past = out.loc[~fut, cols]
    if mode == 'constant':
        for c in cols:
            out.loc[fut, c] = np.full(n, 999999.0, dtype=out[c].dtype)
    elif mode == 'ramp':
        last = past.iloc[-1]
        slope = past.diff().iloc[-1].fillna(0.0)
        step = np.arange(1, n + 1, dtype=float)
        for c in cols:
            vals = (float(last[c]) + float(slope[c]) * step).astype(out[c].dtype)
            out.loc[fut, c] = vals
    elif mode == 'freeze':
        for c in cols:
            out.loc[fut, c] = np.full(n, past[c].iloc[-1], dtype=out[c].dtype)
    elif mode == 'noise':
        rng = np.random.default_rng(20260904)
        for c in cols:
            out.loc[fut, c] = rng.normal(0.0, 10000.0, size=n).astype(out[c].dtype)
    elif mode == 'reverse':
        for c in cols:
            out.loc[fut, c] = out.loc[fut, c].iloc[::-1].to_numpy(dtype=out[c].dtype)
    else:
        raise ValueError(mode)
    return out

def maxdiff(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.union(b.index)
    x, y = (a.reindex(idx), b.reindex(idx))
    na = x.isna() & y.isna()
    bad_na = x.isna() ^ y.isna()
    if bad_na.any():
        return float('inf')
    d = (x[~na].astype(float) - y[~na].astype(float)).abs()
    return float(d.max()) if len(d) else 0.0

def assert_series_equal(a, b, label, atol=ATOL):
    d = maxdiff(pd.Series(a), pd.Series(b))
    req(np.isfinite(d) and d <= atol, f'{label}: max |delta| = {d:.3g}')
    return d

def raw_outputs_to_cutoff(tf, cutoff):
    f = OC.filter_trajectory(tf, end_limit=cutoff)
    t = O2.phys_coef_index(tf, end_limit=cutoff)
    g = {e: O2.gbm_score_at(tf, e) for e in GBM_CHECK_TIMES if e <= cutoff}
    return (f, g, t)

def _gbm_window_task(end_ns, idx_ns, Xv, yv):
    end = pd.Timestamp(end_ns)
    lo, hi = (end - C.WINDOW - O2.LOOKBACK, end - C.WINDOW)
    a = int(np.searchsorted(idx_ns, lo.value, side='left'))
    b = int(np.searchsorted(idx_ns, hi.value, side='right'))
    if b - a < 2000:
        return (end_ns, np.nan)
    if hi.value - int(idx_ns[a]) < int(O2.LOOKBACK.value * 0.9):
        return (end_ns, np.nan)
    c = int(np.searchsorted(idx_ns, (end - C.WINDOW).value, side='left'))
    d = int(np.searchsorted(idx_ns, end.value, side='right'))
    if d - c < 400:
        return (end_ns, np.nan)
    with threadpool_limits(limits=1):
        mdl = HistGradientBoostingRegressor(random_state=0, **O2.GBM_PARAMS).fit(Xv[a:b:O2.STRIDE], yv[a:b:O2.STRIDE])
        val = float((yv[c:d] - mdl.predict(Xv[c:d])).mean())
    return (end_ns, val)

def deep_gbm_residual(tf, end_limit, n_jobs=4):
    ok = O2.online_clean(tf)
    X, y = (tf.loc[ok, O2.FEATS], tf.loc[ok, O2.TC])
    m = X.notna().all(axis=1) & y.notna()
    X, y = (X[m], y[m])
    idx_ns = X.index.view('i8')
    Xv = X.to_numpy()
    yv = y.to_numpy()
    lim = pd.Timestamp(end_limit)
    ends = [e for e in C.trailing_grid(tf.index) if e <= lim]
    jobs = max(1, min(int(n_jobs), os.cpu_count() or 1))
    res = Parallel(n_jobs=jobs, backend='loky', max_nbytes='10M')((delayed(_gbm_window_task)(int(e.value), idx_ns, Xv, yv) for e in ends))
    vals = {pd.Timestamp(ns): float(v) for ns, v in res if np.isfinite(v)}
    return pd.Series(vals, dtype=float).sort_index()

def deep_gbm_gate(tf, Gbase, modes, cutoffs, n_jobs):
    print('\n=== DEEP adaptive-GBM raw future-invariance gate ===', flush=True)
    print(f'parallel refits per replay: n_jobs={n_jobs}; cutoffs={[str(x.date()) for x in cutoffs]}', flush=True)
    worker = Path(__file__).with_name('verify_deep_gbm_causality.py')
    rows = []
    for cutoff in cutoffs:
        for mode in modes:
            print(f'[deep GBM cutoff={cutoff.date()} mode={mode}]', flush=True)
            cp = subprocess.run([sys.executable, str(worker), '--cutoff', str(cutoff.date()), '--mode', mode, '--jobs', str(n_jobs)], cwd=Path(__file__).resolve().parent, capture_output=True, text=True)
            if cp.stdout:
                print(cp.stdout.rstrip(), flush=True)
            if cp.returncode != 0:
                if cp.stderr:
                    print(cp.stderr.rstrip(), flush=True)
                raise RuntimeError(f'deep adaptive-GBM worker failed at {cutoff.date()} / {mode}')
            line = next((x for x in cp.stdout.splitlines() if x.startswith('RESULT_JSON ')), None)
            if line is None:
                raise RuntimeError('deep adaptive-GBM worker returned no RESULT_JSON')
            rows.append(json.loads(line[len('RESULT_JSON '):]))
    out = pd.DataFrame(rows)
    out.to_csv(P.RESULTS / 'online_causality_deep_gbm.csv', index=False)
    req(len(out) == len(modes) * len(cutoffs) and out.passed.all(), f'deep adaptive-GBM gate passed {len(out)}/{len(out)} hostile raw replays')
    req(float(out.max_gbm_delta.max()) <= ATOL, f'deep adaptive-GBM maximum pre-cutoff delta = {float(out.max_gbm_delta.max()):.3g}')
    print(f"written: {P.RESULTS / 'online_causality_deep_gbm.csv'}", flush=True)
    return out

def corrupt_matrix_future(M, cutoff, mode):
    out = M.copy()
    fut = out.index > cutoff
    if not fut.any():
        return out
    n, p = (int(fut.sum()), out.shape[1])
    if mode == 'constant':
        out.loc[fut, :] = 1000000000000.0
    elif mode == 'ramp':
        last = out.loc[~fut].iloc[-1].fillna(0.0)
        step = np.arange(1, n + 1, dtype=float)
        for c in out.columns:
            out.loc[fut, c] = last[c] + step * 1000000.0
    elif mode == 'freeze':
        out.loc[fut, :] = out.loc[~fut].iloc[-1].to_numpy()
    elif mode == 'noise':
        rng = np.random.default_rng(17)
        out.loc[fut, :] = rng.normal(0, 1000000000.0, size=(n, p))
    elif mode == 'reverse':
        out.loc[fut, :] = out.loc[fut, :].iloc[::-1].to_numpy()
    return out

def inject_target(base, s, cutoff):
    out = base.copy()
    idx = out.index[out.index <= cutoff]
    out.loc[idx, CASE] = s.reindex(idx)
    return out

def episode_signature(Fm, Gm, Tm, cutoff):
    zf, zg, zt = (PA.zof(Fm, CASE), PA.zof(Gm, CASE), PA.zof(Tm, CASE))
    start = PR.CASES[CASE]['monitor_from']
    zfs = zf[CASE].loc[start:cutoff]
    zgs = zg[CASE].reindex(zfs.index)
    zts = zt[CASE].reindex(zfs.index)
    eps = PA.run_parallel(zfs, zgs, zts, latch=True)
    sig = tuple(((pd.Timestamp(e['warning']).isoformat(), e['status'], None if e.get('confirmation') is None else pd.Timestamp(e['confirmation']).isoformat(), e.get('source')) for e in eps))
    if any((e['status'] == 'confirmed' for e in eps)):
        final_state = 'CONFIRMED'
    elif eps and eps[-1]['status'] == 'pending':
        final_state = 'WARNING'
    else:
        final_state = 'NORMAL'
    return (zf[CASE].loc[:cutoff], zg[CASE].loc[:cutoff], zt[CASE].loc[:cutoff], sig, final_state)

def primitive_gate(tf, cutoff):
    base = {'online_clean': OC.online_clean(tf, 'T_gbb').astype(float), 'filter_resistance': OC.filter_resistance(tf), 'causal_derivative': C.causal_derivative(tf['T_gbb']), 'aligned_power': C.align_regressors(tf[['P']])['P']}
    for mode in MODES:
        bad = corrupt_future(tf, cutoff, mode)
        cur = {'online_clean': OC.online_clean(bad, 'T_gbb').astype(float), 'filter_resistance': OC.filter_resistance(bad), 'causal_derivative': C.causal_derivative(bad['T_gbb']), 'aligned_power': C.align_regressors(bad[['P']])['P']}
        for name in base:
            assert_series_equal(base[name].loc[:cutoff], cur[name].loc[:cutoff], f'primitive {name} / {mode}', atol=0.0 if name == 'online_clean' else ATOL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', default=str(DEFAULT_CUTOFF.date()))
    ap.add_argument('--modes', default=','.join(MODES), help='comma-separated end-to-end corruption modes')
    ap.add_argument('--skip-primitives', action='store_true')
    ap.add_argument('--deep-gbm', action='store_true', help='re-derive the complete adaptive-GBM path under raw future corruption')
    ap.add_argument('--deep-gbm-cutoffs', default=','.join((str(x.date()) for x in DEEP_GBM_CUTOFFS)), help='comma-separated cutoffs for full raw adaptive-GBM replay')
    ap.add_argument('--deep-gbm-jobs', type=int, default=min(4, os.cpu_count() or 1), help='parallel independent GBM refits; scheduling only, calculations unchanged')
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f'unknown mode: {m}')
    deep_cutoffs = [pd.Timestamp(x.strip()) for x in args.deep_gbm_cutoffs.split(',') if x.strip()]
    if args.deep_gbm and (not deep_cutoffs):
        raise SystemExit('--deep-gbm requires at least one cutoff')
    print(f'loading raw primary fleet from {TI.PKL}', flush=True)
    df = TI.load_raw()
    tf = combined_frame(df, CASE)
    del df
    if not args.skip_primitives:
        print('\n=== primitive future-invariance gate (5 hostile modes) ===', flush=True)
        primitive_gate(tf, cutoff)
    print('\n=== production-path raw reconstruction to cutoff ===', flush=True)
    f0, g0, t0 = raw_outputs_to_cutoff(tf, cutoff)
    Fbase = P.load_trajectory('filter_traj_online')
    Gbase = P.load_trajectory('resid_gbm_adaptive')
    Tbase = P.load_trajectory('thermal_traj_online__VA')
    assert_series_equal(f0, Fbase[CASE].loc[:cutoff].dropna(), 'raw->filter vs frozen')
    assert_series_equal(t0, Tbase[CASE].loc[:cutoff].dropna(), 'raw->thermal vs frozen')
    for e, v in g0.items():
        d = abs(float(v) - float(Gbase.loc[e, CASE]))
        req(d <= ATOL, f'raw->adaptive GBM at {e.date()} vs frozen: |delta|={d:.3g}')
    if args.deep_gbm:
        deep_gbm_gate(tf, Gbase, modes, deep_cutoffs, args.deep_gbm_jobs)
    base_sig = episode_signature(Fbase, Gbase, Tbase, cutoff)
    req(base_sig[4] == 'CONFIRMED', 'baseline MF12 state at cutoff is CONFIRMED')
    req(any((e[1] == 'confirmed' for e in base_sig[3])), 'baseline MF12 episode contains a confirmation')
    rows = []
    print('\n=== end-to-end future-invariance gate ===', flush=True)
    for mode in modes:
        print(f'\n[mode={mode}] recomputing target trajectories', flush=True)
        bad = corrupt_future(tf, cutoff, mode)
        f1, g1, t1 = raw_outputs_to_cutoff(bad, cutoff)
        dfilt = assert_series_equal(f0, f1, f'filter trajectory / {mode}')
        dtherm = assert_series_equal(t0, t1, f'thermal trajectory / {mode}')
        dgbm = 0.0
        for e in g0:
            d = abs(float(g0[e]) - float(g1[e]))
            dgbm = max(dgbm, d)
            req(d <= ATOL, f'adaptive GBM score {e.date()} / {mode}: |delta|={d:.3g}')
        Fm = inject_target(Fbase, f1, cutoff)
        Gm = Gbase.copy()
        for e, v in g1.items():
            Gm.loc[e, CASE] = v
        Tm = inject_target(Tbase, t1, cutoff)
        Fm = corrupt_matrix_future(Fm, cutoff, mode)
        Gm = corrupt_matrix_future(Gm, cutoff, mode)
        Tm = corrupt_matrix_future(Tm, cutoff, mode)
        sig = episode_signature(Fm, Gm, Tm, cutoff)
        dzf = assert_series_equal(base_sig[0], sig[0], f'normalised filter z / {mode}')
        dzg = assert_series_equal(base_sig[1], sig[1], f'normalised GBM z / {mode}')
        dzt = assert_series_equal(base_sig[2], sig[2], f'normalised thermal z / {mode}')
        req(sig[3] == base_sig[3], f'episode warning/source/confirmation invariant / {mode}')
        req(sig[4] == base_sig[4] == 'CONFIRMED', f'final state invariant / {mode}')
        rows.append(dict(mode=mode, cutoff=cutoff, max_filter_delta=dfilt, max_gbm_delta=dgbm, max_thermal_delta=dtherm, max_z_filter_delta=dzf, max_z_gbm_delta=dzg, max_z_thermal_delta=dzt, final_state=sig[4], passed=True))
    out = pd.DataFrame(rows)
    out.to_csv(P.RESULTS / 'online_causality_end_to_end.csv', index=False)
    print('\nALL END-TO-END FUTURE-INVARIANCE ASSERTIONS PASSED')
    print(f"written: {P.RESULTS / 'online_causality_end_to_end.csv'}")
if __name__ == '__main__':
    main()
