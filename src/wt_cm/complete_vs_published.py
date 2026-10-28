from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from scipy.stats import chi2
warnings.filterwarnings('ignore')
import paths as P
import protocol as PR
import detect_final as D
import headline as H
import baseline_rule as B
import state_machine as SM
import parallel_arch as PA
HOR = 180
K, HTH = (3.0, 0.5)
CASES = ['MF07', 'MF12']
LABEL = {'MF07': 'WT6', 'MF12': 'WT8'}
LIT = {'P1': 'P1 Zaher ANN-NBM', 'P2': 'P2 Schlechtingen linear', 'P3': 'P3 GBM-NBM + EWMA', 'P5': 'P5 multivariate CUSUM'}
NATIVE_THR = {'P1': 3.0, 'P2': 3.0, 'P3': 3.0 * np.sqrt(0.2 / (2 - 0.2)), 'P5': 5.0}
CAL_WINDOW = {'MF12': ('2019-01-01', '2019-12-31', 'held-out calibration year after training, before monitoring'), 'MF07': ('2018-01-01', '2018-06-30', 'last 6 training months (in-sample; no held-out gap before WT6 monitoring)')}
NATIVE_RULE = {'P1': 'daily residual > mu+3sd (training)', 'P2': 'daily residual > mu+3sd (training)', 'P3': 'EWMA(lambda=0.2) > L=3 limit', 'P5': 'MCUSUM(k=0.5) > h=5'}

def poisson_ci(n, T):
    if not T:
        return (np.nan, np.nan, np.nan)
    lo = chi2.ppf(0.025, 2 * n) / 2 / T if n > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * n + 2) / 2 / T
    ub1 = chi2.ppf(0.95, 2 * n + 2) / 2 / T
    return (lo, hi, ub1)

def standardised_stat(code, M, case):
    if code in ('P1', 'P2'):
        tr_end = pd.Timestamp(PR.SCALE_END[case])
        tr = M.loc[:tr_end]
        return (M - tr.mean()) / tr.std()
    return M

def bounded_episodes(s, thr, horizon=HOR):
    s = s.dropna()
    starts, open_at, fresh = ([], None, True)
    for t, v in s.items():
        if open_at is not None and (t - open_at).days > horizon:
            open_at = None
        if v <= thr:
            fresh = True
            continue
        if open_at is None and fresh:
            starts.append(t)
            open_at, fresh = (t, False)
    return starts

def daily_exposure(s, end=H.RECORD_END):
    s = s.dropna()
    s = s.loc[:end]
    return len(s) / 365.25

def baseline_eval(U, case, thr):
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    mf = pd.Timestamp(c['monitor_from'])
    sc = U[case].loc[mf:ev - pd.Timedelta('1D')].dropna()
    ep = bounded_episodes(sc, thr)
    first = ep[0] if ep else None
    censored = bool(first is not None and (first - sc.index[0]).days < 15)
    n, T = (0, 0.0)
    for t in D.HEALTHY:
        if t not in U:
            continue
        s = U[t].loc[mf:].dropna()
        n += len(bounded_episodes(s, thr))
        T += daily_exposure(s)
    return dict(lead=(ev - first).days if first is not None else np.nan, censored=censored, n_ref=n, expo=T)

def threshold_grid(U, case):
    mf = PR.CASES[case]['monitor_from']
    vals = np.concatenate([U[t].loc[mf:].dropna().values for t in D.HEALTHY if t in U] + [U[case].loc[mf:].dropna().values])
    q = np.quantile(vals, np.linspace(0.5, 1.0, 600))
    return np.unique(np.r_[q, vals.max() + 1e-09])

def cape_eval(case, arch):
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    F = P.load_trajectory('filter_traj_online')
    G = P.load_trajectory('resid_gbm_adaptive')
    T = P.load_trajectory('thermal_traj_online__VA')
    Zf, Zg, Zt = (PA.zof(F, case), PA.zof(G, case), PA.zof(T, case))
    Zg = Zg.reindex(Zf.index)

    def run(turb, sl, latch):
        zf, zg, zt = (Zf[turb].loc[sl], Zg[turb].loc[sl], Zt[turb].loc[sl])
        if arch == 'filter+thermal':
            zg = None
        if arch == 'gbm+thermal':
            zf = zf * np.nan
        if arch == 'parallel warning only':
            zt = zt * np.nan
        return (PA.run_parallel(zf, zg, zt, latch=latch), zf if arch != 'gbm+thermal' else zg)
    sl, slc = (slice(c['monitor_from'], ev), slice(c['monitor_from'], None))
    eps, zcase = run(case, sl, True)
    if arch == 'gbm+thermal' and zcase.dropna().empty:
        return None
    ce = next((e for e in eps if e['status'] == 'confirmed'), None)
    nw = nc = 0
    expo = 0.0
    for t in D.HEALTHY:
        he, zref = run(t, slc, False)
        nw += len(he)
        nc += sum((e['status'] == 'confirmed' for e in he))
        expo += H.evaluable_exposure(zref, H.RECORD_END)
    first_valid = zcase.dropna().index[0] if zcase.notna().any() else None
    w_cens = bool(eps and first_valid is not None and (eps[0]['warning'] <= first_valid))
    return dict(warn_lead=(ev - eps[0]['warning']).days if eps else np.nan, warn_censored=w_cens, conf_lead=(ev - ce['confirmation']).days if ce else np.nan, n_warn=nw, n_conf=nc, expo=expo, source=eps[0].get('source') if eps else None)

def plugin_eval(code, case):
    GRID = P.load_trajectory('filter_traj_online').index
    THE = P.load_trajectory('thermal_traj_online__VA')
    M = P.load_trajectory(f'published_{case}__{code}')
    out = {}
    for col in M.columns:
        s = M[col].dropna()
        out[col] = pd.Series([s.loc[t - pd.Timedelta('45D'):t].mean() if len(s.loc[t - pd.Timedelta('45D'):t]) else np.nan for t in GRID], index=GRID)
    Mg = pd.DataFrame(out)
    kw = dict(scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
    Zw, Zt = (B.rolling_z_v2(Mg, **kw), B.rolling_z_v2(THE, **kw))
    c = PR.CASES[case]
    ev = pd.Timestamp(c['event'])
    eps = SM.run_machine(Zw[case].loc[c['monitor_from']:ev], Zt[case].loc[c['monitor_from']:ev], K, HTH, HTH, HOR)
    ce = next((e for e in eps if e['status'] == 'confirmed'), None)
    nw = nc = 0
    expo = 0.0
    for t in D.HEALTHY:
        zw, zt = (Zw[t].loc[c['monitor_from']:], Zt[t].loc[c['monitor_from']:])
        he = SM.run_machine(zw, zt, K, HTH, HTH, HOR, latch=False)
        nw += len(he)
        nc += sum((e['status'] == 'confirmed' for e in he))
        expo += H.evaluable_exposure(zw, H.RECORD_END)
    return dict(warn_lead=(ev - eps[0]['warning']).days if eps else np.nan, conf_lead=(ev - ce['confirmation']).days if ce else np.nan, n_warn=nw, n_conf=nc, expo=expo)

def row(case, system, family, op, warn_lead, action_lead, n_warn, n_action, expo, rule, censored=False, note=''):
    lo, hi, ub = poisson_ci(n_action, expo)
    return dict(case=LABEL[case], case_id=case, system=system, family=family, operating_point=op, first_alarm_lead_d=warn_lead, first_alarm_left_censored=censored, action_lead_d=action_lead, ref_warning_episodes=n_warn, ref_action_episodes=n_action, ref_exposure_ty=round(expo, 2), ref_warning_rate=round(n_warn / expo, 3) if expo else np.nan, ref_action_rate=round(n_action / expo, 3) if expo else np.nan, ref_action_rate_ci95=f'[{lo:.3f}, {hi:.3f}]', ref_action_rate_ub95_one_sided=round(ub, 3), rule=rule, note=note)

def main():
    rows, sweep = ([], [])
    for case in CASES:
        cw = cape_eval(case, 'complete')
        rows.append(row(case, 'CAPE-WT complete', 'proposed', 'fixed (Table 3)', cw['warn_lead'], cw['conf_lead'], cw['n_warn'], cw['n_conf'], cw['expo'], 'filter OR GBM warning -> thermal confirmation', cw['warn_censored'], f"first warning source: {cw['source']}"))
        for arch, lab in [('filter+thermal', 'CAPE-WT filter route + thermal'), ('gbm+thermal', 'CAPE-WT GBM route + thermal'), ('parallel warning only', 'CAPE-WT warning layer, no confirmation')]:
            r = cape_eval(case, arch)
            if r is None:
                rows.append(row(case, lab, 'ablation', 'fixed (Table 3)', np.nan, np.nan, 0, 0, 0.0, arch, note='GBM route not eligible before event'))
                continue
            act_lead = r['warn_lead'] if arch == 'parallel warning only' else r['conf_lead']
            n_act = r['n_warn'] if arch == 'parallel warning only' else r['n_conf']
            rows.append(row(case, lab, 'ablation', 'fixed (Table 3)', r['warn_lead'], act_lead, r['n_warn'], n_act, r['expo'], arch, r['warn_censored']))
        n_conf_cape, n_warn_cape = (cw['n_conf'], cw['n_warn'])
        for code, name in LIT.items():
            M = P.load_trajectory(f'published_{case}__{code}')
            U = standardised_stat(code, M, case)
            nat = baseline_eval(U, case, NATIVE_THR[code])
            rows.append(row(case, name, 'published, stand-alone', 'native', nat['lead'], nat['lead'], nat['n_ref'], nat['n_ref'], nat['expo'], NATIVE_RULE[code], nat['censored'], 'alarm = maintenance-facing decision'))
            curve = []
            for thr in threshold_grid(U, case):
                r = baseline_eval(U, case, thr)
                curve.append(dict(case=LABEL[case], method=name, threshold=float(thr), lead=r['lead'], censored=r['censored'], n_ref=r['n_ref'], expo=r['expo'], rate=r['n_ref'] / r['expo'] if r['expo'] else np.nan))
            cdf = pd.DataFrame(curve).sort_values('threshold')
            sweep.append(cdf)
            mf = pd.Timestamp(PR.CASES[case]['monitor_from'])
            cal_lo, cal_hi, cal_note = CAL_WINDOW[case]
            calv = pd.concat([U[t].loc[cal_lo:cal_hi] for t in D.HEALTHY if t in U]).dropna()
            if len(calv):
                thr_c = float(calv.max())
                rc = baseline_eval(U, case, thr_c)
                rows.append(row(case, name, 'published, stand-alone', 'causal: zero alarms on pre-monitoring calibration', rc['lead'], rc['lead'], rc['n_ref'], rc['n_ref'], rc['expo'], f'threshold {thr_c:.3f} = max healthy-fleet statistic {cal_lo}..{cal_hi}', rc['censored'], cal_note))
            for op, target in [('oracle: matched to CAPE-WT confirmed burden', n_conf_cape), ('oracle: matched to CAPE-WT warning burden', n_warn_cape)]:
                ok = cdf[cdf.n_ref <= target]
                if ok.empty:
                    continue
                b = ok.iloc[0]
                nxt = ok.iloc[1:11]['lead'].fillna(0)
                robust = float(nxt.median()) if len(nxt) else np.nan
                fragile = bool(len(nxt) and abs(np.nan_to_num(b.lead) - robust) > 30)
                note = 'threshold uses future reference data'
                if fragile:
                    note += f'; FRAGILE: lead falls to {robust:.0f} d over the next 10 thresholds'
                rows.append(row(case, name, 'published, stand-alone', op, b.lead, b.lead, int(b.n_ref), int(b.n_ref), b.expo, f'threshold {b.threshold:.3f} (retrospective)', bool(b.censored), note))
            pr = plugin_eval(code, case)
            rows.append(row(case, f'{name} + CAPE-WT escalation', 'published engine, CAPE-WT escalation', 'Page-CUSUM k=3 (Table 3)', pr['warn_lead'], pr['conf_lead'], pr['n_warn'], pr['n_conf'], pr['expo'], 'plug-in'))
    out = pd.DataFrame(rows)
    out.to_csv(P.RESULTS / 'complete_vs_published.csv', index=False)
    pd.concat(sweep).to_csv(P.RESULTS / 'complete_vs_published_sweep.csv', index=False)
    show = ['case', 'system', 'operating_point', 'first_alarm_lead_d', 'first_alarm_left_censored', 'action_lead_d', 'ref_warning_episodes', 'ref_action_episodes', 'ref_exposure_ty', 'ref_action_rate', 'ref_action_rate_ub95_one_sided']
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        for case in CASES:
            print(f'\n=== {LABEL[case]} ({case}) ===')
            print(out[out.case_id == case][show].to_string(index=False))
    print('\nwritten: results/runtime/complete_vs_published.csv, complete_vs_published_sweep.csv')
if __name__ == '__main__':
    main()
