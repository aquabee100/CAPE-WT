import numpy as np
import pandas as pd
import paths as P
import detect_final as D
import normalise as N
import protocol as PR

def run_machine(zf, zt, k, hf, ht, horizon_days, latch=True):
    state = 'NORMAL'
    sf = st = 0.0
    warn_t = None
    episodes = []
    for t in zf.index:
        if state == 'WARNING' and (t - warn_t).days > horizon_days:
            episodes.append(dict(warning=warn_t, status='expired', confirmation=None))
            state, sf, st, warn_t = ('NORMAL', 0.0, 0.0, None)
        if state == 'WARNING':
            vt = zt.get(t, np.nan)
            if np.isfinite(vt):
                st = max(0.0, st + vt - k)
                if st > ht:
                    episodes.append(dict(warning=warn_t, status='confirmed', confirmation=t))
                    if latch:
                        return episodes
                    state, sf, st, warn_t = ('NORMAL', 0.0, 0.0, None)
            continue
        vf = zf.get(t, np.nan)
        if np.isfinite(vf):
            sf = max(0.0, sf + vf - k)
            if sf > hf:
                warn_t, state, sf, st = (t, 'WARNING', 0.0, 0.0)
    if warn_t is not None:
        episodes.append(dict(warning=warn_t, status='pending', confirmation=None))
    return episodes

def warnings_of(eps):
    return [e['warning'] for e in eps]

def confirmations_of(eps):
    return [e['confirmation'] for e in eps if e['status'] == 'confirmed']

def first_confirmed_episode(eps):
    for e in eps:
        if e['status'] == 'confirmed':
            return e
    return None

def evaluate(Ft, Tt, case, k=3.0, h=0.5, horizon=180, cal=None):
    c = PR.CASES[case]
    cal = cal or c['cal']
    event = pd.Timestamp(c['event'])
    Zf = N.rolling_z(Ft, scale_end=PR.SCALE_END[case])
    Zt = N.rolling_z(Tt, scale_end=PR.SCALE_END[case])
    sl = slice(c['monitor_from'], event)
    sl_ctrl = slice(c['monitor_from'], None)
    hf = ht = h
    eps = run_machine(Zf[case].loc[sl], Zt[case].loc[sl], k, hf, ht, horizon)
    ce = first_confirmed_episode(eps)
    n_warn = n_conf = n_pend = 0
    exposure = 0.0
    for h in D.HEALTHY:
        hfh, hth = (hf, ht)
        zf, zt = (Zf[h].loc[sl_ctrl], Zt[h].loc[sl_ctrl])
        he = run_machine(zf, zt, k, hfh, hth, horizon, latch=False)
        n_warn += len(warnings_of(he))
        n_conf += len(confirmations_of(he))
        n_pend += sum((1 for e in he if e['status'] == 'pending'))
        exposure += (zf.index[-1] - zf.index[0]).days / 365.25
    return dict(case=case, k=k, horizon=horizon, warn_lead=(event - eps[0]['warning']).days if eps else None, conf_warn_lead=(event - ce['warning']).days if ce else None, conf_lead=(event - ce['confirmation']).days if ce else None, confirm_delay=(ce['confirmation'] - ce['warning']).days if ce else None, n_episodes_fault=len(eps), n_confirmed_fault=len(confirmations_of(eps)), fp_warn_episodes=n_warn, fp_conf_episodes=n_conf, fp_pending_episodes=n_pend, exposure_yrs=round(exposure, 1), far_warn=round(n_warn / exposure, 3) if exposure else np.nan, far_conf=round(n_conf / exposure, 3) if exposure else np.nan)
if __name__ == '__main__':
    import itertools
    Ft = P.load('filter_traj')
    Tt = P.load('causal_traj')['VA']
    print('=== TWO-STAGE STATE MACHINE, k = 3.0, h = 0.5, horizon 180 d ===')
    print(f"{'case':7s}{'warn':>7s}{'confwarn':>10s}{'conf':>7s}{'delay':>7s}{'FPwarn':>8s}{'FPconf':>8s}{'FARw':>8s}{'FARc':>7s}{'expo':>7s}")
    for case in ['MF07', 'MF12']:
        r = evaluate(Ft, Tt, case)
        f = lambda v: 'none' if v is None else f'{v}d'
        print(f"{r['case']:7s}{f(r['warn_lead']):>7s}{f(r['conf_warn_lead']):>10s}{f(r['conf_lead']):>7s}{f(r['confirm_delay']):>7s}{r['fp_warn_episodes']:>8}{r['fp_conf_episodes']:>8}{r['far_warn']:>8}{r['far_conf']:>7}{r['exposure_yrs']:>7}")
    rows = [evaluate(Ft, Tt, c, k=k, horizon=hz) for c, k, hz in itertools.product(['MF07', 'MF12'], [2.0, 2.5, 3.0, 3.5, 4.0], [90, 120, 180, 270])]
    pd.DataFrame(rows).to_csv(P.RESULTS / 'state_machine_results.csv', index=False)
    print('\nwritten: results/state_machine_results.csv')
