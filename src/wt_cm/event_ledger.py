import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR, state_machine as SM
import baseline_rule as B, thermal_id as ti, filter_ch as F
DOCUMENTED = pd.DataFrame([dict(turbine='MF12', date='2022-02-23', kind='gearbox failure', provenance='documented', confidence='high'), dict(turbine='MF07', date='2019-05-30', kind='gearbox maintenance', provenance='documented', confidence='secondary/mapping-dependent')]).assign(date=lambda d: pd.to_datetime(d['date']))

def nearest(events, turbine, when):
    e = events[events['turbine'] == turbine]
    if not len(e):
        return (None, np.nan)
    dt = (e['date'] - when).dt.days
    i = dt.abs().idxmin()
    return (e.loc[i, 'date'].date(), int(dt.loc[i]))

def op_state(frames, turbine, when, win=30):
    f = frames[turbine].loc[when - pd.Timedelta(days=win):when + pd.Timedelta(days=win)]
    if not len(f):
        return {}
    return dict(alarm_hours=round(float((f['alarm'] > 300).sum()) * 10 / 60, 1), stopped_hours=round(float((f['run'] < 300).sum()) * 10 / 60, 1), mean_power=round(float(f['P'].mean()), 1))
if __name__ == '__main__':
    Tt = P.load('causal_traj')['VA']
    Ft = P.load_trajectory('filter_traj_online')
    iv = pd.read_csv(P.RESULTS / 'interventions.csv', parse_dates=['start', 'end'])
    iv = iv.rename(columns={'start': 'date'})
    df = ti.load_raw()
    frames = {t: F.frame(df, t) for t in ti.TURBINES}
    del df
    rows = []
    for case in ['MF07', 'MF12']:
        c = PR.CASES[case]
        Zf = B.rolling_z_v2(Ft, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=360)
        Zt = B.rolling_z_v2(Tt, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=360)
        for t in D.HEALTHY + [case]:
            sl = slice(c['monitor_from'], None if t in D.HEALTHY else c['event'])
            eps = SM.run_machine(Zf[t].loc[sl], Zt[t].loc[sl], 3.0, 0.5, 0.5, 180, latch=t == case)
            for e in eps:
                w = e['warning']
                dd, ddd = nearest(DOCUMENTED, t, w)
                di, dii = nearest(iv, t, w)
                rows.append(dict(case=case, turbine=t, role='fault' if t == case else 'control', warning=w.date(), status=e['status'], confirmation=e['confirmation'].date() if e['confirmation'] else None, doc_event=dd, days_to_doc=ddd, derived_isolation=di, days_to_derived=dii, **op_state(frames, t, w)))
    led = pd.DataFrame(rows).sort_values(['case', 'role', 'turbine', 'warning'])
    print(led.to_string(index=False))
    led.to_csv(P.RESULTS / 'event_ledger.csv', index=False)
    print('\nwritten: results/event_ledger.csv')
    print('\ndistance-to-derived-isolation, control warnings only (days, signed):')
    ctl = led[led['role'] == 'control']['days_to_derived'].dropna()
    print(f'  n={len(ctl)}  min {ctl.min():.0f}  q25 {ctl.quantile(0.25):.0f}  median {ctl.median():.0f}  q75 {ctl.quantile(0.75):.0f}  max {ctl.max():.0f}')
