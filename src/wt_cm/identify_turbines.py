import numpy as np
import pandas as pd
import thermal_id as ti
FLEET_UP = 6
PRODUCING = 100.0
STOPPED = 10.0

def anomalous_downtime(frames, window):
    P = pd.DataFrame({t: frames[t]['P'] for t in frames})
    up = (P > PRODUCING).sum(axis=1)
    out = {}
    for t in frames:
        m = ((P[t] <= STOPPED) & (up >= FLEET_UP)).loc[window]
        g = (m != m.shift()).cumsum()
        runs = m.groupby(g).agg(['sum', 'size'])
        longest = runs[runs['sum'] > 0]['size'].max() if (runs['sum'] > 0).any() else 0
        out[t] = dict(hours=m.sum() * 10 / 60, longest_h=(longest or 0) * 10 / 60)
    return pd.DataFrame(out).T
if __name__ == '__main__':
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    print('=== WT8: gearbox failure 23 Feb 2022 ===')
    d = anomalous_downtime(frames, slice('2022-02-20', '2022-03-31'))
    print(d.sort_values('hours', ascending=False).round(1).to_string())
    print('-> WT8 = MF12 (also matches the published failure date exactly)\n')
    print('=== WT6: 12 h maintenance 30 May 2019 ===')
    d = anomalous_downtime(frames, slice('2019-05-25', '2019-06-05'))
    print(d.sort_values('hours', ascending=False).round(1).to_string())
    print('\n=== corroboration: top anomalous-downtime days of 2019 ===')
    P = pd.DataFrame({t: frames[t]['P'] for t in frames})
    up = (P > PRODUCING).sum(axis=1)
    for t in frames:
        m = ((P[t] <= STOPPED) & (up >= FLEET_UP)).loc['2019']
        top = (m.resample('D').sum() * 10 / 60).sort_values(ascending=False).head(3)
        print(f'  {t}: ' + ', '.join((f'{i.date()} {v:.1f}h' for i, v in top.items())))
    print('\n-> MF07 is the most likely mapping to WT6: it is the only turbine with\n   substantial anomalous downtime in the maintenance window, and 2019-05-29 —\n   the day before the published maintenance date — is among its highest\n   anomalous-downtime days of the year. No other turbine shows either feature.\n\n   Report as "most likely mapping", not as an established identity: no\n   maintenance records were available, and the inference rests on downtime\n   patterns alone.')
