import numpy as np
import pandas as pd
import paths as P
import thermal_id as ti
FLEET_UP = 6
PRODUCING = 100.0
OFFGRID = 60.0
MIN_HOURS = 2.0
MERGE_GAP = '12h'
STEP_WINDOW = '90D'

def isolation_mask(frames):
    P_ = pd.DataFrame({t: frames[t]['P'] for t in frames})
    up = (P_ > PRODUCING).sum(axis=1)
    return pd.DataFrame({t: (frames[t]['gridon'] < OFFGRID) & (up >= FLEET_UP) for t in frames})

def episodes(mask):
    out = []
    for t in mask.columns:
        m = mask[t]
        grp = (m != m.shift()).cumsum()
        runs = [idx for _, idx in m[m].groupby(grp[m])]
        cur = None
        for idx in runs:
            start, end, hrs = (idx.index[0], idx.index[-1], len(idx) * 10 / 60)
            if cur and start - cur['end'] < pd.Timedelta(MERGE_GAP):
                cur['end'], cur['hours'] = (end, cur['hours'] + hrs)
            else:
                if cur:
                    out.append(cur)
                cur = dict(turbine=t, start=start, end=end, hours=hrs)
        if cur:
            out.append(cur)
    df = pd.DataFrame(out)
    return df[df['hours'] >= MIN_HOURS].sort_values('start').reset_index(drop=True)

def dp_step(frames, eps):
    rows = []
    for _, e in eps.iterrows():
        f = frames[e['turbine']]
        dp = f['p_before'] - f['p_after']
        ok = f['P'].between(500, 2100) & (f['rpm'] > 8) & f['T_oil'].between(20, 90) & dp.between(0.01, 20)
        dp = dp.where(ok)
        w = pd.Timedelta(STEP_WINDOW)
        pre = dp.loc[e['start'] - w:e['start'] - pd.Timedelta('1D')].mean()
        post = dp.loc[e['end'] + pd.Timedelta('2D'):e['end'] + w].mean()
        rows.append(dict(turbine=e['turbine'], start=e['start'], hours=e['hours'], dp_pre=pre, dp_post=post, ratio=post / pre if pre and pre == pre else np.nan))
    return pd.DataFrame(rows)
if __name__ == '__main__':
    df = ti.load_raw()
    frames = {}
    for t in ti.TURBINES:
        f = ti.turbine_frame(df, t)
        cols = [c for c in df.columns if str(c).startswith(t + '_')]
        for name, sig in [('gridon', 'HourCounters Average GridOn Avg'), ('p_before', 'Gear Filter BeforeInlinePressureOil'), ('p_after', 'Gear Filter AfterInlinePressureOil')]:
            f[name] = df[[c for c in cols if sig in str(c)][0]].astype('float32')
        frames[t] = f
    del df
    eps = episodes(isolation_mask(frames))
    step = dp_step(frames, eps)
    eps.to_csv(P.RESULTS / 'interventions.csv', index=False)
    step.to_csv(P.RESULTS / 'intervention_dp_step.csv', index=False)
    print(step.round(4).to_string(index=False))
    r = step['ratio'].dropna()
    print(f'\nepisodes: {len(eps)}  |  median |1-ratio|: {np.median(np.abs(1 - r)):.3f}')
    print('longest episode:', eps.loc[eps['hours'].idxmax()].to_dict())
