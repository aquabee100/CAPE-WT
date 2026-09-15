from pathlib import Path
import numpy as np
import pandas as pd
import kelm_k1 as K
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
HEADER_ROW = 9
CHANNELS = {'T_oil': 'Gear oil temperature (°C)', 'T_oil_in': 'Gear oil inlet temperature (°C)', 'P_inlet': 'Gear oil inlet pressure (bar)', 'P_pump': 'Gear oil pump pressure (bar)', 'particles': 'Metal particle count', 'power': 'Power (kW)', 'rpm': 'Rotor speed (RPM)', 'T_bearing': 'Gear bearing temperature (°C)'}
CASES = {'KWF1 recurrent oil-filter episode': dict(turbine=1, block='2019-01-01_-_2020-01-01', intervals=[('2019-02-24 16:46:28', '2019-04-04 12:35:45'), ('2019-04-09 10:09:03', '2019-05-21 10:06:24'), ('2019-05-28 20:55:45', '2019-05-30 07:34:04')]), 'KWF6 bearing-temperature case': dict(turbine=6, block='2021-01-01_-_2021-07-01', intervals=[('2021-01-21 08:08:49', '2021-01-21 08:39:04'), ('2021-03-27 05:18:50', '2021-03-27 06:27:02')]), 'KWF2 gear-speed case': dict(turbine=2, block='2021-01-01_-_2021-07-01', intervals=[('2021-03-09 11:11:25', '2021-03-09 11:20:21')]), 'KWF4 oil-pump context': dict(turbine=4, block='2021-01-01_-_2021-07-01', intervals=[('2021-05-14 14:02:14', '2021-05-14 14:02:22')])}

def load_channels(turb, block):
    f = list(RAW.glob(f'Turbine_Data_Kelmarsh_{turb}_{block}*.csv'))[0]
    df = pd.read_csv(f, skiprows=HEADER_ROW, low_memory=False)
    idx = pd.to_datetime(df.iloc[:, 0], errors='coerce')
    out = pd.DataFrame(index=idx)
    for k, col in CHANNELS.items():
        out[k] = pd.to_numeric(df[col], errors='coerce').values if col in df.columns else np.nan
    return out[out.index.notna()].sort_index()

def summarise(s, a, b):
    seg = s.loc[a:b].dropna()
    return (round(float(seg.median()), 4), len(seg)) if len(seg) else (np.nan, 0)
if __name__ == '__main__':
    missing = None
    rows, per_interval = ([], [])
    for case, cfg in CASES.items():
        t, blk = (cfg['turbine'], cfg['block'])
        f = load_channels(t, blk)
        if missing is None:
            df0 = pd.read_csv(list(RAW.glob(f'Turbine_Data_Kelmarsh_{t}_{blk}*.csv'))[0], skiprows=HEADER_ROW, nrows=1, low_memory=False)
            missing = [c for c in CHANNELS.values() if c not in df0.columns]
            if missing:
                print('channels absent from archive:', missing, '\n')
        peers = {p: load_channels(p, blk) for p in range(1, 7) if p != t}
        iv = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cfg['intervals']]
        env0, env1 = (iv[0][0], iv[-1][1])
        pre0, post1 = (env0 - pd.Timedelta('90D'), env1 + pd.Timedelta('30D'))
        for ch in CHANNELS:
            if f[ch].notna().sum() == 0:
                continue
            pre, npre = summarise(f[ch], pre0, env0)
            post, npost = summarise(f[ch], env1, post1)
            dur = pd.concat([f[ch].loc[a:b] for a, b in iv]).dropna()
            during = round(float(dur.median()), 4) if len(dur) else np.nan
            pk = [summarise(p[ch], env0, env1)[0] for p in peers.values()]
            pk_pre = [summarise(p[ch], pre0, env0)[0] for p in peers.values()]
            rows.append(dict(case=case, channel=ch, n_intervals=len(iv), pre_median=pre, during_median=during, post_median=post, delta_during_pre=round(during - pre, 4) if pd.notna(during) and pd.notna(pre) else np.nan, peer_pre_median=round(float(np.nanmedian(pk_pre)), 4), peer_during_median=round(float(np.nanmedian(pk)), 4), peer_delta=round(float(np.nanmedian(pk) - np.nanmedian(pk_pre)), 4), n_pre=npre, n_during=len(dur), n_post=npost))
            for j, (a, b) in enumerate(iv, 1):
                m, n = summarise(f[ch], a, b)
                per_interval.append(dict(case=case, interval=j, start=a, end=b, hours=round((b - a).total_seconds() / 3600, 2), channel=ch, median=m, n=n))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'kelmarsh_k2a_case_summary.csv', index=False)
    pd.DataFrame(per_interval).to_csv(OUT / 'kelmarsh_k2a_per_interval.csv', index=False)
    for case in CASES:
        print(f'\n=== {case} ===')
        print(d[d.case == case].drop(columns=['case', 'n_intervals']).to_string(index=False))
