import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
ANCHOR = pd.Timestamp('2019-02-15')
END = pd.Timestamp('2019-06-30')
INTERVALS = [('2019-02-24 16:46:28', '2019-04-04 12:35:45'), ('2019-04-09 10:09:03', '2019-05-21 10:06:24'), ('2019-05-28 20:55:45', '2019-05-30 07:34:04')]
PEERS = ['KWF2', 'KWF3', 'KWF4', 'KWF5', 'KWF6']
c = pd.read_csv(OUT / 'kelmarsh_k1_coefficients.csv', parse_dates=['t'])
c = c[c.period.astype(str) == '2019']
I = c.pivot_table(index='t', columns='turbine', values='index')
dI = I.sub(I.loc[ANCHOR]).loc[ANCHOR:END]
print('=== peer-relative index change dI, anchored 2019-02-15 ===')
out = dI.copy()
out['peer_median'] = dI[PEERS].median(axis=1)
out['peer_q25'] = dI[PEERS].quantile(0.25, axis=1)
out['peer_q75'] = dI[PEERS].quantile(0.75, axis=1)
out['KWF1_minus_peer'] = dI['KWF1'] - out['peer_median']
out['KWF1_rank_of_6'] = dI.rank(axis=1).loc[:, 'KWF1']
print(out.round(3).to_string())
out.to_csv(OUT / 'kelmarsh_k2b_trajectory.csv')
rows = []
for j, (a, b) in enumerate(INTERVALS, 1):
    a, b = (pd.Timestamp(a), pd.Timestamp(b))
    seg = dI.loc[a:b]
    if not len(seg):
        rows.append(dict(interval=j, start=a.date(), end=b.date(), n_windows=0))
        continue
    rows.append(dict(interval=j, start=a.date(), end=b.date(), n_windows=len(seg), KWF1_median_dI=round(float(seg['KWF1'].median()), 3), peer_median_dI=round(float(seg[PEERS].median(axis=1).median()), 3), KWF1_minus_peer=round(float((seg['KWF1'] - seg[PEERS].median(axis=1)).median()), 3), KWF1_rank_of_6=round(float(seg.rank(axis=1)['KWF1'].median()), 1), inside_peer_range=bool(((seg['KWF1'] >= seg[PEERS].min(axis=1)) & (seg['KWF1'] <= seg[PEERS].max(axis=1))).all())))
t = pd.DataFrame(rows)
t.to_csv(OUT / 'kelmarsh_k2b_intervals.csv', index=False)
print('\n=== per active interval ===')
print(t.to_string(index=False))
