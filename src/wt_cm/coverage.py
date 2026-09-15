import numpy as np
import pandas as pd
import detect_final as D
import normalise as N
import paths as P
import protocol as PR
if __name__ == '__main__':
    Ft = P.load('filter_traj')
    Tt = P.load('causal_traj')['VA']
    rows = []
    for case in ['MF07', 'MF12']:
        c = PR.CASES[case]
        Zf = N.rolling_z(Ft, scale_end=PR.SCALE_END[case])
        Zt = N.rolling_z(Tt, scale_end=PR.SCALE_END[case])
        sl = slice(c['monitor_from'], None)
        for t in D.HEALTHY + [case]:
            zf, zt = (Zf[t].loc[sl], Zt[t].loc[sl])
            n = len(zf)
            rows.append(dict(case=case, turbine=t, n_windows=n, filter_cov=round(np.isfinite(zf.values).mean(), 3), thermal_cov=round(np.isfinite(zt.values).mean(), 3), both_cov=round((np.isfinite(zf.values) & np.isfinite(zt.values)).mean(), 3)))
    df = pd.DataFrame(rows)
    df.to_csv(P.RESULTS / 'detector_coverage.csv', index=False)
    print(df.to_string(index=False))
    for case in ['MF07', 'MF12']:
        s = df[(df.case == case) & df.turbine.isin(D.HEALTHY)]
        print(f'\n{case} healthy controls: both-channel coverage {s.both_cov.min():.1%}-{s.both_cov.max():.1%} (median {s.both_cov.median():.1%})')
