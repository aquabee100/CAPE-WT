import itertools, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, thermal_id as ti
import signal_preservation as S, sigpres_raw as R
ONSET = '2020-12-01'
if __name__ == '__main__':
    import sys
    cells = sys.argv[1:] or list(R.CELLS)
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    for cell in cells:
        base = pd.read_parquet(P.RESULTS / f'resid_{cell}.parquet')
        ck = P.RESULTS / f'winter__{cell}.partial.csv'
        rows, done = ([], set())
        if ck.exists():
            prev = pd.read_csv(ck)
            rows = prev.to_dict('records')
            done = {(r['turbine'], r['amplitude'], r['ramp_days']) for r in rows}
            print(f'resuming {cell}: {len(done)}/42', flush=True)
        for turb in D.HEALTHY:
            org = base[turb]
            z_org = S.zseries(base, turb, org)
            for A, Dd in itertools.product(S.AMPLITUDES, S.RAMPS):
                if (turb, A, Dd) in done:
                    continue
                inj = R.CELLS[cell](S.perturb(frames[turb], ONSET, A, Dd), turb)
                t0 = pd.Timestamp(ONSET)
                dr = (inj - org).dropna()
                zm = S.metrics(S.zseries(base, turb, inj), z_org, ONSET) or {}
                rows.append(dict(cell=cell, turbine=turb, onset=ONSET, amplitude=A, ramp_days=Dd, raw_auc_0_45=round(R.aucp(dr, t0, 0, 45), 4), raw_auc_45_180=round(R.aucp(dr, t0, 45, 180), 4), raw_auc_0_180=round(R.aucp(dr, t0, 0, 180), 4), **zm))
                pd.DataFrame(rows).to_csv(ck, index=False)
                print(f'  {cell} {turb} A={A} D={Dd} [{len(rows)}/42]', flush=True)
        pd.DataFrame(rows).to_csv(P.RESULTS / f'winter__{cell}.csv', index=False)
        ck.unlink(missing_ok=True)
        print(f'written: results/winter__{cell}.csv', flush=True)
