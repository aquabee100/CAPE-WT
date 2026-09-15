import itertools
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import paths as P
import causal as C
import detect_final as D
import protocol as PR
import baseline_rule as B
import headline as H
import state_machine as SM
import thermal_id as ti
import online_2x2 as O
AMPLITUDES = [0.5, 1.0, 2.0]
RAMPS = [60, 120]
ONSETS = ['2020-06-01']
HORIZON = 180
K, HTH = (3.0, 0.5)
CELLS = {'phys_frozen': lambda tf: O.phys_residual(tf, False), 'phys_adaptive': lambda tf: O.phys_residual(tf, True), 'gbm_frozen': lambda tf: O.gbm_residual(tf, False), 'gbm_adaptive': lambda tf: O.gbm_residual(tf, True)}

def perturb(tf, t0, A, D_days):
    out = tf.copy()
    t0 = pd.Timestamp(t0)
    dt = np.asarray((out.index - t0).total_seconds()) / 86400.0
    d = np.clip(dt / D_days, 0.0, 1.0) * A
    d = np.where(dt < 0, 0.0, d)
    out['T_gbb'] = out['T_gbb'] + d
    return out

def zseries(M_base, turbine, series, case='MF12'):
    M = M_base.copy()
    M[turbine] = series
    Z = B.rolling_z_v2(M, scale_end=PR.SCALE_END[case], min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
    return Z[turbine]

def metrics(z_inj, z_org, t0):
    t0 = pd.Timestamp(t0)
    dz = (z_inj - z_org).dropna()
    win = dz.loc[t0:t0 + pd.Timedelta(days=HORIZON)]
    if not len(win):
        return None

    def aucp(a, b):
        seg = dz.loc[t0 + pd.Timedelta(days=a):t0 + pd.Timedelta(days=b)]
        if len(seg) < 2:
            return 0.0
        dtd = np.diff(seg.index.values).astype('timedelta64[D]').astype(float)
        return float((np.clip(seg.values[:-1], 0, None) * dtd).sum())
    eps = SM.run_machine(z_inj.loc[t0:t0 + pd.Timedelta(days=HORIZON)], z_inj * 0 - 99, K, HTH, HTH, HORIZON, latch=False)
    w = SM.warnings_of(eps)
    return dict(peak_dz=round(float(win.max()), 3), auc_0_45=round(aucp(0, 45), 2), auc_45_180=round(aucp(45, 180), 2), auc_0_180=round(aucp(0, 180), 2), warned=int(bool(w)), delay_days=int((pd.Timestamp(w[0]) - t0).days) if w else None)
if __name__ == '__main__':
    import sys
    cells = sys.argv[1:] or list(CELLS)
    df = ti.load_raw()
    frames = {t: ti.turbine_frame(df, t) for t in ti.TURBINES}
    del df
    for cell in cells:
        base_path = P.RESULTS / f"resid_{cell.replace('phys', 'phys').replace('_', '_')}.parquet"
        M_base = pd.read_parquet(P.RESULTS / f'resid_{cell}.parquet')
        rows = []
        for turb in D.HEALTHY:
            z_org = zseries(M_base, turb, M_base[turb])
            for t0, A, Dd in itertools.product(ONSETS, AMPLITUDES, RAMPS):
                s = CELLS[cell](perturb(frames[turb], t0, A, Dd))
                m = metrics(zseries(M_base, turb, s), z_org, t0)
                if m:
                    rows.append(dict(cell=cell, turbine=turb, onset=t0, amplitude=A, ramp_days=Dd, **m))
                print(f'  {cell} {turb} A={A} D={Dd} done', flush=True)
        pd.DataFrame(rows).to_csv(P.RESULTS / f'sigpres__{cell}.csv', index=False)
        print(f'written: results/sigpres__{cell}.csv ({len(rows)} scenarios)', flush=True)
