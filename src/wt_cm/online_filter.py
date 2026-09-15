import pandas as pd
import filter_ch as F
import online_core as OC
import paths as P
import thermal_id as ti
if __name__ == '__main__':
    df = ti.load_raw()
    out = {}
    for turbine in ti.TURBINES:
        tf = F.frame(df, turbine)
        s = OC.filter_trajectory(tf)
        if len(s):
            out[turbine] = s
        print('online filter done', turbine, flush=True)
    del df
    M = pd.DataFrame(out)
    P.save_trajectory(M, 'filter_traj_online')
    print(f'written: results/filter_traj_online.csv  shape={M.shape}')
