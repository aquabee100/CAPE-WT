from __future__ import annotations
import runpy
import subprocess
import sys
from pathlib import Path
import pandas as pd
import paths as P
STAGES = [('online_filter.py', ['filter_traj_online.csv']), ('online_2x2.py', ['thermal_traj_online__VA.csv', 'resid_gbm_adaptive.csv']), ('headline.py', ['headline_online.csv']), ('parallel_arch.py', ['parallel_architecture.csv']), ('availability_strat.py', ['availability_stratified.csv', 'availability_per_turbine.csv']), ('online_episode_ledger.py', ['online_episode_ledger.csv', 'online_case_episode_summary.csv'])]

def valid_output(name: str) -> bool:
    p = P.RESULTS / name
    if not p.exists():
        return False
    try:
        if p.suffix == '.csv':
            pd.read_csv(p)
        return True
    except Exception:
        return False

def run_script(script: str):
    print(f"\n{'=' * 78}\n[ONLINE] {script}\n{'=' * 78}", flush=True)
    if script == 'online_2x2.py':
        names = [n[:-4] for n in ['thermal_traj_online__VA.csv', 'resid_gbm_adaptive.csv'] if not valid_output(n)]
        if names:
            subprocess.run([sys.executable, script, *names], check=True)
    else:
        runpy.run_path(script, run_name='__main__')
if __name__ == '__main__':
    fresh = '--fresh' in sys.argv
    raw = Path('data/MANF2017-2022.pkl')
    needs_raw = any(not valid_output(o) for _, outs in STAGES for o in outs)
    if needs_raw and not raw.exists():
        raise SystemExit('Primary SCADA is not bundled. See DATA_ACCESS.md.')
    if fresh:
        for _, outs in STAGES:
            for out in outs:
                p = P.RESULTS / out
                if p.exists():
                    p.unlink()
        print('[ONLINE] cleared runtime online outputs only')
    for script, outputs in STAGES:
        if all((valid_output(o) for o in outputs)):
            print(f'[skip] {script}: outputs already valid')
            continue
        run_script(script)
    print('\n=== CAPE-WT ONLINE HEADLINE ===')
    print(pd.read_csv(P.RESULTS / 'parallel_architecture.csv').to_string(index=False))
    print('\nAuthoritative inputs:')
    for f in ['filter_traj_online.csv', 'resid_gbm_adaptive.csv', 'thermal_traj_online__VA.csv']:
        print('  ', P.RESULTS / f)
    print('\nVerification is intentionally separate:')
    print('  python verify_freeze_integrity.py   # stored-artifact consistency')
    print('  python verify_online_causality.py   # raw-data future-invariance')
