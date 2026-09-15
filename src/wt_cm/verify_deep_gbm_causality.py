from __future__ import annotations
import argparse
import json
import pandas as pd
import paths as P
import thermal_id as TI
import verify_online_causality as V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', required=True)
    ap.add_argument('--mode', required=True, choices=V.MODES)
    ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    df = TI.load_raw()
    tf = V.combined_frame(df, V.CASE)
    del df
    bad = V.corrupt_future(tf, cutoff, args.mode)
    cur = V.deep_gbm_residual(bad, cutoff, n_jobs=args.jobs)
    frozen = P.load_trajectory('resid_gbm_adaptive')[V.CASE].loc[:cutoff].dropna()
    d = V.maxdiff(frozen, cur)
    if not (pd.notna(d) and d <= V.ATOL):
        raise AssertionError(f'complete adaptive GBM through {cutoff.date()} / {args.mode}: max |delta|={d}')
    row = dict(cutoff=str(cutoff.date()), mode=args.mode, n_decisions=int(len(frozen)), max_gbm_delta=float(d), passed=True)
    print(f'[PASS] complete adaptive GBM through {cutoff.date()} / {args.mode}: {len(frozen)} decisions, max |delta|={d:.3g}')
    print('RESULT_JSON ' + json.dumps(row, sort_keys=True))
if __name__ == '__main__':
    main()
