import pandas as pd, warnings
warnings.filterwarnings('ignore')
import paths as P, detect_final as D, protocol as PR
import baseline_rule as B, headline as H, state_machine as SM, parallel_arch as PA
START, SCALE, K, HTH, HOR = (pd.Timestamp('2018-07-01'), PR.SCALE_END['MF07'], 3.0, 0.5, 180)
z = lambda M: B.rolling_z_v2(M, scale_end=SCALE, min_history=1, min_history_days=H.FULL_GRID_SPAN.days)
Zf = z(P.load_trajectory('filter_traj_online'))
Zg = z(P.load_trajectory('resid_gbm_adaptive'))
Zt = z(P.load_trajectory('thermal_traj_online__VA'))
fo = go = un = 0
expo = 0.0
prov = {'filter': 0, 'GBM': 0, 'both': 0}
for t in D.HEALTHY:
    zf, zg, zt = (Zf[t].loc[START:], Zg[t].reindex(Zf.index).loc[START:], Zt[t].loc[START:])
    av = zg.notna()
    fo += len(SM.warnings_of(SM.run_machine(zf.where(av), zt, K, HTH, HTH, HOR, latch=False)))
    go += len(SM.warnings_of(SM.run_machine(zg.where(av), zt, K, HTH, HTH, HOR, latch=False)))
    eps = PA.run_parallel(zf.where(av), zg.where(av), zt, latch=False)
    un += len(eps)
    for e in eps:
        prov[e['source']] = prov.get(e['source'], 0) + 1
    expo += H.evaluable_exposure(zf.where(av), H.RECORD_END)
rows = [dict(configuration='filter only', episodes=fo, exposure_yr=round(expo, 2), far_warn=round(fo / expo, 3), note='proposed cold-start configuration'), dict(configuration='adaptive GBM only', episodes=go, exposure_yr=round(expo, 2), far_warn=round(go / expo, 3), note='COUNTERFACTUAL ONLY - not a proposed configuration'), dict(configuration='parallel union', episodes=un, exposure_yr=round(expo, 2), far_warn=round(un / expo, 3), note='proposed mature-history configuration')]
d = pd.DataFrame(rows)
d.to_csv(P.RESULTS / 'channel_counterfactual.csv', index=False)
print(d.to_string(index=False))
print(f'\nsum of singles {fo + go}, union {un}, union/sum {un / (fo + go):.2f}  -> SUB-additive')
print(f'union / filter-only = {un / fo:.3f}  (+{100 * (un / fo - 1):.0f}%)')
print('first-trigger provenance:', prov)
