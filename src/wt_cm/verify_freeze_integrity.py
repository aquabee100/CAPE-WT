from pathlib import Path
import pandas as pd
import paths as P
ROOT = Path(__file__).resolve().parents[1]
KOUT = ROOT / 'kelmarsh' / 'rerun'

def req(cond, msg):
    if not bool(cond):
        raise AssertionError(msg)
    print('[PASS]', msg)

def main():
    F = P.load_trajectory('filter_traj_online')
    G = P.load_trajectory('resid_gbm_adaptive')
    T = P.load_trajectory('thermal_traj_online__VA')
    for name, M in [('filter', F), ('GBM', G), ('thermal', T)]:
        req(M.index.is_monotonic_increasing and M.index.is_unique, f'{name}: chronological unique decision index')
        req((M.index.to_series().diff().dropna() >= pd.Timedelta(0)).all(), f'{name}: no backward time step')
    req(F.shape == (133, 9), 'filter trajectory is 133 x 9')
    req(T.shape == (133, 9), 'thermal trajectory is 133 x 9')
    req(G.shape == (111, 9), 'adaptive GBM trajectory is 111 x 9 after warm-up')
    S = pd.read_csv(P.RESULTS / 'online_case_episode_summary.csv').set_index('case')
    req(int(S.loc['MF07', 'first_warning_lead']) == 294, 'MF07 online warning lead = 294 d')
    req(int(S.loc['MF07', 'confirmation_lead']) == 219, 'MF07 physical confirmation lead = 219 d')
    req(S.loc['MF07', 'first_warning_source'] == 'filter', 'MF07 warning source = filter (cold start)')
    req(int(S.loc['MF12', 'first_warning_lead']) == 199, 'MF12 online warning lead = 199 d')
    req(int(S.loc['MF12', 'confirmation_lead']) == 139, 'MF12 physical confirmation lead = 139 d')
    req(S.loc['MF12', 'first_warning_source'] == 'GBM', 'MF12 first warning source = adaptive GBM')
    req(int(S.loc['MF12', 'warning_to_confirmation_delay']) == 60, 'MF12 warning->confirmation delay = 60 d')
    L = pd.read_csv(P.RESULTS / 'online_episode_ledger.csv', parse_dates=['warning', 'confirmation'])
    c = L[L.status.eq('confirmed')]
    req((c.confirmation >= c.warning).all(), 'every confirmation follows its own warning')
    req(((c.confirmation - c.warning).dt.days <= 180).all(), 'every confirmation lies within the 180 d episode horizon')
    PARCH = pd.read_csv(P.RESULTS / 'parallel_architecture.csv')
    q = PARCH[(PARCH.case == 'MF07') & (PARCH.architecture == 'parallel filter OR GBM')].iloc[0]
    req(int(q.gbm_windows_available_pre_event) == 0, 'MF07 has zero adaptive-GBM decision windows before event')
    q = PARCH[(PARCH.case == 'MF12') & (PARCH.architecture == 'parallel filter OR GBM')].iloc[0]
    req(int(q.gbm_windows_available_pre_event) == 53, 'MF12 has 53 adaptive-GBM decision windows before event')
    A = pd.read_csv(P.RESULTS / 'availability_stratified.csv')
    req(int(A.iloc[0].warn_episodes) == 0, 'cold-start reference exposure has zero warning episodes')
    req(int(A.iloc[1].warn_episodes) == 16, 'mature reference exposure has 16 warning episodes')
    PT = pd.read_csv(P.RESULTS / 'availability_per_turbine.csv')
    req(int(PT.confirmed.sum()) == 0, 'zero confirmed control alarms on the unique reference-fleet timeline')
    expo = float(PT.expo_A.sum() + PT.expo_B.sum())
    req(abs(expo - 27.86571222197) < 0.01, 'unique reference-fleet exposure = 27.87 turbine-years')
    K1 = pd.read_csv(KOUT / 'kelmarsh_k1_coefficients.csv')
    req(len(K1) == 192, 'Kelmarsh transferred physical model: 192 fitted windows')
    req((K1.a_nac > 0).all(), 'Kelmarsh cooling coefficient positive in 192/192 windows')
    K3A = pd.read_csv(KOUT / 'kelmarsh_k3a_lookahead.csv')
    req(len(K3A) == 168, 'Kelmarsh centered/trailing availability comparison: 168 pairs')
    req((K3A.n_centered == K3A.n_trailing).all(), 'centered and delayed trailing windows use identical observations in 168/168 pairs')
    req(float(K3A.abs_difference.max()) == 0.0, 'centered vs delayed-trailing physical index mismatch = 0')
    K3B = pd.read_csv(KOUT / 'kelmarsh_k3b_derivative.csv')
    req(len(K3B) == 192, 'Kelmarsh causal/legacy comparison: 192 matched windows')
    req((K3B.rmse_legacy < K3B.rmse_causal).all(), 'future-aware legacy RMSE is lower in 192/192 windows')
    DG = pd.read_csv(P.RESULTS / 'online_causality_deep_gbm.csv')
    req(len(DG) == 10 and DG.passed.astype(bool).all(), 'stored deep adaptive-GBM causality record contains 10/10 passing raw replays')
    req(set(DG['mode']) == {'constant', 'ramp', 'freeze', 'noise', 'reverse'}, 'stored deep adaptive-GBM record contains all five hostile future modes')
    req(set(DG['cutoff'].astype(str)) == {'2021-08-08', '2021-10-07'}, 'stored deep adaptive-GBM record contains warning and confirmation cutoffs')
    req(float(DG.max_gbm_delta.max()) <= 1e-11, 'stored deep adaptive-GBM maximum pre-cutoff delta is within tolerance')
    PUB = pd.read_csv(P.RESULTS / 'published_twostage.csv')

    def prow(channel, case):
        return PUB[(PUB.channel == channel) & (PUB.case == case)].iloc[0]
    req(prow('adaptive GBM (proposed)', 'MF07').evaluable == 'no', 'adaptive GBM remains unavailable for MF07 in the common plug-in test')
    req(int(prow('P2 Schlechtingen linear FSRC', 'MF12').conf_lead) == 124, 'P2 MF12 confirmation occurs one 15-day state step after its 139-day warning')
    eval_pub = PUB[PUB.evaluable == 'yes']
    req(int(eval_pub.confirmed.fillna(0).sum()) == 0, 'no confirmed control alarms across all evaluable literature plug-in configurations')
    PF = pd.read_csv(P.RESULTS / 'published_frontier.csv')
    q = PF[(PF.method == 'P3 GBM-NBM + EWMA') & (PF.case == 'MF12') & (PF.target_burden == 0.221)].iloc[0]
    req(int(q.lead_days) == 447, 'P3 matched-warning-burden MF12 lead = 447 d')
    q = PF[(PF.method == 'P5 multivariate CUSUM') & (PF.case == 'MF12') & (PF.target_burden == 0.221)].iloc[0]
    req(int(q.lead_days) == 351, 'P5 matched-warning-burden MF12 lead = 351 d')
    q = PF[(PF.method == 'P2 Schlechtingen linear FSRC') & (PF.case == 'MF12') & (PF.target_burden == 0.221)].iloc[0]
    req(int(q.lead_days) == 766 and bool(q.near_record_boundary), 'P2 matched-burden 766 d MF12 crossing is flagged near the monitoring boundary')
    print('\nALL CAPE-WT FREEZE-INTEGRITY ASSERTIONS PASSED')
    print('NOTE: this is an artifact-consistency check, not a raw-data causality test.')
if __name__ == '__main__':
    main()
