import ast
import json
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FAIL = []

def ok(msg):
    print(f'  PASS  {msg}')

def skip(msg):
    print(f'  SKIP  {msg}')

def bad(msg):
    FAIL.append(msg)
    print(f'  FAIL  {msg}')

def check_deps():
    print('\n[1] dependencies')
    core = [('pandas', '2.0'), ('numpy', '1.24'), ('sklearn', '1.3'), ('scipy', '1.10')]
    for mod, minver in core:
        try:
            m = __import__(mod)
            v = getattr(m, '__version__', '?')
            ok(f'{mod} {v}')
        except ImportError:
            bad(f'{mod} missing (core online path needs >= {minver})')
    try:
        import pyarrow
        ok(f'pyarrow {pyarrow.__version__} (optional legacy/parquet branches)')
    except ImportError:
        skip('pyarrow not installed — optional for legacy/parquet branches; headline online CSV path is unaffected')

def check_data():
    print('\n[2] data')
    pkl = HERE / 'data' / 'MANF2017-2022.pkl'
    if pkl.exists():
        ok(f'primary archive {pkl.stat().st_size / 1000000.0:.0f} MB')
    else:
        skip(f'primary raw archive not bundled — place MANF2017-2022.pkl at {pkl} for raw regeneration/causality tests')
    kd = Path(os.environ.get('KELMARSH_RAW', HERE.parent / 'kelmarsh' / 'Kelmarsh'))
    n = len(list(kd.glob('*.csv'))) if kd.exists() else 0
    if n:
        ok(f'external fleet {n} CSVs at {kd}')
    else:
        skip(f'external raw fleet not bundled — set KELMARSH_RAW to rerun external analyses (looked in {kd})')

def check_config():
    print('\n[3] config matches code')
    try:
        import causal as C, normalise as N, headline as H, online_2x2 as O
    except Exception as e:
        bad(f'cannot import core modules: {e}')
        return
    cfg = json.loads((HERE / 'config.json').read_text())
    live = {'score_window_days': C.WINDOW.days, 'decision_stride_days': C.STEP.days, 'baseline_days': N.ROLL_WINDOW.days, 'baseline_span_days': H.FULL_GRID_SPAN.days, 'embargo_days': N.ROLL_EMBARGO.days, 'min_baseline_windows': H.MIN_VALID_BASE_WINDOWS, 'adaptive_lookback_days': O.LOOKBACK.days, 'thermal_target': O.TC, 'gbm_features': O.FEATS}
    for k, v in live.items():
        ok(f'{k} = {v}') if cfg.get(k) == v else bad(f'{k}: config says {cfg.get(k)}, code says {v}')

def check_hygiene():
    print('\n[4] code hygiene')
    quarantined = {p.stem for p in (HERE / 'superseded').glob('*.py')} if (HERE / 'superseded').exists() else set()
    local = {p.stem for p in HERE.glob('*.py')}
    hard, leaks = ([], [])
    for f in HERE.glob('*.py'):
        src = f.read_text()
        if f.name == 'verify.py':
            continue
        if '/home/' in src or '/mnt/user-data' in src:
            hard.append(f.name)
        for n in ast.walk(ast.parse(src)):
            mods = [a.name.split('.')[0] for a in n.names] if isinstance(n, ast.Import) else [n.module.split('.')[0]] if isinstance(n, ast.ImportFrom) and n.module else []
            for m in mods:
                if m in quarantined:
                    leaks.append(f'{f.name} imports quarantined {m}')
    ok('no hardcoded absolute paths') if not hard else bad(f'hardcoded paths in {hard}')
    ok('no live import of a superseded module') if not leaks else [bad(x) for x in leaks]
    ok(f'{len(local)} live modules, {len(quarantined)} quarantined')

def check_headline():
    print('\n[5] headline numbers')
    import pandas as pd
    import paths as P
    R = P.RESULTS
    if not (R / 'parallel_architecture.csv').exists():
        skip('runtime headline outputs unavailable')
        return
    try:
        p = pd.read_csv(R / 'parallel_architecture.csv')
        par = p[p.architecture.str.startswith('parallel')].set_index('case')
        checks = [('MF12 warning', par.loc['MF12', 'warn_lead'], 199), ('MF12 confirmation', par.loc['MF12', 'conf_lead'], 139), ('MF07 warning', par.loc['MF07', 'warn_lead'], 294), ('MF07 confirmation', par.loc['MF07', 'conf_lead'], 219), ('MF12 confirmed control alarms', par.loc['MF12', 'confirmed_alarms'], 0), ('MF07 confirmed control alarms', par.loc['MF07', 'confirmed_alarms'], 0)]
        for name, got, want in checks:
            ok(f'{name} = {got:.0f}') if float(got) == want else bad(f'{name}: expected {want}, got {got}')
        c = pd.read_csv(R / 'channel_counterfactual.csv').set_index('configuration')
        ok('matched exposure 21.54 ty') if abs(c.loc['filter only', 'exposure_yr'] - 21.54) < 0.01 else bad(f"matched exposure {c.loc['filter only', 'exposure_yr']}")
        k = pd.read_csv(R / 'kelmarsh_k1_coefficients.csv') if (R / 'kelmarsh_k1_coefficients.csv').exists() else None
    except Exception as e:
        bad(f'headline regeneration failed: {e}')

if __name__ == '__main__':
    print('=' * 64)
    print('wt_cm verification')
    print('=' * 64)
    check_deps()
    check_data()
    check_config()
    check_hygiene()
    check_headline()
    print('\n' + '=' * 64)
    if FAIL:
        print(f'{len(FAIL)} CHECK(S) FAILED:')
        for f in FAIL:
            print('  -', f)
        sys.exit(1)
    print('all checks passed')
