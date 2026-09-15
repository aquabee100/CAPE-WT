from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'runtime'
RESULTS.mkdir(parents=True, exist_ok=True)

def save(obj, name):
    if isinstance(obj, dict):
        for k, v in obj.items():
            pd.DataFrame(v).to_parquet(RESULTS / f'{name}__{k}.parquet')
        (RESULTS / f'{name}__keys.txt').write_text('\n'.join(map(str, obj)))
    else:
        pd.DataFrame(obj).to_parquet(RESULTS / f'{name}.parquet')

def load(name):
    keys = RESULTS / f'{name}__keys.txt'
    if keys.exists():
        return {k: pd.read_parquet(RESULTS / f'{name}__{k}.parquet') for k in keys.read_text().split()}
    return pd.read_parquet(RESULTS / f'{name}.parquet')

def exists(name):
    return (RESULTS / f'{name}.parquet').exists() or (RESULTS / f'{name}__keys.txt').exists()

def save_trajectory(df, name):
    p = RESULTS / f'{name}.csv'
    x = pd.DataFrame(df).copy()
    x.index = pd.to_datetime(x.index)
    x.index.name = 't'
    x.to_csv(p)
    return p

def load_trajectory(name):
    p = RESULTS / f'{name}.csv'
    x = pd.read_csv(p, index_col='t', parse_dates=['t'])
    x.index = pd.to_datetime(x.index)
    return x
