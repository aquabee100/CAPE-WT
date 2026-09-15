import re
from pathlib import Path
import numpy as np
import pandas as pd
import os
from pathlib import Path
_HERE = Path(__file__).resolve().parent
RAW = Path(os.environ.get('KELMARSH_RAW', _HERE / 'Kelmarsh'))
OUT = Path(os.environ.get('KELMARSH_OUT', _HERE))
OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)
HEADER_ROW = 9
STATUS_HEADER_ROW = 9
ROLE_PATTERNS = [('target_temperature', 'gear.*(bearing|oil).*temp|gearbox.*temp'), ('drivetrain_temperature', '(generator|main bearing|shaft|hydraulic|nacelle|rotor bearing).*temp'), ('ambient_environment', 'ambient.*temp|outdoor|air (temp|density)|humidity|pressure \\(hpa\\)'), ('load', '^power|active power|reactive power|apparent power|energy'), ('speed', '(rotor|generator|gear).*(rpm|speed)'), ('oil_pressure', 'oil.*press|press.*oil|pump.*press'), ('oil_condition', 'particle|oil level|oil quality'), ('wind', 'wind speed|wind direction|nacelle position|vane'), ('operational', 'pitch|state|status|available|curtail|yaw|blade')]
EVENT_FAMILY = [('oil_filter', 'oil filter|filter.*(gear|oil)|choked'), ('gear_temperature', '(high|low).*temp.*gear|gear.*temp'), ('gear_speed', 'gear.*speed|implausible.*gear'), ('oil_pump', 'oil pump|pump.*oil|lubric'), ('other_gearbox', 'gear')]

def classify(name, table):
    low = str(name).lower()
    for role, pat in table:
        if re.search(pat, low):
            return role
    return None

def read_turbine(path):
    return pd.read_csv(path, skiprows=HEADER_ROW, low_memory=False)

def period_of(path):
    m = re.search('(\\d{4}-\\d{2}-\\d{2})_-_(\\d{4}-\\d{2}-\\d{2})', path.name)
    return f'{m.group(1)}..{m.group(2)}' if m else '?'

def schema_audit():
    rows = []
    for f in sorted(RAW.glob('Turbine_Data_Kelmarsh_*.csv')):
        turb = re.search('Kelmarsh_(\\d+)_', f.name).group(1)
        per = period_of(f)
        df = read_turbine(f)
        ts = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        for c in df.columns[1:]:
            s = pd.to_numeric(df[c], errors='coerce')
            rows.append(dict(turbine=f'KWF{turb}', period=per, raw_variable=c, unit=(re.search('\\(([^)]*)\\)\\s*$', str(c)) or [None, ''])[1], n=len(s), missing_fraction=round(float(s.isna().mean()), 4), vmin=round(float(s.min()), 3) if s.notna().any() else np.nan, vmax=round(float(s.max()), 3) if s.notna().any() else np.nan, candidate_role=classify(c, ROLE_PATTERNS), first=ts.min(), last=ts.max()))
        del df
    return pd.DataFrame(rows)

def event_ledger():
    rows = []
    for f in sorted(RAW.glob('Status_Kelmarsh_*.csv')):
        turb = re.search('Kelmarsh_(\\d+)_', f.name).group(1)
        df = pd.read_csv(f, skiprows=HEADER_ROW, low_memory=False)
        df['turbine'] = f'KWF{turb}'
        df['period'] = period_of(f)
        rows.append(df)
    ev = pd.concat(rows, ignore_index=True)
    txt = ev.get('Message', pd.Series('', index=ev.index)).fillna('').astype(str) + ' ' + ev.get('Comment', pd.Series('', index=ev.index)).fillna('').astype(str)
    ev['event_family'] = [classify(t, EVENT_FAMILY) for t in txt]
    return ev
if __name__ == '__main__':
    sc = schema_audit()
    sc.to_csv(OUT / 'kelmarsh_schema.csv', index=False)
    print(f"schema: {len(sc)} turbine-variable-period rows, {sc['raw_variable'].nunique()} distinct variables\n")
    print('candidate roles (distinct variables):')
    print(sc.drop_duplicates('raw_variable')['candidate_role'].value_counts(dropna=False).to_string())
    print('\ngearbox/oil-relevant variables, coverage by period:')
    keep = sc[sc.candidate_role.isin(['target_temperature', 'oil_pressure', 'oil_condition'])]
    piv = keep.groupby(['raw_variable', 'period'])['missing_fraction'].mean().unstack().round(3)
    print(piv.to_string())
    ev = event_ledger()
    ev.to_csv(OUT / 'kelmarsh_events_raw.csv', index=False)
    g = ev[ev.event_family.notna()]
    g.to_csv(OUT / 'kelmarsh_gear_events.csv', index=False)
    print(f'\nstatus records: {len(ev)} total, {len(g)} gearbox/oil-related')
    print('\nevent_family counts:')
    print(g['event_family'].value_counts().to_string())
    print('\nstatus columns:', list(ev.columns))
