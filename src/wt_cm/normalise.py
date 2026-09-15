import numpy as np
import pandas as pd
import detect_final as D
import protocol as PR

def pooled_scale(dev, mu):
    mad = (dev - mu).abs().median() * 1.4826
    s = mad[[t for t in D.HEALTHY if t in mad.index]].median()
    return float(s) if np.isfinite(s) and s > 0 else np.nan

def to_z_pooled(M, start=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    dev = M.loc[start:PR.DEV_END] if start else M.loc[:PR.DEV_END]
    mu = dev.mean()
    s = pooled_scale(dev, mu)
    return (M - mu) / s

def to_z_own(M, start=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    dev = M.loc[start:PR.DEV_END] if start else M.loc[:PR.DEV_END]
    return (M - dev.mean()) / dev.std()

def dev_reliability(M, start=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    dev = M.loc[start:PR.DEV_END] if start else M.loc[:PR.DEV_END]
    a, b = (dev.iloc[::2], dev.iloc[1::2])
    sa = (a - a.mean()).abs().median() * 1.4826
    sb = (b - b.mean()).abs().median() * 1.4826
    h = [t for t in D.HEALTHY if t in sa.index]
    x, y = (np.log(sa[h].values), np.log(sb[h].values))
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return 0.0
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    return float(max(0.0, 2 * r / (1 + r))) if r > 0 else 0.0

def to_z_shrunk(M, start=None):
    w = dev_reliability(M, start)
    M = M.replace([np.inf, -np.inf], np.nan)
    dev = M.loc[start:PR.DEV_END] if start else M.loc[:PR.DEV_END]
    mu = dev.mean()
    own = (dev - mu).abs().median() * 1.4826
    pool = pooled_scale(dev, mu)
    scale = own.pow(w) * pool ** (1 - w)
    return (M - mu) / scale

def canonical_z(M, start=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    dev = M.loc[start:PR.DEV_END] if start else M.loc[:PR.DEV_END]
    loc = dev.median()
    s_i = (dev - loc).abs().median() * 1.4826
    h = [t for t in D.HEALTHY if t in s_i.index]
    s_pool = s_i[h].median()
    if not np.isfinite(s_pool) or s_pool <= 0:
        return M * np.nan
    return (M - loc) / float(s_pool)

def cusum_hold(z, k):
    s, out = (0.0, [])
    for v in z.values:
        if np.isfinite(v):
            s = max(0.0, s + v - k)
        out.append(s)
    return pd.Series(out, index=z.index)

def canonical_z_fleet(M, start=None):
    Z = canonical_z(M, start)
    loo = pd.DataFrame({t: Z.drop(columns=t).median(axis=1) for t in Z.columns}, index=Z.index)
    return Z - loo
ROLL_WINDOW = pd.Timedelta('365D')
ROLL_EMBARGO = pd.Timedelta('180D')
MIN_BASE_WINDOWS = 12

def rolling_z(M, start=None, scale_end=None):
    M = M.replace([np.inf, -np.inf], np.nan)
    end = scale_end or PR.DEV_END
    dev = M.loc[start:end] if start else M.loc[:end]
    s_pool = pooled_scale(dev, dev.median())
    if not np.isfinite(s_pool) or s_pool <= 0:
        return M * np.nan
    out = pd.DataFrame(index=M.index, columns=M.columns, dtype=float)
    for t in M.index:
        lo, hi = (t - ROLL_EMBARGO - ROLL_WINDOW, t - ROLL_EMBARGO)
        base = M.loc[lo:hi]
        if not len(base):
            continue
        counts = base.notna().sum()
        med = base.median()
        ok = counts >= MIN_BASE_WINDOWS
        out.loc[t, ok[ok].index] = (M.loc[t, ok[ok].index] - med[ok[ok].index]) / s_pool
    return out
