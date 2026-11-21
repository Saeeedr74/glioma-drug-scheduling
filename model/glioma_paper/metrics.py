"""The one metric function used for every number in every table.

Horizon: days 0..2000 (states), doses on days 0..1999. A dose logged on day 2000 acts outside
the horizon and is ignored.
"""
import math

import numpy as np

from .model import T_REF, U_MAX, U_MIN, X1_RECOVERY, X2_THRESHOLD

N_DAYS = int(T_REF)


def first_day(cond):
    idx = np.flatnonzero(cond)
    return float(idx[0]) if idx.size else math.nan


def perm_day(cond):
    """First day from which cond holds until the last day; NaN if it does not hold on the last day."""
    cond = np.asarray(cond, dtype=bool)
    if not cond[-1]:
        return math.nan
    bad = np.flatnonzero(~cond)
    return 0.0 if bad.size == 0 else float(bad[-1] + 1)


def runs(mask):
    """Start indices and lengths of the runs of True values."""
    m = np.concatenate(([0], np.asarray(mask, dtype=int), [0]))
    d = np.diff(m)
    st = np.flatnonzero(d == 1)
    en = np.flatnonzero(d == -1)
    return st, en - st


def metrics(X, u, ref=None, discrete=True):
    """X: states on days 0..2000, shape (2001, 4). u: doses, at least 2000 values (days 0..1999).
    ref: reference on days 0..2000 (optional, for the tracking error)."""
    X = np.asarray(X, dtype=float)
    assert X.shape[0] >= N_DAYS + 1, X.shape
    X = X[:N_DAYS + 1]
    u = np.asarray(u, dtype=float)[:N_DAYS]
    assert u.shape[0] == N_DAYS and np.all(np.isfinite(u)), "doses on days 0..1999 are needed"
    x1, x2, x3, x4 = X.T
    m = {}
    m["t_x2_first"] = first_day(x2 < X2_THRESHOLD)
    m["t_x2_perm"] = perm_day(x2 < X2_THRESHOLD)
    m["t_x1_first"] = first_day(x1 >= X1_RECOVERY)
    m["t_x1_perm"] = perm_day(x1 >= X1_RECOVERY)
    m["x1_T"], m["x2_T"], m["x3_T"], m["x4_T"] = (float(v) for v in X[-1])
    m["x1_min"] = float(x1.min())
    m["x3_min"] = float(x3.min())
    m["x4_max"] = float(x4.max())
    m["cum_dose"] = float(u.sum())                       # sum of the daily doses, days 0..1999
    m["mean_dose"] = float(u.mean())
    m["u_min"], m["u_max"] = float(u.min()), float(u.max())
    if ref is not None:
        ref = np.asarray(ref, dtype=float)[:N_DAYS + 1]
        m["rmse_over_x2pre"] = float(np.sqrt(np.mean((x2 - ref)**2))/ref[0])
        el = np.log10(np.maximum(x2, 1e-4)) - np.log10(np.maximum(ref, 1e-4))
        m["jtrack_log"] = float(np.trapezoid(el**2, np.arange(N_DAYS + 1))/T_REF)
    if discrete:
        hi = np.abs(u - U_MAX) <= 1e-9
        lo = np.abs(u - U_MIN) <= 1e-9
        m["only_50_200"] = bool(np.all(hi | lo))
        st, ln = runs(hi)
        m["n_hi"] = int(ln.size)
        m["hi_mean"] = float(ln.mean()) if ln.size else math.nan
        m["hi_min"] = float(ln.min()) if ln.size else math.nan
        m["hi_max"] = float(ln.max()) if ln.size else 0.0
        m["n_hi_gt5"] = int(np.sum(ln > 5))
        m["hi_days"] = int(hi.sum())
        m["frac_hi"] = float(hi.mean())
        m["last_hi_day"] = float(st[-1] + ln[-1] - 1) if ln.size else math.nan
        # maintenance phases: every run of u = 50 on days 0..1999 ...
        sl, ll = runs(lo)
        m["n_lo"] = int(ll.size)
        m["lo_mean"] = float(ll.mean()) if ll.size else math.nan
        m["lo_min"] = float(ll.min()) if ll.size else math.nan
        m["lo_max"] = float(ll.max()) if ll.size else math.nan
        # ... and the complete ones only, i.e. those between two high-dose phases
        if ln.size >= 2:
            gaps = st[1:] - (st[:-1] + ln[:-1])
            m["n_lo_between"] = int(gaps.size)
            m["lo_between_mean"] = float(gaps.mean())
            m["lo_between_min"] = float(gaps.min())
            m["lo_between_max"] = float(gaps.max())
        else:
            m["n_lo_between"] = 0
            m["lo_between_mean"] = m["lo_between_min"] = m["lo_between_max"] = math.nan
    return m


def sensitivity_factor(x2T_groups, x2T_nominal):
    """Mean over the groups of x2_group(T)/x2_nominal(T)."""
    return float(np.mean(np.asarray(x2T_groups, dtype=float)/x2T_nominal))
