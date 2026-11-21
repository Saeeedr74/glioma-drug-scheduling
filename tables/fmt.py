"""Number formatting of the tables of the paper: two decimals (rounded half up) in plain notation for
0.1 <= |x| < 1e5, scientific notation with a two-decimal mantissa otherwise; integers stay integers."""
from decimal import Decimal, ROUND_HALF_UP
import math

def _q(x, places="0.01"):
    return Decimal(repr(float(x))).quantize(Decimal(places), rounding=ROUND_HALF_UP)

def sci(x, times=r"{\times}"):
    x = float(x)
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    mant = _q(x / 10**e)
    if abs(mant) >= 10:           # 9.996 -> 10.00
        mant = _q(float(mant) / 10); e += 1
    return f"{mant}{times}10^{{{e}}}"

def num(x, integer=False, lo=0.1, hi=1e5):
    """Plain two decimals for lo <= |x| < hi, scientific otherwise; integers stay integers."""
    x = float(x)
    if integer:
        assert abs(x - round(x)) < 1e-9, x
        return str(int(round(x)))
    if x == 0:
        return "0"
    if lo <= abs(x) < hi:
        return str(_q(x))
    return sci(x)

def m(s):
    return f"${s}$"
