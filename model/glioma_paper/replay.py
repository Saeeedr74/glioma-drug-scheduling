"""Open-loop replay of a dose schedule on the nominal or a perturbed plant."""
import numpy as np

from .model import NOMINAL, rk4_step


def replay(u, x0, p=None):
    """u[k] is the dose applied on day k, i.e. on [k, k+1). Returns X of shape (len(u)+1, 4):
    the states on days 0..len(u)."""
    p = NOMINAL if p is None else p
    x = tuple(float(v) for v in x0)
    X = [x]
    for uk in u:
        x = rk4_step(x, float(uk), p)
        X.append(x)
    return np.array(X)


def max_rel_dx2(x2_a, x2_b):
    """max |dx2| / max(x2, 1e-12), the replay criterion of check 1.4 b."""
    a = np.asarray(x2_a, dtype=float)
    b = np.asarray(x2_b, dtype=float)
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n] - b[:n])/np.maximum(b[:n], 1e-12)))
