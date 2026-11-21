"""Exact-arithmetic check of the sign of Theta (Lemma 3 of the Lyapunov appendix).

Claim: for the nominal parameters, u in [50, 200] and c = 3,

    Theta(x1, x2) = -Om1*w^2 + eps*w + x2*(b1*w + c*(chi(x2) - b2*x1)),   w = 1 - x1,

is negative at every point of the rectangle Pi = [x1_min, 1] x [0, x2_bar] outside the box
R = {0 <= 1 - x1 <= w_R, 0 <= x2 <= x2_R}, with w_R = 6.92e-3 and x2_R = 1.78e-5.

verify_proof_symbolic_and_grid.py locates the box on a grid in double precision. This script
repeats the check with the same cell bound in exact rational arithmetic (fractions.Fraction),
on an adaptive partition, so that no rounding error enters. The rectangle checked here,
[0.148, 1] x [0, 0.353], contains Pi (x1_min = 0.14824..., x2_bar = 0.35237...).

Cell bound on [x1a, x1b] x [x2a, x2b] (valid at every point of the cell):
    max over w in [1 - x1b, 1 - x1a] of (-Om1*w^2 + eps*w)  +  max(x2a*B, x2b*B),
    B = b1*(1 - x1a) + c*(chi(x2a) - b2*x1a),
because the bracket of Theta increases with w and chi decreases with x2 (p2*x4bar <= Om2*a2^2).

Needs the Python standard library only. Run:  python lyapunov/verify_lemma_exact.py
"""
import sys
from fractions import Fraction as F

Om1, Om2 = F("0.0068"), F("0.012")
b1, b2 = F("0.018"), F("0.0018")
p1, p2 = F("4.7e-8"), F("4.7e-5")
a1, a2, zeta = F(1), F(1), F("0.2")
u_min, u_max, c = F(50), F(200), F(3)
x4bar = u_min / zeta
eps = p1 * u_max / (zeta * a1)
m = Om2 - p2 * x4bar / a2

X1_MIN, X2_BAR = F("0.148"), F("0.353")      # rectangle that contains Pi
W_R, X2_R = F("0.00692"), F("0.0000178")     # box R of the lemma


def chi(x2):
    return Om2 * (1 - x2) - p2 * x4bar / (a2 + x2)


def g(w):
    return -Om1 * w * w + eps * w


VERTEX = eps / (2 * Om1)


def cell_bound(x1a, x1b, x2a, x2b):
    wa, wb = 1 - x1b, 1 - x1a
    fw = max(g(wa), g(wb))
    if wa <= VERTEX <= wb:
        fw = max(fw, g(VERTEX))
    B = b1 * wb + c * (chi(x2a) - b2 * x1a)
    return fw + max(x2a * B, x2b * B)


def verify(rect, max_depth=60):
    """Bisect until the bound is negative on every cell. Returns (ok, number of cells, largest bound)."""
    stack, n, worst = [(rect, 0)], 0, None
    while stack:
        (x1a, x1b, x2a, x2b), depth = stack.pop()
        bound = cell_bound(x1a, x1b, x2a, x2b)
        n += 1
        if bound < 0:
            worst = bound if worst is None or bound > worst else worst
            continue
        if depth > max_depth:
            return False, n, (x1a, x1b, x2a, x2b)
        if (x1b - x1a) >= 200 * (x2b - x2a) or depth % 2 == 0:
            mid = (x1a + x1b) / 2
            stack += [((x1a, mid, x2a, x2b), depth + 1), ((mid, x1b, x2a, x2b), depth + 1)]
        else:
            mid = (x2a + x2b) / 2
            stack += [((x1a, x1b, x2a, mid), depth + 1), ((x1a, x1b, mid, x2b), depth + 1)]
    return True, n, worst


def main():
    print(f"x4bar = {float(x4bar)}, eps = {float(eps):.6e}, m = {float(m):.6e}")
    print(f"condition p2*x4bar <= Om2*a2^2: {p2 * x4bar <= Om2 * a2**2};  condition m < beta2: {m < b2}")
    # Pi minus R is covered by two closed rectangles
    parts = {"1 - x1 >= w_R": (X1_MIN, 1 - W_R, F(0), X2_BAR),
             "1 - x1 <= w_R and x2 >= x2_R": (1 - W_R, F(1), X2_R, X2_BAR)}
    all_ok = True
    for name, rect in parts.items():
        ok, n, info = verify(rect)
        all_ok &= ok
        if ok:
            print(f"{name}: Theta < 0 on all {n} cells (largest bound {float(info):.3e})")
        else:
            print(f"{name}: NOT verified; first failing cell {[float(v) for v in info]}")
    print("Lemma verified in exact arithmetic." if all_ok else "Lemma NOT verified.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
