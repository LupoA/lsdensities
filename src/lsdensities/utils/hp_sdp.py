"""
Arbitrary-precision primal-dual interior-point solver for the block-diagonal
semidefinite programs of the causal bootstrap (arXiv:2605.20509), following
the paper's Appendix B (which in turn follows SDPB, its Ref. [37]).

The SDP is the paper's Eq. (B1) ("dual" in solver terminology, and the one
whose optimum is the bound):

    maximize    b . y + D . Y
    subject to  Tr[A_p Y] + (B y)_p = c_p,   p = 1..P,
                Y >= 0 (block diagonal),

together with its dual Eq. (B2):

    minimize    c . x
    subject to  X = sum_p A_p x_p - D >= 0,   B^T x = b.

For feasible points, c.x - (b.y + D.Y) = X . Y >= 0 (Eq. B3): the duality gap.
The solver stops when the gap and both infeasibilities are below the requested
thresholds, which certifies the computed optimum to that accuracy.

Only the two constraint-matrix structures the causal bootstrap needs are
supported, per block:
  - "rank1": A_p = w_p v_p v_p^T, v_p = column p of a matrix V (the
    sum-of-squares blocks with the polynomial identity imposed at sample
    points, paper Eqs. (54)-(55) and App. B: "for the rank-one choices of A_i
    used in this work, the trace in Eq. (B12) can be evaluated efficiently");
  - "rank2": A_p = u_p e^T + e u_p^T, e = last unit vector (the block Z of
    the noisy-data problem, paper Eqs. (35)-(36)).

Matrix arithmetic is done by FLINT (python-flint arb_mat, as in the paper's
own implementation) at the working precision of the caller's mpmath context
(mp.dps), on midpoints only: the "approx" solves give no error bounds, and
none are needed, since the result is certified by the duality gap and the
residuals, which are recomputed at the working precision at every
iteration. Inputs and outputs are mpmath numbers.
"""

import numpy as np
from flint import arb, arb_mat, ctx
from mpmath import mp, mpf

__all__ = ["BlockSDP", "SDPError"]


class SDPError(RuntimeError):
    pass


#   Conversions and small helpers (arb_mat, midpoints only)

def _to_arb(v):
    return arb(mp.nstr(mpf(v), mp.dps + 5, strip_zeros=False))


def _to_mpf(x):
    return mpf(x.mid().str(mp.dps + 5, radius=False))


def _mat(rows):
    """arb_mat from a list of rows of mpf."""
    n, m = len(rows), len(rows[0]) if rows else 0
    return arb_mat(n, m, [_to_arb(v) for row in rows for v in row])


def _col(values):
    return arb_mat(len(values), 1, [_to_arb(v) for v in values])


def _zeros(n, m):
    return arb_mat(n, m)


def _identity(n, scale=1):
    out = arb_mat(n, n)
    for i in range(n):
        out[i, i] = scale
    return out


def _sym(A):
    return ((A + A.transpose()) * arb(0.5)).mid()


def _trace_product(A, B):
    """Tr[A B] = sum_ij A_ij B_ji, without forming the product."""
    n = A.nrows()
    return sum((A[i, j] * B[j, i] for i in range(n) for j in range(n)), arb(0))


def _max_abs(A):
    entries = A.entries()
    return max((abs(_to_mpf(v)) for v in entries), default=mpf(0))


def _solve(A, B):
    return A.solve(B, algorithm="approx").mid()


def _cholesky(A):
    """Lower-triangular L with A = L L^T (midpoints); raises SDPError if A is not positive definite."""
    n = A.nrows()
    a = [[A[i, j].mid() for j in range(n)] for i in range(n)]
    L = [[arb(0)] * n for _ in range(n)]
    for j in range(n):
        s = a[j][j] - sum((L[j][k] * L[j][k] for k in range(j)), arb(0))
        if not s.mid() > 0:
            raise SDPError("matrix not positive definite")
        L[j][j] = s.sqrt().mid()
        for i in range(j + 1, n):
            L[i][j] = ((a[i][j] - sum((L[i][k] * L[j][k] for k in range(j)), arb(0))) / L[j][j]).mid()
    return arb_mat(L)


def _max_step(L, dM, reduction):
    """Largest step a <= 1 keeping L L^T + a dM positive definite, times `reduction`
    (paper App. B / SDPB). The eigenvalues of the rescaled direction L^-1 dM L^-T only
    need to be roughly right: every accepted step is re-checked by a Cholesky factorisation."""
    Linv_dM = _solve(L, dM)
    M = _solve(L, Linv_dM.transpose())
    M = np.array([[float(M[i, j].mid()) for j in range(M.ncols())] for i in range(M.nrows())])
    lam = float(np.linalg.eigvalsh((M + M.T) / 2).min())
    return 1.0 if lam >= 0 else min(1.0, reduction / -lam)


class BlockSDP:
    """
    blocks: list of dicts, one per PSD block, each with
        {"type": "rank1", "V": n x P matrix (list of rows of mpf), "w": length-P list}
     or {"type": "rank2", "U": n x P matrix (last row ignored/zero)}
    c: length-P list; B: P x m matrix (list of rows) or None (no free variables);
    b: length m list or None; D: list (per block) of n x n matrices (lists of rows) or None.
    All numbers mpmath (or anything mpf() accepts); the working precision is mp.dps.
    """

    def __init__(self, blocks, c, B=None, b=None, D=None):
        self.c_mp = [mpf(v) for v in c]
        self.P = len(self.c_mp)
        self.B_mp = B
        self.b_mp = [mpf(v) for v in b] if b is not None else []
        self.m = len(self.b_mp)
        self.blocks_mp = blocks
        self.sizes = [len(blk["V"]) if blk["type"] == "rank1" else len(blk["U"]) for blk in blocks]
        self.D_mp = D

    def _setup(self):
        """Convert the problem to arb_mat at the current working precision."""
        self.blocks = []
        for blk in self.blocks_mp:
            if blk["type"] == "rank1":
                self.blocks.append({"type": "rank1", "V": _mat(blk["V"]), "w": [_to_arb(v) for v in blk["w"]]})
            else:
                n = len(blk["U"])
                U = [list(row) for row in blk["U"][: n - 1]] + [[mpf(0)] * self.P]
                self.blocks.append({"type": "rank2", "U": _mat(U)})
        self.c = _col(self.c_mp)
        self.D = [_mat(Db) for Db in self.D_mp] if self.D_mp is not None else [_zeros(n, n) for n in self.sizes]

    #   A_p applied in the three ways the algorithm needs (x: P x 1 arb_mat)

    def _sum_A(self, x):
        """sum_p A_p x_p, per block."""
        out = []
        for blk, n in zip(self.blocks, self.sizes):
            if blk["type"] == "rank1":
                V, w = blk["V"], blk["w"]
                scaled = arb_mat(n, self.P, [V[i, p] * w[p] * x[p, 0] for i in range(n) for p in range(self.P)])
                out.append((scaled * V.transpose()).mid())
            else:
                Ux = blk["U"] * x
                M = arb_mat(n, n)
                for i in range(n - 1):
                    M[i, n - 1] = Ux[i, 0]
                    M[n - 1, i] = Ux[i, 0]
                out.append(M)
        return out

    def _trace_A(self, Ms):
        """Tr[A_p M] for every p, as a P x 1 arb_mat (M not necessarily symmetric)."""
        tr = [arb(0)] * self.P
        for blk, M, n in zip(self.blocks, Ms, self.sizes):
            if blk["type"] == "rank1":
                V, w = blk["V"], blk["w"]
                MV = M * V
                for p in range(self.P):
                    tr[p] += w[p] * sum((V[i, p] * MV[i, p] for i in range(n)), arb(0))
            else:
                U = blk["U"]
                last = n - 1
                # Tr[(u e^T + e u^T) M] = (M[last,:] + M[:,last]^T) u
                r = arb_mat(1, n, [M[last, i] + M[i, last] for i in range(n)]) * U
                for p in range(self.P):
                    tr[p] += r[0, p]
        return arb_mat(self.P, 1, tr).mid()

    def _schur(self, Xs, Ys):
        """S_ij = sum_blocks Tr[A_i X^-1 A_j Y] (paper Eq. B12), using the rank structure."""
        P = self.P
        S = [[arb(0)] * P for _ in range(P)]
        for blk, X, Y, n in zip(self.blocks, Xs, Ys, self.sizes):
            if blk["type"] == "rank1":
                V, w = blk["V"], blk["w"]
                Vt = V.transpose()
                VXV = Vt * _solve(X, V)
                VYV = Vt * Y * V
                for i in range(P):
                    for j in range(P):
                        S[i][j] += w[i] * w[j] * VXV[i, j] * VYV[j, i]
            else:
                U = blk["U"]
                Ut = U.transpose()
                XU = _solve(X, U)
                YU = Y * U
                UXU, UYU = Ut * XU, Ut * YU
                e = n - 1
                Xinv_e = _solve(X, arb_mat(n, 1, [1 if i == e else 0 for i in range(n)]))
                eXe, eYe = Xinv_e[e, 0], Y[e, e]
                eXu = [XU[e, j] for j in range(P)]  # = u_j^T X^-1 e (X symmetric)
                eYu = [YU[e, j] for j in range(P)]
                for i in range(P):
                    for j in range(P):
                        S[i][j] += (eXu[j] * eYu[i] + eXe * UYU[j, i]
                                    + UXU[i, j] * eYe + eXu[i] * eYu[j])
        return _sym(arb_mat(S))

    #   Elimination of the free variables (paper App. B, Eqs. B15-B26)

    def _eliminate_free_variables(self):
        """
        With free variables y, the dual constraints Tr[A_p Y] + (B y)_p = c_p
        determine y = R^-1 Q1^T (c - Tr[A Y]), where B = Q1 R (thin QR). Write
        w = Q1 R^-T b, and N for an orthonormal basis of the complement of
        range(B). Then the problem is equivalent to one without free
        variables:
            maximize  alpha0 + Dt . Y   s.t.  Tr[At_q Y] = ct_q,  Y >= 0,
            At_q = sum_p N_pq A_p,  ct = N^T c,  Dt = D - sum_p w_p A_p,
            alpha0 = w . c,
        with primal x = w + N xt (paper Eqs. B19-B25). Solving the reduced
        problem avoids the B^T S^-1 B system, whose condition number is the
        square of the Schur complement's and destroys the precision near the
        optimum.
        """
        P, m = self.P, self.m
        Q, R = mp.qr(mp.matrix(self.B_mp), mode="full")
        Q1 = [[Q[i, j] for j in range(m)] for i in range(P)]
        R1 = [[R[i, j] for j in range(m)] for i in range(m)]
        z = [mpf(0)] * m
        for i in range(m):  # R^T z = b
            z[i] = (self.b_mp[i] - mp.fsum(R1[k][i] * z[k] for k in range(i))) / R1[i][i]
        w = [mp.fdot(Q1[p], z) for p in range(P)]
        self._Q1, self._R1 = Q1, R1
        self._N = _mat([[Q[i, j] for j in range(m, P)] for i in range(P)])
        self._w = _col(w)
        self._alpha0 = (self._w.transpose() * self.c)[0, 0]
        self._D_red = [(Db - Sw).mid() for Db, Sw in zip(self.D, self._sum_A_full(self._w))]
        self._c_red = (self._N.transpose() * self.c).mid()

    def _recover_y(self, Y):
        """y = R^-1 Q1^T (c - Tr[A Y])."""
        trAY = self._trace_A_full(Y)
        r = [self.c_mp[p] - _to_mpf(trAY[p, 0]) for p in range(self.P)]
        rhs = [mp.fdot([self._Q1[p][i] for p in range(self.P)], r) for i in range(self.m)]
        y = [mpf(0)] * self.m
        for i in reversed(range(self.m)):
            y[i] = (rhs[i] - mp.fsum(self._R1[i][k] * y[k] for k in range(i + 1, self.m))) / self._R1[i][i]
        return y

    #   Main loop (on the problem without free variables)

    def solve(self, gap_threshold=mpf("1e-30"), feasibility_threshold=mpf("1e-30"),
              max_iterations=500, initial_scale=mpf("1e5"), step_reduction=0.7,
              verbose=False):
        old_prec = ctx.prec
        ctx.prec = mp.prec + 20
        try:
            return self._solve(mpf(gap_threshold), mpf(feasibility_threshold), max_iterations,
                               _to_arb(initial_scale), step_reduction, verbose)
        finally:
            ctx.prec = old_prec

    def _solve(self, gap_threshold, feasibility_threshold, max_iterations, initial_scale, step_reduction, verbose):
        self._setup()
        self._sum_A_full, self._trace_A_full = self._sum_A, self._trace_A
        if self.m:
            self._eliminate_free_variables()
            N, Nt, Pr = self._N, self._N.transpose(), self.P - self.m

            def sum_A(xt):
                return self._sum_A_full((N * xt).mid())

            def trace_A(Ms):
                return (Nt * self._trace_A_full(Ms)).mid()

            def schur(Xs, Ys):
                return _sym(Nt * self._schur(Xs, Ys) * N)

            c, D, alpha0 = self._c_red, self._D_red, self._alpha0
        else:
            Pr = self.P
            sum_A, trace_A, schur = self._sum_A, self._trace_A, self._schur
            c, D, alpha0 = self.c, self.D, arb(0)

        x = arb_mat(Pr, 1)
        X = [_identity(n, initial_scale) for n in self.sizes]
        Y = [_identity(n, initial_scale) for n in self.sizes]
        dim = sum(self.sizes)

        for iteration in range(1, max_iterations + 1):
            #   residuals (paper Eqs. B7-B9)
            Pres = [(Ax - Xb - Db).mid() for Ax, Xb, Db in zip(sum_A(x), X, D)]
            dres = (c - trace_A(Y)).mid()

            primal_obj = _to_mpf(alpha0 + (c.transpose() * x)[0, 0])
            dual_obj = _to_mpf(alpha0 + sum((_trace_product(Db, Yb) for Db, Yb in zip(D, Y)), arb(0)))
            gap = abs(primal_obj - dual_obj) / max(mpf(1), abs(primal_obj) + abs(dual_obj))
            primal_err = max(_max_abs(M) for M in Pres)
            dual_err = _max_abs(dres)
            if verbose:
                print(f"{iteration:4d} primal {mp.nstr(primal_obj, 12):>20} dual {mp.nstr(dual_obj, 12):>20} "
                      f"gap {mp.nstr(gap, 3):>9} p.err {mp.nstr(primal_err, 3):>9} d.err {mp.nstr(dual_err, 3):>9}")
            if gap < gap_threshold and primal_err < feasibility_threshold and dual_err < feasibility_threshold:
                Y_mp = [[[_to_mpf(Yb[i, j]) for j in range(Yb.ncols())] for i in range(Yb.nrows())] for Yb in Y]
                y = self._recover_y(Y) if self.m else []
                return {"status": "optimal", "dual_objective": dual_obj, "primal_objective": primal_obj,
                        "y": y, "Y": Y_mp, "iterations": iteration, "gap": gap,
                        "primal_error": primal_err, "dual_error": dual_err}
            if abs(dual_obj) > mpf("1e60") or abs(primal_obj) > mpf("1e60"):
                raise SDPError("objective diverging: problem infeasible or unbounded")

            mu = sum((_trace_product(Xb, Yb) for Xb, Yb in zip(X, Y)), arb(0)) / dim
            LX = [_cholesky(Xb) for Xb in X]
            LY = [_cholesky(Yb) for Yb in Y]
            S = schur(X, Y)
            feasible = primal_err < feasibility_threshold and dual_err < feasibility_threshold

            def newton(R):
                """Solve the linearised central-path equations (paper Eqs. B11, B13, B14)."""
                Zs = [_solve(Xb, Pb * Yb - Rb) for Xb, Pb, Yb, Rb in zip(X, Pres, Y, R)]
                rhs = (-dres - trace_A(Zs)).mid()
                dx = _solve(S, rhs)
                dX = [(Pb + Adx).mid() for Pb, Adx in zip(Pres, sum_A(dx))]
                dY = [_sym(_solve(Xb, Rb - dXb * Yb)) for Xb, Rb, dXb, Yb in zip(X, R, dX, Y)]
                return dx, dX, dY

            def centering(beta, extra=None):
                R = []
                for k, (Xb, Yb) in enumerate(zip(X, Y)):
                    Rb = _identity(Xb.nrows(), beta * mu) - Xb * Yb
                    if extra is not None:
                        Rb = Rb - extra[k]
                    R.append(Rb.mid())
                return R

            #   Mehrotra predictor-corrector, with SDPB's centering choices
            beta_p = arb(0) if feasible else arb("0.3")
            dx, dX, dY = newton(centering(beta_p))
            r = _to_mpf(sum((_trace_product(Xb + dXb, Yb + dYb) for Xb, dXb, Yb, dYb in zip(X, dX, Y, dY)), arb(0))
                        / (mu * dim))
            beta_c = r**2 if r < 1 else r
            beta_c = min(max(mpf("0.1"), beta_c), mpf(1)) if feasible else max(mpf("0.3"), beta_c)
            dx, dX, dY = newton(centering(_to_arb(beta_c), [(dXb * dYb).mid() for dXb, dYb in zip(dX, dY)]))

            alpha_p = min(_max_step(L, dXb, step_reduction) for L, dXb in zip(LX, dX))
            alpha_d = min(_max_step(L, dYb, step_reduction) for L, dYb in zip(LY, dY))
            # accept the steps only if they keep X and Y positive definite
            for _ in range(60):
                X_new = [_sym(Xb + dXb * arb(alpha_p)) for Xb, dXb in zip(X, dX)]
                try:
                    for Xb in X_new:
                        _cholesky(Xb)
                    break
                except SDPError:
                    alpha_p /= 2
            for _ in range(60):
                Y_new = [_sym(Yb + dYb * arb(alpha_d)) for Yb, dYb in zip(Y, dY)]
                try:
                    for Yb in Y_new:
                        _cholesky(Yb)
                    break
                except SDPError:
                    alpha_d /= 2
            x = (x + dx * arb(alpha_p)).mid()
            X, Y = X_new, Y_new

        raise SDPError(f"no convergence in {max_iterations} iterations")
