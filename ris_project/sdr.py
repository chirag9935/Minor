"""
Phase 2, Milestone 6(b): SDR (semidefinite relaxation) for RIS phases,
applied to the max-min-SINR problem with the BS beamformer W fixed
(assignment Approach A, step 2). Sum-rate is not SDR-friendly; max-min
SINR is, because every SINR constraint becomes LINEAR in a relaxed PSD
matrix variable.

Derivation (verified numerically against a direct computation, see
tests/test_core.py and the module docstring below for the exact
conjugation convention used -- getting this wrong is the single easiest
way to silently build a solver for the wrong problem):

    h_k^H w_j = g_k^H diag(theta) H w_j
              = sum_m conj(g_k)_m * theta_m * (H w_j)_m
              = theta^T u_kj                              (theta NOT conjugated)
    where      u_kj = conj(g_k) elementwise* (H w_j)        (length-M vector)

    |h_k^H w_j|^2 = |theta^T u_kj|^2
                  = trace( (u_kj u_kj^H) * V ),  V := conj(theta) theta^T

V is Hermitian, PSD (rank 1 for any valid theta), with diag(V) = 1 (since
V[m,m] = |theta_m|^2 = 1). The SDR relaxes "V has rank 1" away, keeping
only V >> 0, diag(V) = 1 -- a convex SDP. Every SINR constraint is then
LINEAR in V:
    SINR_k(V) = trace(A_kk V) / (sum_{j!=k} trace(A_kj V) + sigma2) >= gamma
    <=>        trace(A_kk V) - gamma*sum_{j!=k} trace(A_kj V) >= gamma*sigma2
where A_kj = u_kj u_kj^H. For a target gamma this is a convex feasibility
SDP; bisecting on gamma finds the largest feasible max-min SINR.

After solving, theta is recovered from the (generally rank > 1) V by
Gaussian randomization: draw many r ~ CN(0, V), project each to the
unit-modulus manifold (theta_cand = exp(j*angle(r))), evaluate the
ACTUAL (not relaxed) max-min SINR for each candidate, keep the best.
"""
from __future__ import annotations

import time

import numpy as np
import cvxpy as cp
import torch

torch.set_num_threads(1)


def build_u(G: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """u[k,j] = conj(g_k) * (H w_j) (elementwise), shape (K, K, M)."""
    K = G.shape[1]
    Hw = H @ W  # (M, K), column j = H w_j
    u = np.zeros((K, K, H.shape[0]), dtype=complex)
    for k in range(K):
        for j in range(K):
            u[k, j] = np.conj(G[:, k]) * Hw[:, j]
    return u


def actual_min_sinr(theta: np.ndarray, u: np.ndarray, sigma2: float) -> float:
    """Direct (non-relaxed) min-SINR achieved by a given unit-modulus theta."""
    K = u.shape[0]
    received = np.array([[theta @ u[k, j] for j in range(K)] for k in range(K)])
    power = np.abs(received) ** 2
    sinr = np.diag(power) / (power.sum(axis=1) - np.diag(power) + sigma2)
    return float(np.min(sinr))


def _build_feasibility_problem(u: np.ndarray, sigma2: float, M: int):
    """Builds the feasibility SDP ONCE per draw, with the target SINR gamma
    as a cp.Parameter; bisection then only updates gamma.value and re-solves,
    avoiding repeated cvxpy canonicalization overhead across the ~20
    bisection calls per draw. (Diagnosed, but NOT the cause of the M=64
    out-of-memory kill of the full M6(b) sweep -- see the solver note on
    sdr_max_min_sinr for the actual root cause and fix.)"""
    K = u.shape[0]
    gamma = cp.Parameter(nonneg=True)
    V = cp.Variable((M, M), hermitian=True)
    constraints = [V >> 0, cp.diag(V) == 1]
    for k in range(K):
        A_kk = np.outer(u[k, k], np.conj(u[k, k]))
        terms = [cp.real(cp.trace(A_kk @ V))]
        for j in range(K):
            if j == k:
                continue
            A_kj = np.outer(u[k, j], np.conj(u[k, j]))
            terms.append(-gamma * cp.real(cp.trace(A_kj @ V)))
        constraints.append(cp.sum(terms) >= gamma * sigma2)

    prob = cp.Problem(cp.Minimize(0), constraints)
    return prob, gamma, V


def _feasible(gamma_val: float, prob: cp.Problem, gamma_param: cp.Parameter,
              V: cp.Variable, solver: str) -> tuple:
    """Re-solves the (already canonicalized) feasibility SDP at a new target
    gamma. Returns (feasible: bool, V or None)."""
    gamma_param.value = gamma_val
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        return False, None
    feasible = prob.status in ("optimal", "optimal_inaccurate")
    return feasible, (V.value if feasible else None)


def sdr_max_min_sinr(G: np.ndarray, H: np.ndarray, W: np.ndarray, sigma2: float,
                      n_randomizations: int = 200, gamma_hi: float = 1e4,
                      bisection_iters: int = 20, solver: str = "SCS",
                      rng: np.random.Generator | None = None) -> tuple:
    """Bisects on gamma to find the SDR max-min-SINR solution, then recovers
    theta via Gaussian randomization. Returns (theta, achieved_min_sinr,
    elapsed_seconds).

    Solver note (root-caused after the full M6(b) sweep was OS-killed for
    running out of memory): CLARABEL (an interior-point method) solves a
    SINGLE feasibility SDP at M=64 (a 64x64 complex-Hermitian V, embedded as
    a 128x128 real PSD cone) in ~20-25s using ~3.9 GB RSS -- confirmed via a
    standalone single-call probe, not a leak across repeated calls. This is
    architectural, not a bug: a dense interior-point method's per-iteration
    barrier Hessian for an nxn PSD cone is an n^2 x n^2 object, i.e. O(n^4)
    memory (n=128 here); CLARABEL was never going to fit the stated M=64
    budget. SCS (a first-order operator-splitting method) solves the same
    M=64 problem in ~0.4s using ~0.3 GB -- confirmed to agree with CLARABEL
    on every feasible/infeasible decision tested at M=32 (including right at
    the feasibility boundary), so this is a performance fix, not an accuracy
    trade-off. CLARABEL remains fine (and was used throughout development)
    at the smaller M values in the Phase 2 spec's own M=16/32 configs."""
    rng = np.random.default_rng() if rng is None else rng
    t0 = time.time()
    M = H.shape[0]
    u = build_u(G, H, W)

    # Numerical hygiene: this problem's natural scale is tiny (sigma2 ~1e-12 W,
    # |u| ~1e-7), well below typical SDP-solver absolute tolerances -- solved
    # unscaled, the solver can report "feasible" for a gamma/V pair that
    # actually violates a constraint by a small absolute (but large relative)
    # margin (confirmed empirically: an unscaled solve accepted gamma~0.41 at
    # a point whose true constraint value was *negative*). Rescale u (and
    # correspondingly sigma2) so sigma2_scaled=1 before solving; gamma is a
    # dimensionless SINR ratio, so this is an exact equivalence, not an
    # approximation -- the scaled and unscaled problems share the same
    # feasible gamma and the same optimal V (diag(V)=1 is scale-free).
    u_scale = 1.0 / np.sqrt(sigma2)
    u_s = u * u_scale
    sigma2_s = 1.0

    prob, gamma_param, V_var = _build_feasibility_problem(u_s, sigma2_s, M)

    gamma_lo, gamma_best, V_best = 0.0, 0.0, np.eye(M, dtype=complex)
    for _ in range(bisection_iters):
        gamma_mid = (gamma_lo + gamma_hi) / 2
        feasible, V = _feasible(gamma_mid, prob, gamma_param, V_var, solver)
        if feasible:
            gamma_lo, gamma_best, V_best = gamma_mid, gamma_mid, V
        else:
            gamma_hi = gamma_mid

    # Gaussian randomization to recover a unit-modulus theta from V_best
    V_best = (V_best + V_best.conj().T) / 2  # enforce exact Hermitian symmetry (solver round-off)
    eigvals, eigvecs = np.linalg.eigh(V_best)
    eigvals = np.clip(eigvals, 0, None)
    sqrt_V = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    best_theta, best_sinr = None, -np.inf
    for _ in range(n_randomizations):
        z = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2)
        r = sqrt_V @ z
        theta_cand = np.exp(1j * np.angle(r))
        sinr_cand = actual_min_sinr(theta_cand, u, sigma2)
        if sinr_cand > best_sinr:
            best_sinr, best_theta = sinr_cand, theta_cand

    elapsed = time.time() - t0
    return best_theta, best_sinr, elapsed


# ---------------------------------------------------------------------
# Manifold optimization baseline for the SAME max-min-SINR objective
# (for the Phase 2 spec's required SDR-vs-manifold comparison). Operates
# directly on the true (non-relaxed) objective via a smooth soft-min
# surrogate, avoiding any SDP relaxation gap.
# ---------------------------------------------------------------------

def _softmin_objective_and_grad(theta_np: np.ndarray, u: np.ndarray, sigma2: float,
                                 beta: float) -> tuple:
    """Smooth soft-min surrogate for min_k SINR_k: -1/beta * logsumexp(-beta*SINR_k).
    Approaches the true min as beta -> inf; beta finite keeps it differentiable
    everywhere (the true min has a kink at ties)."""
    theta = torch.tensor(theta_np, dtype=torch.complex128, requires_grad=True)
    ut = torch.tensor(u, dtype=torch.complex128)  # (K, K, M)
    K = ut.shape[0]

    received = torch.einsum("m,kjm->kj", theta, ut)  # (K,K): received[k,j] = theta^T u_kj
    power = received.abs() ** 2
    signal = torch.diagonal(power)
    interference = power.sum(dim=1) - signal
    sinr = signal / (interference + sigma2)

    obj = -torch.logsumexp(-beta * sinr, dim=0) / beta
    obj.backward()
    grad = theta.grad.detach().numpy().copy()
    return float(obj.item()), grad


def manifold_max_min_sinr(G: np.ndarray, H: np.ndarray, W: np.ndarray, sigma2: float,
                           theta0: np.ndarray, iters: int = 100, beta: float = 50.0) -> tuple:
    """Riemannian gradient ascent (same tangent-projection/Armijo/retraction
    structure as ris_opt.phase_update_manifold) on the soft-min-SINR
    surrogate. Returns (theta, achieved_min_sinr, elapsed_seconds)."""
    t0 = time.time()
    u = build_u(G, H, W)
    theta = np.array(theta0, dtype=complex).copy()
    theta /= np.abs(theta)

    obj, grad = _softmin_objective_and_grad(theta, u, sigma2, beta)
    for _ in range(iters):
        tan_grad = grad - (np.conj(theta) * grad).real * theta
        grad_norm = float(np.sqrt(np.sum(np.abs(tan_grad) ** 2)))
        if grad_norm < 1e-12:
            break
        step = np.pi
        accepted = False
        for _bt in range(40):
            cand = theta + step * (tan_grad / grad_norm)
            cand /= np.abs(cand)
            obj_cand, grad_cand = _softmin_objective_and_grad(cand, u, sigma2, beta)
            if obj_cand >= obj + 1e-4 * step * grad_norm:
                theta, obj, grad = cand, obj_cand, grad_cand
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    elapsed = time.time() - t0
    return theta, actual_min_sinr(theta, u, sigma2), elapsed
