"""
Objective B (assignment Option B, Phase 2 Milestone 1): minimize BS
transmit power subject to a per-user SINR target gamma, with |theta_m|=1.

    minimize_{W, theta}   ||W||_F^2
    subject to             SINR_k >= gamma for every user k
                            |theta_m| = 1

K=1 (closed form)
------------------
With MRT, SINR = P*||h_eff||^2/sigma2 (no interference). For fixed theta,
the minimum power achieving SINR=gamma is P_min = gamma*sigma2/||h_eff||^2
(Cauchy-Schwarz). Minimizing P_min over theta is therefore the SAME
problem as maximizing ||h_eff(theta)||^2 -- which is exactly what
Objective A's K=1 AO (ao.run_ao with mode="mrt") already converges to:
at a fixed point, w is MRT-matched to h_eff AND theta is phase-aligned to
w, so the direction found is the same regardless of which objective
(max-rate for a fixed P, or min-power for a fixed SINR target) drove the
search -- only the power *level* differs between the two problems, not
the optimal (theta, W-direction) pair. So we reuse run_ao directly with
an arbitrary reference power (its optimal theta/direction do not depend
on P for K=1 MRT), then compute the exact minimum power in closed form.
This matches "AO converges in one pass" (K=1 AO converges in ~2-3
iterations in practice, see exp3_convergence).

K>1, ZF (closed form precoder + manifold RIS step)
----------------------------------------------------
With W = H_eff^H (H_eff H_eff^H)^-1 diag(sqrt(gamma*sigma2)), ZF gives
SINR_k = gamma exactly for every user (zero interference), with total
power P = gamma*sigma2 * trace((H_eff H_eff^H)^-1) (closed form, derived
in this module's docstring tests). The RIS step minimizes this trace over
theta via Riemannian gradient DESCENT on the unit-modulus manifold (same
tangent-projection / Armijo / retraction structure as ris_opt.py's
phase_update_manifold, but minimizing instead of maximizing).

Infeasibility / outage
------------------------
If the required power exceeds Pmax, the instance is reported as an
outage (infeasible at the given gamma and Pmax) rather than silently
clipped.
"""
from __future__ import annotations

import numpy as np
import torch

from ao import run_ao
from metrics import effective_channel

torch.set_num_threads(1)


# ---------------------------------------------------------------------
# K = 1
# ---------------------------------------------------------------------

def required_power_single_user(h_eff: np.ndarray, gamma: float, sigma2: float) -> float:
    """P_min = gamma*sigma2 / ||h_eff||^2 (K=1, MRT, closed form)."""
    return float(gamma * sigma2 / np.sum(np.abs(h_eff) ** 2))


def min_power_single_user(H: np.ndarray, G: np.ndarray, gamma: float, sigma2: float,
                           theta0: np.ndarray | None = None, max_iter: int = 20) -> tuple:
    """Returns (theta, W, P_required). W has ||W||_F^2 = P_required exactly
    and achieves SINR = gamma exactly (not just >=) -- the minimum-power
    solution always meets the constraint with equality."""
    theta, W_ref, _ = run_ao(H, G, 1.0, sigma2, mode="mrt", theta0=theta0, max_iter=max_iter)
    h_eff = effective_channel(G, H, theta)[0]
    P_req = required_power_single_user(h_eff, gamma, sigma2)
    W = (W_ref / np.linalg.norm(W_ref)) * np.sqrt(P_req)  # same direction, rescaled
    return theta, W, P_req


# ---------------------------------------------------------------------
# K > 1, ZF
# ---------------------------------------------------------------------

def zf_min_power_precoder(H_eff: np.ndarray, gamma: float, sigma2: float) -> tuple:
    """ZF precoder achieving SINR_k = gamma exactly for every user with
    minimum total power. Returns (W, P_total):
        W = H_eff^H (H_eff H_eff^H)^-1 diag(sqrt(gamma*sigma2))
        P_total = gamma*sigma2 * trace((H_eff H_eff^H)^-1)
    """
    gram_inv = np.linalg.inv(H_eff @ H_eff.conj().T)
    pseudo_inv = H_eff.conj().T @ gram_inv
    W = pseudo_inv * np.sqrt(gamma * sigma2)
    P_total = float(gamma * sigma2 * np.real(np.trace(gram_inv)))
    return W, P_total


def _power_objective_and_grad(theta_np: np.ndarray, G: np.ndarray, H: np.ndarray) -> tuple:
    """trace((H_eff H_eff^H)^-1) and its gradient wrt theta, via torch autograd.
    (gamma, sigma2 are an overall scalar multiplier on power and do not affect
    the optimal theta, so the RIS step only needs this trace, unscaled.)"""
    theta = torch.tensor(theta_np, dtype=torch.complex128, requires_grad=True)
    Gt = torch.tensor(G, dtype=torch.complex128)
    Ht = torch.tensor(H, dtype=torch.complex128)

    Phi_H = theta.unsqueeze(1) * Ht
    H_eff = Gt.conj().T @ Phi_H
    gram = H_eff @ H_eff.conj().T
    obj = torch.real(torch.trace(torch.linalg.inv(gram)))

    obj.backward()
    grad = theta.grad.detach().numpy().copy()
    return float(obj.item()), grad


def phase_update_manifold_min_power(G: np.ndarray, H: np.ndarray, theta0: np.ndarray,
                                     iters: int = 50) -> tuple:
    """Riemannian gradient DESCENT on trace((H_eff H_eff^H)^-1) over the
    unit-modulus manifold (lower trace -> lower required power). Mirrors
    ris_opt.phase_update_manifold's structure (tangent projection, unit
    direction, Armijo, retraction) with the sign flipped for descent and a
    different objective. Returns (theta, objective_trace); the trace is
    non-increasing by construction."""
    theta = np.array(theta0, dtype=complex).copy()
    theta /= np.abs(theta)

    obj, grad = _power_objective_and_grad(theta, G, H)
    trace = [obj]

    for _ in range(iters):
        tan_grad = grad - (np.conj(theta) * grad).real * theta
        grad_norm = float(np.sqrt(np.sum(np.abs(tan_grad) ** 2)))
        if grad_norm < 1e-10:
            break
        u = -tan_grad / grad_norm  # descent direction

        step = np.pi
        accepted = False
        theta_new = obj_new = grad_new = None
        for _bt in range(40):
            cand = theta + step * u
            cand /= np.abs(cand)
            obj_cand, grad_cand = _power_objective_and_grad(cand, G, H)
            if obj_cand <= obj - 1e-4 * step * grad_norm:  # Armijo for descent
                theta_new, obj_new, grad_new = cand, obj_cand, grad_cand
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        theta, obj, grad = theta_new, obj_new, grad_new
        trace.append(obj)

    return theta, trace


def min_power_zf(H: np.ndarray, G: np.ndarray, gamma: float, sigma2: float,
                  theta0: np.ndarray | None = None, max_iter: int = 20,
                  manifold_iters: int = 50, tol: float = 1e-6) -> tuple:
    """AO for Objective B, K>1, ZF. Returns (theta, W, P_required, power_history).
    power_history is non-increasing by construction (safeguarded AO, same
    pattern as ao.run_ao)."""
    M = H.shape[0]
    if theta0 is None:
        theta = np.ones(M, dtype=complex)
    else:
        theta0 = np.asarray(theta0, dtype=complex)
        theta = theta0 / np.abs(theta0)

    W = None
    best_power = np.inf
    history = []

    for _ in range(max_iter):
        H_eff = effective_channel(G, H, theta)
        W_cand, P_cand = zf_min_power_precoder(H_eff, gamma, sigma2)
        if P_cand <= best_power:
            W, best_power = W_cand, P_cand
        history.append(best_power)

        theta_cand, _ = phase_update_manifold_min_power(G, H, theta, iters=manifold_iters)
        H_eff_cand = effective_channel(G, H, theta_cand)
        _, P_cand = zf_min_power_precoder(H_eff_cand, gamma, sigma2)
        if P_cand <= best_power:
            theta, best_power = theta_cand, P_cand
        history.append(best_power)

        if len(history) >= 4 and abs(history[-1] - history[-3]) < tol:
            break

    H_eff = effective_channel(G, H, theta)
    W, P_required = zf_min_power_precoder(H_eff, gamma, sigma2)
    return theta, W, P_required, history


def is_outage(P_required: float, Pmax: float) -> bool:
    """True if the required power exceeds the BS power budget (infeasible)."""
    return P_required > Pmax


# ---------------------------------------------------------------------
# AF relay (K=1 only) -- required BS (hop-1) power for a target end-to-end SINR
# ---------------------------------------------------------------------

def af_relay_required_power(cfg, rng, ue_pos: np.ndarray, gamma: float, sigma2: float) -> float:
    """Inverts the AF relay's end-to-end SNR formula
    gamma = gamma1*gamma2/(gamma1+gamma2+1) (gamma2 fixed: the relay always
    transmits at its own power budget Pmax, per benchmarks.af_relay_rate)
    for gamma1, then converts to the required BS (hop-1) transmit power
    P = gamma1*sigma2/||h_br||^2. Returns float('inf') if gamma2 <= gamma
    (infeasible at *any* hop-1 power, since the second hop alone caps the
    achievable end-to-end SINR below the target)."""
    from benchmarks import _single_path_channel  # relay-specific channel helper

    h_br = _single_path_channel(cfg, rng, cfg.Nt, cfg.bs_pos, cfg.ris_pos, cfg.bs_gain_lin)
    ue = ue_pos[0]
    h_ru = _single_path_channel(cfg, rng, 1, cfg.ris_pos, ue, cfg.ue_gain_lin)[0]
    gamma2 = cfg.Pmax_w * np.abs(h_ru) ** 2 / sigma2

    if gamma2 <= gamma:
        return float("inf")
    gamma1 = gamma * (gamma2 + 1) / (gamma2 - gamma)
    return float(gamma1 * sigma2 / np.sum(np.abs(h_br) ** 2))
