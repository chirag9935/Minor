"""
RIS phase-shift optimization, fixing the BS beamformer(s) W.

Single user (K=1): the problem max_theta |g^H diag(theta) H w|^2 subject to
|theta_m|=1 has a closed-form global optimum: align every term's phase
(phase_update_single_user).

Multi-user (K>1): no closed form (coupled through the sum-rate SINR
expressions), so we maximise the sum rate directly on the complex-circle
manifold {theta in C^M : |theta_m| = 1} with Riemannian gradient ascent
(phase_update_manifold), using PyTorch autograd for the Euclidean gradient,
tangent-space projection, Armijo backtracking, and normalisation retraction.
See [4], [9] in docs/REFERENCES.md.
"""
from __future__ import annotations

import numpy as np
import torch

# These autograd calls operate on tiny tensors (M up to a few hundred
# entries); torch's default multi-threaded matmul/autograd adds thread-
# management overhead that dwarfs the actual compute here, and competes
# for cores with any outer-level parallelism (e.g. several experiment
# scripts running at once). A single thread is as fast per-call and avoids
# oversubscription.
torch.set_num_threads(1)


def phase_update_single_user(g: np.ndarray, H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Closed-form optimal RIS phases for K=1: theta_m = exp(-j*angle(conj(g_m)*a_m)),
    a = H w. This exactly aligns every reflected path so they combine coherently."""
    g = np.asarray(g).reshape(-1)
    w = np.asarray(w).reshape(-1)
    a = H @ w
    return np.exp(-1j * np.angle(np.conj(g) * a))


def _objective_and_grad(theta_np: np.ndarray, G: np.ndarray, H: np.ndarray, W: np.ndarray,
                         sigma2: float) -> tuple:
    """Sum rate and its Wirtinger (Euclidean) gradient wrt theta via torch autograd."""
    theta = torch.tensor(theta_np, dtype=torch.complex128, requires_grad=True)
    Gt = torch.tensor(G, dtype=torch.complex128)
    Ht = torch.tensor(H, dtype=torch.complex128)
    Wt = torch.tensor(W, dtype=torch.complex128)

    Phi_H = theta.unsqueeze(1) * Ht            # diag(theta) @ H, shape (M, Nt)
    H_eff = Gt.conj().T @ Phi_H                 # shape (K, Nt)
    received = H_eff @ Wt                       # shape (K, K)
    power = received.abs() ** 2
    signal = torch.diagonal(power)
    interference = power.sum(dim=1) - signal
    sinr = signal / (interference + sigma2)
    rate = torch.sum(torch.log2(1 + sinr))

    rate.backward()
    grad = theta.grad.detach().numpy().copy()
    return float(rate.item()), grad


def phase_update_manifold(G: np.ndarray, H: np.ndarray, W: np.ndarray, theta0: np.ndarray,
                           sigma2: float, iters: int = 50) -> tuple:
    """Riemannian gradient ascent on the unit-modulus manifold, maximising the sum rate.

    Per iteration: Euclidean gradient (autograd) -> tangent-space projection
    grad_tan = grad - Re(conj(theta)*grad)*theta -> normalise to a unit
    ascent direction u -> Armijo backtracking line search starting from a
    step of pi radians -> retraction theta <- theta/|theta|. Normalising
    the direction before the line search matters: the raw tangent gradient
    can be tiny far from the optimum (e.g. a near-cancelling random init
    with M large), which would otherwise force many iterations of
    negligible progress. Only steps that do not decrease the objective are
    accepted, so the returned trace is non-decreasing by construction.

    Returns (theta, objective_trace).
    """
    theta = np.array(theta0, dtype=complex).copy()
    theta /= np.abs(theta)

    obj, grad = _objective_and_grad(theta, G, H, W, sigma2)
    trace = [obj]

    for _ in range(iters):
        tan_grad = grad - (np.conj(theta) * grad).real * theta
        grad_norm = float(np.sqrt(np.sum(np.abs(tan_grad) ** 2)))
        if grad_norm < 1e-10:
            break
        u = tan_grad / grad_norm

        step = np.pi
        accepted = False
        theta_new = obj_new = grad_new = None
        for _bt in range(40):
            cand = theta + step * u
            cand /= np.abs(cand)
            obj_cand, grad_cand = _objective_and_grad(cand, G, H, W, sigma2)
            if obj_cand >= obj + 1e-4 * step * grad_norm:
                theta_new, obj_new, grad_new = cand, obj_cand, grad_cand
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

        theta, obj, grad = theta_new, obj_new, grad_new
        trace.append(obj)

    return theta, trace
