"""
Rate and energy-efficiency metrics shared by ao.py, ris_opt.py, benchmarks.py
and the experiments.

System model
------------
    y_k = h_k^H w_k x_k + sum_{j!=k} h_k^H w_j x_j + n_k,   n_k ~ CN(0, sigma2)
    h_k^H = g_k^H diag(theta) H  (+ d_k^H if a direct/blocked term is included)
    SINR_k = |h_k^H w_k|^2 / (sum_{j!=k} |h_k^H w_j|^2 + sigma2)
    sum rate R = sum_k log2(1 + SINR_k)   [bit/s/Hz]

Energy efficiency (see docs/REFERENCES.md, [5]):
    P_total = Ptx/eta + P_BS_static + M*P_RIS + K*P_UE
    EE = BW * R / P_total   [bit/Joule]
"""
from __future__ import annotations

import numpy as np

from config import Config


def effective_channel(g: np.ndarray, H: np.ndarray, theta: np.ndarray,
                       d: np.ndarray | None = None) -> np.ndarray:
    """h_k^H = g_k^H diag(theta) H for every user k, returned stacked as rows: shape (K, Nt).

    g: (M, K), H: (M, Nt), theta: (M,), d (optional direct link): (K, Nt).
    """
    Phi_H = (theta[:, None]) * H  # diag(theta) @ H, shape (M, Nt)
    H_eff = g.conj().T @ Phi_H    # shape (K, Nt)
    if d is not None:
        H_eff = H_eff + d
    return H_eff


def sinr_per_user(H_eff: np.ndarray, W: np.ndarray, sigma2: float) -> np.ndarray:
    """SINR_k for every user, H_eff rows h_k^H (K,Nt), W columns w_k (Nt,K)."""
    K = H_eff.shape[0]
    received = H_eff @ W  # shape (K, K), entry [k,j] = h_k^H w_j
    power = np.abs(received) ** 2
    signal = np.diag(power)
    interference = power.sum(axis=1) - signal
    return signal / (interference + sigma2)


def sum_rate(H_eff: np.ndarray, W: np.ndarray, sigma2: float) -> float:
    """Sum achievable rate in bit/s/Hz: sum_k log2(1 + SINR_k)."""
    sinr = sinr_per_user(H_eff, W, sigma2)
    return float(np.sum(np.log2(1 + sinr)))


def total_power_w(cfg: Config, Ptx_w: float, M: int | None = None, K: int | None = None) -> float:
    """P_total = Ptx/eta + P_BS_static + M*P_RIS + K*P_UE, all in watts."""
    M = cfg.M if M is None else M
    K = cfg.K if K is None else K
    return Ptx_w / cfg.pa_efficiency + cfg.p_bs_static_w + M * cfg.p_ris_elem_w + K * cfg.p_ue_w


def energy_efficiency(cfg: Config, rate: float, Ptx_w: float,
                       M: int | None = None, K: int | None = None) -> tuple:
    """Returns (EE [bit/Joule], SE [bit/s/Hz]). EE = BW * rate / P_total."""
    P_total = total_power_w(cfg, Ptx_w, M, K)
    EE = cfg.BW * rate / P_total
    SE = rate
    return EE, SE
