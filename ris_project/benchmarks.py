"""
Baseline benchmarks for comparison against AO (see Step 4 of the
assignment / Section 0 of the master prompt):

1. no_ris_rate      - direct (blocked) link only.
2. random_phase_rate - RIS present but with uniformly random phases, averaged.
3. af_relay_rate     - half-duplex Amplify-and-Forward relay (K=1 only).
"""
from __future__ import annotations

import numpy as np

from config import Config
from channel import free_space_pathloss, molecular_absorption
from beamforming import mrt, zf, rzf
from metrics import effective_channel, sum_rate


def no_ris_rate(d: np.ndarray, P: float, sigma2: float) -> float:
    """Benchmark 1: direct (blocked) link only, no RIS. MRT for K=1, ZF for K>1.
    d: direct channel, shape (K, Nt) (rows h_k^H = d_k^H)."""
    K = d.shape[0]
    if K == 1:
        W = mrt(d[0], P)
    else:
        W = zf(d, P)
    return sum_rate(d, W, sigma2)


def random_phase_rate(H: np.ndarray, G: np.ndarray, P: float, sigma2: float, mode: str,
                       rng: np.random.Generator, n_draws: int = 200,
                       d: np.ndarray | None = None) -> float:
    """Benchmark 2: RIS present, phases drawn uniformly at random, rate averaged
    over n_draws draws. mode in {"mrt" (K=1), "zf", "rzf"} selects the BS precoder."""
    M = H.shape[0]
    rates = np.zeros(n_draws)
    for i in range(n_draws):
        theta = np.exp(1j * rng.uniform(0, 2 * np.pi, M))
        H_eff = effective_channel(G, H, theta, d)
        if mode == "mrt":
            W = mrt(H_eff[0], P)
        elif mode == "zf":
            W = zf(H_eff, P)
        elif mode == "rzf":
            W = rzf(H_eff, P, sigma2)
        else:
            raise ValueError(f"unknown mode: {mode}")
        rates[i] = sum_rate(H_eff, W, sigma2)
    return float(np.mean(rates))


def _single_path_channel(cfg: Config, rng: np.random.Generator, N: int,
                          tx_pos: tuple, rx_pos: tuple, gain_lin: float) -> np.ndarray:
    """Single-path Rayleigh channel (no SV sparsity, no blockage) between tx_pos
    and rx_pos, used for the two AF-relay hops (the relay sits at the RIS
    location, so it has unobstructed LoS-like propagation to both ends)."""
    d = np.linalg.norm(np.array(rx_pos) - np.array(tx_pos))
    amp = np.sqrt(free_space_pathloss(d, cfg.wavelength) * molecular_absorption(cfg.absorption_coeff(), d) * gain_lin)
    return amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


def af_relay_rate(cfg: Config, rng: np.random.Generator, ue_pos: np.ndarray, P: float, sigma2: float) -> float:
    """Benchmark 3 (K=1 only): half-duplex Amplify-and-Forward relay with a single
    antenna placed at the RIS location.

    gamma1 = BS->relay SNR with MRT at the BS, transmit power P (the swept Ptx).
    gamma2 = relay->UE SNR, relay transmits at its own fixed power budget Pmax.
    End-to-end SNR gamma = gamma1*gamma2 / (gamma1 + gamma2 + 1) (classical AF
    relay SNR, see docs/REFERENCES.md [3]); half-duplex halves the rate
    (pre-log factor 1/2, two time slots per symbol).

    Assumption: both hops use the same free-space-path-loss + molecular-
    absorption model as the rest of the link, with no blockage penalty
    (the relay is not behind the blocker).
    """
    h_br = _single_path_channel(cfg, rng, cfg.Nt, cfg.bs_pos, cfg.ris_pos, cfg.bs_gain_lin)
    gamma1 = P * np.sum(np.abs(h_br) ** 2) / sigma2  # MRT: SNR = P*||h||^2/sigma2

    ue = ue_pos[0]
    h_ru = _single_path_channel(cfg, rng, 1, cfg.ris_pos, ue, cfg.ue_gain_lin)[0]
    gamma2 = cfg.Pmax_w * np.abs(h_ru) ** 2 / sigma2

    gamma = gamma1 * gamma2 / (gamma1 + gamma2 + 1)
    return 0.5 * np.log2(1 + gamma)
