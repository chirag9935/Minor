"""
Phase 2, Milestone 4: hardware limitations -- discrete (quantized) RIS
phase shifts.

Real RIS elements can only realise a finite set of phase states (b-bit
control -> 2^b equally-spaced levels in [0, 2*pi)), not a continuous
phase. This module evaluates the rate loss from quantizing an AO-optimal
continuous phase solution, and a simple "quantization-aware" refinement
that recovers some of that loss by locally re-optimizing over the
discrete level set.
"""
from __future__ import annotations

import numpy as np

from metrics import effective_channel, sum_rate


def quantize_phase(theta: np.ndarray, bits: int | None) -> np.ndarray:
    """Round each theta_m's phase to the nearest of 2^bits equally-spaced
    levels in [0, 2*pi). bits=None means continuous (no quantization,
    returns theta unchanged). Always returns exactly unit-modulus values."""
    if bits is None:
        return theta / np.abs(theta)
    levels = 2 ** bits
    step = 2 * np.pi / levels
    angles = np.angle(theta) % (2 * np.pi)
    q_angles = np.round(angles / step) * step
    return np.exp(1j * q_angles)


def quantized_coordinate_refine(G: np.ndarray, H: np.ndarray, W: np.ndarray,
                                 theta_init: np.ndarray, bits: int, sigma2: float,
                                 max_rounds: int = 3) -> np.ndarray:
    """Quantization-aware refinement: starting from the quantized continuous
    solution, sweep each RIS element once per round and set it to whichever
    of the 2^bits discrete levels maximises the sum rate (all other
    elements held fixed -- coordinate descent in the discrete level set).
    Stops early if a full round makes no change. Simple by design (keeps
    the project's "no cleverness" style): O(M * 2^bits) rate evaluations
    per round, each O(M*Nt) -- fine for the M, bits values used here."""
    levels = 2 ** bits
    angle_grid = np.arange(levels) * (2 * np.pi / levels)
    candidates = np.exp(1j * angle_grid)  # (levels,)

    theta = quantize_phase(theta_init, bits).copy()
    M = len(theta)

    for _ in range(max_rounds):
        changed = False
        for m in range(M):
            best_rate = -np.inf
            best_val = theta[m]
            original = theta[m]
            for cand in candidates:
                theta[m] = cand
                H_eff = effective_channel(G, H, theta)
                r = sum_rate(H_eff, W, sigma2)
                if r > best_rate:
                    best_rate = r
                    best_val = cand
            theta[m] = best_val
            if best_val != original:
                changed = True
        if not changed:
            break

    return theta


def rate_with_quantization(G: np.ndarray, H: np.ndarray, W: np.ndarray, theta: np.ndarray,
                            bits: int | None, sigma2: float, refine: bool = False,
                            max_rounds: int = 3) -> tuple:
    """Quantizes theta to `bits` (None = continuous), optionally refines it
    (quantization-aware coordinate descent), and returns (rate, theta_q)."""
    theta_q = quantize_phase(theta, bits)
    if refine and bits is not None:
        theta_q = quantized_coordinate_refine(G, H, W, theta_q, bits, sigma2, max_rounds)
    H_eff = effective_channel(G, H, theta_q)
    return sum_rate(H_eff, W, sigma2), theta_q
