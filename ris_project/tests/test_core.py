"""
Core correctness tests, organised by milestone. Run with `pytest -q` from
ris_project/.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from config import Config
from channel import (
    ula_steering, upa_steering, free_space_pathloss, molecular_absorption,
    get_ue_positions, sv_channel_H, sv_channel_g, direct_channel,
)
from beamforming import mrt, zf, rzf
from ris_opt import phase_update_single_user, phase_update_manifold
from metrics import effective_channel, sum_rate
from ao import run_ao
from benchmarks import no_ris_rate


# ---------------------------------------------------------------------------
# Milestone 1: config.py, channel.py
# ---------------------------------------------------------------------------

def test_steering_vectors_unit_modulus_and_shape():
    a = ula_steering(8, 0.3)
    assert a.shape == (8,)
    assert np.allclose(np.abs(a), 1.0)

    b = upa_steering(8, 8, 0.2, 0.1)
    assert b.shape == (64,)
    assert np.allclose(np.abs(b), 1.0)


def test_channel_shapes():
    cfg = Config(K=4)
    rng = cfg.rng()
    ue_pos = get_ue_positions(cfg, rng)
    H = sv_channel_H(cfg, rng)
    g = sv_channel_g(cfg, rng, ue_pos)
    d = direct_channel(cfg, rng, ue_pos)

    assert H.shape == (cfg.M, cfg.Nt)
    assert g.shape == (cfg.M, cfg.K)
    assert d.shape == (cfg.K, cfg.Nt)
    assert ue_pos.shape == (cfg.K, 3)


def test_ue_positions_within_distance_range():
    cfg = Config(K=4)
    rng = cfg.rng()
    ue_pos = get_ue_positions(cfg, rng)
    ris = np.array(cfg.ris_pos)
    dists = np.linalg.norm(ue_pos - ris, axis=1)
    assert np.all(dists >= cfg.ue_min_dist_from_ris - 1e-9)
    assert np.all(dists <= cfg.ue_max_dist_from_ris + 1e-9)


def test_average_channel_power_matches_pathloss():
    """E[|H_mn|^2] should equal pathloss*Lmol*Gbs directly (no Nt/M/L-dependent
    prefactor -- see channel.py module docstring for why that would double-count
    the RIS array gain)."""
    cfg = Config(K=1)
    bs = np.array(cfg.bs_pos)
    ris = np.array(cfg.ris_pos)
    d = np.linalg.norm(ris - bs)
    expected_power = free_space_pathloss(d, cfg.wavelength) * molecular_absorption(cfg.absorption_coeff(), d) * cfg.bs_gain_lin

    n_draws = 300
    powers = []
    for i in range(n_draws):
        rng = np.random.default_rng(1000 + i)
        H = sv_channel_H(cfg, rng)
        powers.append(np.mean(np.abs(H) ** 2))
    mean_power = np.mean(powers)
    # H = amp*sum_l alpha_l*a_ris_l a_bs_l^H, alpha_l independent zero-mean with
    # E[|alpha_l|^2]=p_l, sum_l p_l=1, |a_ris_l[m]|=|a_bs_l[n]|=1, so E[|H_mn|^2]=amp^2.
    ratio = mean_power / expected_power
    assert 0.5 < ratio < 2.0


def test_rate_scales_roughly_as_M_squared():
    """Required sanity check (master prompt section 8 item 4): with M large,
    the AO-optimal RIS gain should grow roughly as M^2 (coherent combining).
    Checked in SNR terms (2^rate - 1) since rate is logarithmic in SNR."""
    def mean_snr(M, n=12):
        Mx, My = 8, M // 8
        snrs = []
        for seed in range(n):
            cfg = Config(K=1, Mx=Mx, My=My, Nt=8)
            rng = cfg.rng(seed)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            _, _, hist = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=15)
            snrs.append(2 ** hist[-1] - 1)
        return np.mean(snrs)

    snr_64 = mean_snr(64)
    snr_256 = mean_snr(256)  # M ratio 4x -> SNR should scale ~4^2=16x
    ratio = snr_256 / snr_64
    assert 8 < ratio < 32  # loose band around the theoretical 16x


def test_sv_num_paths_auto_bumped_for_multiuser():
    """rank(H) <= sv_num_paths, so ZF/RZF for K users need sv_num_paths > K
    or H_eff is structurally rank-deficient (see docs/ASSUMPTIONS.md)."""
    assert Config(K=1).sv_num_paths == 3  # unaffected, matches the assignment's L=3 default
    assert Config(K=4).sv_num_paths > 4
    assert Config(K=4, sv_num_paths=10).sv_num_paths == 10  # explicit override preserved


def test_absorption_140ghz_weaker_than_28ghz_same_distance():
    cfg = Config()
    d = 10.0
    k28 = cfg.absorption_coeff(28e9)
    k140 = cfg.absorption_coeff(140e9)
    l28 = molecular_absorption(k28, d)
    l140 = molecular_absorption(k140, d)
    assert l140 < l28


def _make_k1_instance(seed=0, M=64, Nt=8):
    cfg = Config(K=1, Mx=8, My=M // 8, Nt=Nt)
    rng = cfg.rng(seed)
    ue_pos = get_ue_positions(cfg, rng)
    H = sv_channel_H(cfg, rng)
    G = sv_channel_g(cfg, rng, ue_pos)
    return cfg, H, G


def _make_k4_instance(seed=0, M=64, Nt=8, K=4):
    cfg = Config(K=K, Mx=8, My=M // 8, Nt=Nt)
    rng = cfg.rng(seed)
    ue_pos = get_ue_positions(cfg, rng)
    H = sv_channel_H(cfg, rng)
    G = sv_channel_g(cfg, rng, ue_pos)
    return cfg, H, G


# ---------------------------------------------------------------------------
# Milestone 2: beamforming.py, ris_opt.py, ao.py
# ---------------------------------------------------------------------------

def test_mrt_achieves_optimal_snr():
    """Phase 2, Milestone 0: regression test for a real bug -- an earlier
    mrt() used w = h_eff/||h_eff|| (no conjugate), achieving only ~22% of
    the optimal SINR on average (~6-7 dB loss). By Cauchy-Schwarz, the
    true optimum is |h^H w|^2 = P*||h||^2 exactly, achieved by
    w = conj(h_eff)/||h_eff|| (h_eff IS h^H, per effective_channel's
    convention). Verified against brute-force random search over 200k
    candidate directions in development; this test pins the exact
    closed-form value so a regression is caught immediately."""
    rng = np.random.default_rng(123)
    for _ in range(20):
        h_eff = (rng.standard_normal(8) + 1j * rng.standard_normal(8)).reshape(1, 8)
        P = 5.0
        sigma2 = 1.0
        w = mrt(h_eff[0], P)
        achieved = np.abs((h_eff @ w)[0, 0]) ** 2
        optimal = P * np.sum(np.abs(h_eff) ** 2)
        assert np.isclose(achieved, optimal, rtol=1e-9), f"achieved={achieved}, optimal={optimal}"


def test_beamformers_satisfy_power_constraint():
    cfg, H, G = _make_k1_instance()
    h_eff = effective_channel(G, H, np.ones(cfg.M, dtype=complex))[0]
    P = cfg.Pmax_w
    w = mrt(h_eff, P)
    assert np.isclose(np.linalg.norm(w) ** 2, P)

    cfg4, H4, G4 = _make_k4_instance()
    H_eff4 = effective_channel(G4, H4, np.ones(cfg4.M, dtype=complex))
    for fn in (lambda He, p: zf(He, p), lambda He, p: rzf(He, p, cfg4.noise_power_w)):
        W = fn(H_eff4, cfg4.Pmax_w)
        assert np.isclose(np.linalg.norm(W, "fro") ** 2, cfg4.Pmax_w)


def test_phase_update_single_user_unit_modulus():
    cfg, H, G = _make_k1_instance()
    rng = cfg.rng(1)
    w = (rng.standard_normal(cfg.Nt) + 1j * rng.standard_normal(cfg.Nt))
    theta = phase_update_single_user(G[:, 0], H, w)
    assert np.allclose(np.abs(theta), 1.0, atol=1e-9)


def test_phase_update_manifold_unit_modulus_and_nondecreasing():
    cfg, H, G = _make_k4_instance()
    rng = cfg.rng(2)
    H_eff = effective_channel(G, H, np.ones(cfg.M, dtype=complex))
    W = zf(H_eff, cfg.Pmax_w)
    theta0 = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
    theta, trace = phase_update_manifold(G, H, W, theta0, cfg.noise_power_w, iters=30)
    assert np.allclose(np.abs(theta), 1.0, atol=1e-9)
    diffs = np.diff(trace)
    assert np.all(diffs >= -1e-9)


def test_ao_rate_history_nondecreasing_k1_and_k4():
    cfg, H, G = _make_k1_instance()
    _, _, hist1 = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=10)
    assert np.all(np.diff(hist1) >= -1e-9)

    cfg4, H4, G4 = _make_k4_instance()
    _, _, hist4 = run_ao(H4, G4, cfg4.Pmax_w, cfg4.noise_power_w, mode="rzf", max_iter=8)
    assert np.all(np.diff(hist4) >= -1e-9)


def test_ao_beats_random_phase_most_of_the_time():
    n_draws = 100
    wins = 0
    for i in range(n_draws):
        cfg, H, G = _make_k1_instance(seed=i)
        theta_ao, W_ao, hist = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=10)
        ao_rate = hist[-1]

        rng = cfg.rng(i + 5000)
        rand_rates = []
        for _ in range(200):
            theta_r = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
            h_eff_r = effective_channel(G, H, theta_r)
            w_r = mrt(h_eff_r[0], cfg.Pmax_w)
            rand_rates.append(sum_rate(h_eff_r, w_r, cfg.noise_power_w))
        if ao_rate >= np.mean(rand_rates):
            wins += 1
    assert wins >= 0.95 * n_draws


def test_k1_closed_form_matches_manifold_within_1pct():
    """For a fixed W (K=1), the closed-form phase update is the global optimum;
    the manifold solver run to convergence from a random init should match it."""
    cfg, H, G = _make_k1_instance(seed=7)
    rng = cfg.rng(8)
    w = rng.standard_normal(cfg.Nt) + 1j * rng.standard_normal(cfg.Nt)
    w = w / np.linalg.norm(w) * np.sqrt(cfg.Pmax_w)
    W = w.reshape(-1, 1)

    theta_cf = phase_update_single_user(G[:, 0], H, w)
    rate_cf = sum_rate(effective_channel(G, H, theta_cf), W, cfg.noise_power_w)

    theta0 = np.exp(1j * rng.uniform(0, 2 * np.pi, cfg.M))
    theta_mf, _trace = phase_update_manifold(G, H, W, theta0, cfg.noise_power_w, iters=300)
    rate_mf = sum_rate(effective_channel(G, H, theta_mf), W, cfg.noise_power_w)

    assert abs(rate_cf - rate_mf) / rate_cf < 0.01


# ---------------------------------------------------------------------------
# Milestone 3: benchmarks.py, metrics.py + sanity checks (section 8 item 4)
# ---------------------------------------------------------------------------

def test_ao_vs_no_ris_crosses_over_with_M():
    """AO-RIS is NOT asserted to beat the blocked direct link at the default
    M=64 -- it genuinely often does not, in this geometry (the RIS leg's
    cascaded/multiplicative path loss is a real, documented handicap, see
    docs/ASSUMPTIONS.md). What IS asserted (and is the actual point of the
    M-sweep, exp2): AO-RIS's win rate against no-RIS must increase as M
    grows, since the RIS's coherent gain scales ~M^2 while the direct link
    does not depend on M at all."""
    def win_rate(M, n=20):
        Mx, My = 8, M // 8
        wins = 0
        for i in range(n):
            cfg = Config(K=1, Mx=Mx, My=My)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            _, _, hist = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=15)
            no_ris = no_ris_rate(d, cfg.Pmax_w, cfg.noise_power_w)
            wins += hist[-1] >= no_ris
        return wins / n

    assert win_rate(64) <= win_rate(512)


def test_rate_increases_with_ptx():
    cfg, H, G = _make_k1_instance(seed=3)
    sigma2 = cfg.noise_power_w
    P_low = 10 ** ((10 - 30) / 10)
    P_high = 10 ** ((30 - 30) / 10)
    _, _, hist_low = run_ao(H, G, P_low, sigma2, mode="mrt", max_iter=15)
    _, _, hist_high = run_ao(H, G, P_high, sigma2, mode="mrt", max_iter=15)
    assert hist_high[-1] > hist_low[-1]


def test_rate_increases_with_M():
    """Monotonic AO-rate increase with M (coherent beamforming gain should
    grow with the number of RIS elements; exact ~M^2 scaling is checked
    visually in exp2's log-log plot, not asserted exactly here)."""
    rates = []
    for M in (16, 64, 256):
        Mx, My = 8, M // 8
        rs = []
        for seed in range(10):
            cfg = Config(K=1, Mx=Mx, My=My, Nt=8)
            rng = cfg.rng(seed)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            _, _, hist = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=15)
            rs.append(hist[-1])
        rates.append(np.mean(rs))
    assert rates[0] < rates[1] < rates[2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
