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
from metrics import effective_channel, sum_rate, energy_efficiency, total_power_w
from ao import run_ao
from benchmarks import no_ris_rate
from objective_b import (
    min_power_single_user, required_power_single_user, zf_min_power_precoder,
    min_power_zf, is_outage,
)
from hardware import quantize_phase, rate_with_quantization
from csi_error import add_csi_error, rzf_robust
from sdr import build_u, actual_min_sinr, sdr_max_min_sinr, manifold_max_min_sinr
from drl_env import RISPhaseEnv, RISJointEnv


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


def test_itu_absorption_model_matches_known_physics():
    """Phase 2, Milestone 3: the default absorption_model="itu" (ITU-R
    P.676 via itur) should reproduce the well-known oxygen/water-vapour
    absorption spectrum shape: a strong oxygen line near 60 GHz (the
    O2 band), attenuation lower in the 70-150 GHz window than at the
    60 GHz peak, and increasing water-vapour absorption with humidity
    at a water-vapour-sensitive frequency (300 GHz) but not at the
    humidity-insensitive oxygen line (60 GHz)."""
    cfg = Config()
    k60 = cfg.absorption_coeff(60e9)
    k100 = cfg.absorption_coeff(100e9)
    assert k60 > k100  # 60 GHz O2 peak >> the window just above it

    cfg_dry = Config(water_vapour_density=2.0)
    cfg_humid = Config(water_vapour_density=15.0)
    assert cfg_humid.absorption_coeff(300e9) > cfg_dry.absorption_coeff(300e9)
    # oxygen line should barely move with humidity (within ~5%)
    k60_dry = cfg_dry.absorption_coeff(60e9)
    k60_humid = cfg_humid.absorption_coeff(60e9)
    assert abs(k60_humid - k60_dry) / k60_dry < 0.05


def test_placeholder_absorption_model_still_reproducible():
    """Phase 1's placeholder table stays available and exact (Phase 2
    Milestone 0 ground rule: keep Phase 1 results reproducible)."""
    cfg = Config(absorption_model="placeholder")
    assert cfg.absorption_coeff(28e9) == 0.1
    assert cfg.absorption_coeff(140e9) == 2.0
    assert cfg.absorption_coeff(300e9) == 8.0


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


def test_energy_efficiency_formula_matches_direct_computation():
    """Required test (Phase 2 spec, Section 10): EE formula. Verifies
    EE = BW*rate/P_total and P_total = Ptx/eta + P_BS_static + M*P_RIS +
    K*P_UE (metrics.total_power_w/energy_efficiency) against a manual,
    from-scratch computation with known inputs -- catches a mismatch
    between the two halves of the power model (e.g. a term dropped or
    double-counted) that a purely qualitative EE-vs-Ptx plot would not."""
    cfg = Config(K=4, Mx=8, My=8, pa_efficiency=0.4, p_bs_static_w=1.0,
                 p_ris_elem_w=5e-3, p_ue_w=0.1, BW=100e6)
    Ptx_w = 10 ** ((20.0 - 30) / 10)  # 20 dBm -> 0.1 W
    rate = 7.3  # bit/s/Hz, an arbitrary fixed value -- only the formula is under test

    expected_P_total = Ptx_w / 0.4 + 1.0 + cfg.M * 5e-3 + cfg.K * 0.1
    expected_EE = cfg.BW * rate / expected_P_total

    P_total = total_power_w(cfg, Ptx_w)
    EE, SE = energy_efficiency(cfg, rate, Ptx_w)

    assert np.isclose(P_total, expected_P_total, rtol=1e-12)
    assert np.isclose(EE, expected_EE, rtol=1e-12)
    assert SE == rate  # SE is just the rate, passed through


# ---------------------------------------------------------------------------
# Phase 2, Milestone 1: objective_b.py (Objective B, min-power)
# ---------------------------------------------------------------------------

def test_objective_b_k1_achieves_target_sinr_exactly():
    cfg, H, G = _make_k1_instance(seed=1)
    gamma = 10 ** (10 / 10)
    sigma2 = cfg.noise_power_w
    theta, W, P_req = min_power_single_user(H, G, gamma, sigma2)
    h_eff = effective_channel(G, H, theta)[0]
    achieved = np.abs(h_eff @ W)[0] ** 2 / sigma2
    assert np.isclose(achieved, gamma, rtol=1e-6)
    assert np.isclose(np.linalg.norm(W) ** 2, P_req)
    # closed-form vs direct computation
    assert np.isclose(P_req, gamma * sigma2 / np.sum(np.abs(h_eff) ** 2))


def test_objective_b_zf_closed_form_matches_direct_computation():
    """Required test (Phase 2 spec): closed-form power vs direct computation."""
    cfg, H, G = _make_k4_instance(seed=2)
    gamma = 10 ** (10 / 10)
    sigma2 = cfg.noise_power_w
    theta0 = np.ones(cfg.M, dtype=complex)
    H_eff = effective_channel(G, H, theta0)

    W, P_closed_form = zf_min_power_precoder(H_eff, gamma, sigma2)

    # direct computation: achieved SINR, ||W||_F^2, and trace((H H^H)^-1) by hand
    received = H_eff @ W
    sinr = np.abs(np.diag(received)) ** 2 / sigma2
    assert np.allclose(sinr, gamma, rtol=1e-6)
    assert np.isclose(np.linalg.norm(W, "fro") ** 2, P_closed_form)

    gram_inv = np.linalg.inv(H_eff @ H_eff.conj().T)
    P_direct = gamma * sigma2 * np.real(np.trace(gram_inv))
    assert np.isclose(P_closed_form, P_direct)


def test_objective_b_zf_power_history_nonincreasing():
    cfg, H, G = _make_k4_instance(seed=3)
    gamma = 10 ** (10 / 10)
    sigma2 = cfg.noise_power_w
    theta, W, P_req, hist = min_power_zf(H, G, gamma, sigma2, max_iter=10)
    assert np.all(np.diff(hist) <= 1e-9)
    H_eff = effective_channel(G, H, theta)
    received = H_eff @ W
    sinr = np.abs(np.diag(received)) ** 2 / sigma2
    assert np.allclose(sinr, gamma, rtol=1e-6)


def test_objective_b_required_power_nonincreasing_in_M():
    """Required sanity check (Phase 2 spec): required power is non-increasing
    in M (more RIS elements can only help reduce the power needed)."""
    cfg, H, G = _make_k1_instance(seed=4, M=16)
    gamma = 10 ** (10 / 10)
    sigma2 = cfg.noise_power_w
    _, _, P16 = min_power_single_user(H, G, gamma, sigma2)

    cfg2, H2, G2 = _make_k1_instance(seed=4, M=64)
    _, _, P64 = min_power_single_user(H2, G2, gamma, sigma2)

    assert P64 <= P16


def test_objective_b_outage_flag():
    assert is_outage(P_required=10.0, Pmax=5.0) is True
    assert is_outage(P_required=3.0, Pmax=5.0) is False


# ---------------------------------------------------------------------------
# Phase 2, Milestone 4: hardware.py (discrete RIS phases)
# ---------------------------------------------------------------------------

def test_quantize_phase_levels_and_unit_modulus():
    rng = np.random.default_rng(0)
    theta = np.exp(1j * rng.uniform(0, 2 * np.pi, 64))
    for bits in (1, 2, 3, 4):
        tq = quantize_phase(theta, bits)
        assert np.allclose(np.abs(tq), 1.0, atol=1e-9)
        levels = 2 ** bits
        grid = np.exp(1j * np.arange(levels) * (2 * np.pi / levels))
        # every quantized value must exactly match one of the 2^bits grid points
        dist_to_grid = np.abs(tq[:, None] - grid[None, :])
        assert np.all(np.min(dist_to_grid, axis=1) < 1e-9)


def test_quantized_rate_bounded_by_continuous_and_improves_with_bits():
    """Required sanity checks (Phase 2 spec): quantised rate <= continuous
    rate (on average), and approaches it as bits increase."""
    cfg, H, G = _make_k1_instance(seed=5)
    sigma2 = cfg.noise_power_w
    theta_c, W_c, _ = run_ao(H, G, cfg.Pmax_w, sigma2, mode="mrt", max_iter=15)
    rate_cont, _ = rate_with_quantization(G, H, W_c, theta_c, None, sigma2)

    rates = []
    for bits in (1, 2, 3, 4):
        r, _ = rate_with_quantization(G, H, W_c, theta_c, bits, sigma2, refine=False)
        assert r <= rate_cont + 1e-9
        rates.append(r)
    # monotonically non-decreasing as bits increase (more levels -> closer to continuous)
    assert np.all(np.diff(rates) >= -1e-9)


def test_quantization_aware_refine_never_hurts():
    cfg, H, G = _make_k1_instance(seed=6)
    theta_c, W_c, _ = run_ao(H, G, cfg.Pmax_w, cfg.noise_power_w, mode="mrt", max_iter=15)
    sigma2 = cfg.noise_power_w
    for bits in (1, 2):
        r_plain, _ = rate_with_quantization(G, H, W_c, theta_c, bits, sigma2, refine=False)
        r_refined, _ = rate_with_quantization(G, H, W_c, theta_c, bits, sigma2, refine=True)
        assert r_refined >= r_plain - 1e-9


# ---------------------------------------------------------------------------
# Phase 2, Milestone 5: csi_error.py
# ---------------------------------------------------------------------------

def test_csi_error_variance_matches_epsilon():
    """Required test (Phase 2 spec): CSI error statistics match epsilon."""
    rng = np.random.default_rng(0)
    H = (rng.standard_normal((64, 8)) + 1j * rng.standard_normal((64, 8)))
    G = (rng.standard_normal((64, 4)) + 1j * rng.standard_normal((64, 4)))
    for epsilon in (0.01, 0.1, 0.5):
        errs_H = []
        for trial in range(300):
            Hhat, _ = add_csi_error(H, G, epsilon, np.random.default_rng(trial))
            errs_H.append(Hhat - H)
        errs_H = np.array(errs_H)
        measured_var = np.var(errs_H)
        expected_var = epsilon * np.mean(np.abs(H) ** 2)
        assert abs(measured_var - expected_var) / expected_var < 0.1

    Hhat0, Ghat0 = add_csi_error(H, G, 0.0, rng)
    assert np.array_equal(Hhat0, H) and np.array_equal(Ghat0, G)


def test_csi_error_seed_gives_reproducible_results():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    H = np.ones((8, 4), dtype=complex)
    G = np.ones((8, 2), dtype=complex)
    Hhat1, Ghat1 = add_csi_error(H, G, 0.1, rng1)
    Hhat2, Ghat2 = add_csi_error(H, G, 0.1, rng2)
    assert np.array_equal(Hhat1, Hhat2) and np.array_equal(Ghat1, Ghat2)


def test_rzf_robust_reduces_to_rzf_at_zero_epsilon():
    cfg, H, G = _make_k4_instance(seed=8)
    rng = cfg.rng(8)
    ue_pos = get_ue_positions(cfg, rng)
    H_eff = effective_channel(G, H, np.ones(cfg.M, dtype=complex))
    W_plain = rzf(H_eff, cfg.Pmax_w, cfg.noise_power_w)
    W_robust = rzf_robust(H_eff, cfg.Pmax_w, cfg.noise_power_w, epsilon=0.0)
    assert np.allclose(W_plain, W_robust, atol=1e-9)


def test_csi_error_degrades_rate_on_average():
    """Required sanity check (Phase 2 spec): rate with CSI error <= rate
    with perfect CSI, on average. Mirrors exp_csi_error.py's own K=1
    sweep (AO run on the noisy ESTIMATE, evaluated against the TRUE
    channel) at a reduced draw count for test speed; epsilon=0.2 is
    large enough that the degradation is well clear of Monte-Carlo noise
    at even this reduced N (confirmed at full scale in
    docs/ASSUMPTIONS.md: eps=0 -> eps=0.5 drops K=1 rate 11.58 -> 11.01,
    K=4 (RZF) far more sharply, 10.29 -> 2.49)."""
    cfg = Config(K=1)
    n_draws = 40
    rates = {0.0: [], 0.2: []}
    for eps in rates:
        for i in range(n_draws):
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            Hhat, Ghat = add_csi_error(H, G, eps, cfg.rng(i + 30000))
            theta, W, _ = run_ao(Hhat, Ghat, P, sigma2, mode="mrt", max_iter=15)
            true_rate = sum_rate(effective_channel(G, H, theta), W, sigma2)
            rates[eps].append(true_rate)

    mean_perfect = np.mean(rates[0.0])
    mean_noisy = np.mean(rates[0.2])
    assert mean_noisy <= mean_perfect


# ---------------------------------------------------------------------------
# Phase 2, Milestone 6(b): sdr.py
# ---------------------------------------------------------------------------

def test_sdr_u_kj_algebra_matches_direct_computation():
    """Required test (Phase 2 spec): verify the u_kj/V SDR algebra numerically
    against a direct computation of h_k^H w_j and |h_k^H w_j|^2."""
    rng = np.random.default_rng(0)
    M, Nt, K = 8, 4, 3
    H = rng.standard_normal((M, Nt)) + 1j * rng.standard_normal((M, Nt))
    G = rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))
    W = rng.standard_normal((Nt, K)) + 1j * rng.standard_normal((Nt, K))
    theta = np.exp(1j * rng.uniform(0, 2 * np.pi, M))

    H_eff = effective_channel(G, H, theta)
    direct = H_eff @ W  # (K,K), entry [k,j] = h_k^H w_j

    u = build_u(G, H, W)
    via_u = np.array([[theta @ u[k, j] for j in range(K)] for k in range(K)])
    assert np.allclose(direct, via_u, atol=1e-9)

    V = np.outer(np.conj(theta), theta)
    via_V = np.array([[np.real(np.trace(np.outer(u[k, j], np.conj(u[k, j])) @ V))
                        for j in range(K)] for k in range(K)])
    assert np.allclose(np.abs(direct) ** 2, via_V, atol=1e-9)
    assert np.allclose(np.diag(V), 1.0)  # unit-modulus theta -> unit diagonal


def test_sdr_feasible_point_passes_its_own_constraints():
    """A known-good V (built from a real theta) must be accepted as feasible
    at (a hair below) its own true min-SINR -- catches sign/direction errors
    in the constraint formulation that the algebra test alone would miss."""
    from sdr import _feasible, _build_feasibility_problem
    cfg, H, G = _make_k4_instance(seed=1, M=16)
    rng = cfg.rng(1)
    ue_pos = get_ue_positions(cfg, rng)
    H_eff0 = effective_channel(G, H, np.ones(cfg.M, dtype=complex))
    W = rzf(H_eff0, cfg.Pmax_w, cfg.noise_power_w)
    u = build_u(G, H, W)
    theta0 = np.ones(cfg.M, dtype=complex)
    true_sinr = actual_min_sinr(theta0, u, cfg.noise_power_w)

    scale = 1.0 / np.sqrt(cfg.noise_power_w)
    prob, gamma_param, V_var = _build_feasibility_problem(u * scale, 1.0, cfg.M)
    feasible, _ = _feasible(true_sinr * 0.999, prob, gamma_param, V_var, "CLARABEL")
    assert feasible


def test_manifold_matches_or_beats_sdr_on_max_min_sinr():
    """Phase 2 spec: compare SDR against manifold optimization on the same
    max-min objective. Manifold (direct, non-relaxed optimization) should
    not do worse than SDR+randomization -- documented finding (see
    docs/ASSUMPTIONS.md): SDR's relaxation bound is correct (independently
    confirmed: the manifold result lands inside the SDR-bisected feasible
    region) but Gaussian randomization recovers a visibly looser solution
    in practice for this problem."""
    cfg, H, G = _make_k4_instance(seed=2, M=16)
    rng = cfg.rng(2)
    ue_pos = get_ue_positions(cfg, rng)
    H_eff0 = effective_channel(G, H, np.ones(cfg.M, dtype=complex))
    W = rzf(H_eff0, cfg.Pmax_w, cfg.noise_power_w)
    sigma2 = cfg.noise_power_w

    theta0 = np.exp(1j * cfg.rng(3).uniform(0, 2 * np.pi, cfg.M))
    theta_m, sinr_m, _ = manifold_max_min_sinr(G, H, W, sigma2, theta0, iters=100)
    _, sinr_sdr, _ = sdr_max_min_sinr(G, H, W, sigma2, n_randomizations=200, rng=cfg.rng(4))
    assert sinr_m >= sinr_sdr - 1e-6


# ---------------------------------------------------------------------------
# Milestone 7: DRL environments (drl_env.py)
# ---------------------------------------------------------------------------

def test_drl_phase_env_reward_matches_rate_function():
    """step()'s reward (= rate / ao_rate) must match an INDEPENDENTLY
    computed rate for the same theta/W via metrics.sum_rate -- catches a
    mismatch between the env's internal W-reconstruction rule and the
    rate it reports, which the spec explicitly asks to be tested."""
    cfg = Config(K=1, Mx=4, My=4)  # M=16
    env = RISPhaseEnv(cfg, seed=0)
    obs, info = env.reset(seed=0)
    ao_rate = info["ao_rate"]

    action = np.asarray([0.3] * cfg.M)  # arbitrary fixed, non-zero action
    _, reward, terminated, truncated, step_info = env.step(action)
    assert terminated and not truncated

    theta = np.exp(1j * action)
    W = mrt(effective_channel(env.G, env.H, theta)[0], cfg.Pmax_w)
    expected_rate = sum_rate(effective_channel(env.G, env.H, theta), W, env.sigma2)

    assert np.isclose(step_info["raw_rate"], expected_rate, rtol=1e-9)
    assert np.isclose(reward, expected_rate / ao_rate, rtol=1e-9)
    assert np.isclose(step_info["ao_rate"], ao_rate)


def test_drl_joint_env_reward_matches_rate_function():
    """Same check for the joint (phases + beamformer) env, including the
    ||W||_F^2 = Pmax projection."""
    cfg = Config(K=2, Mx=4, My=4)
    env = RISJointEnv(cfg, seed=0)
    _, info = env.reset(seed=0)
    ao_rate = info["ao_rate"]

    action = env.action_space.sample()
    _, reward, terminated, _, step_info = env.step(action)
    assert terminated

    action = np.asarray(action, dtype=np.float64)  # match step()'s internal precision
    theta = np.exp(1j * action[: cfg.M] * np.pi)
    half = cfg.Nt * cfg.K
    w_raw = action[cfg.M:]
    W = w_raw[:half].reshape(cfg.Nt, cfg.K) + 1j * w_raw[half:].reshape(cfg.Nt, cfg.K)
    W = W * (np.sqrt(cfg.Pmax_w) / np.linalg.norm(W, "fro"))
    expected_rate = sum_rate(effective_channel(env.G, env.H, theta), W, env.sigma2)

    assert np.isclose(step_info["raw_rate"], expected_rate, rtol=1e-9)
    assert np.isclose(reward, expected_rate / ao_rate, rtol=1e-9)
    # the projected W must actually satisfy the power budget
    assert np.isclose(np.linalg.norm(W, "fro") ** 2, cfg.Pmax_w, rtol=1e-9)


def test_drl_env_seed_gives_reproducible_results():
    """Two freshly constructed envs, same seed, same reset seed -> bit-
    identical channel draws (observation, ao_rate) and identical rewards
    for the same action -- the spec's explicit reproducibility test."""
    cfg = Config(K=1, Mx=4, My=4)
    env_a = RISPhaseEnv(cfg, seed=7)
    env_b = RISPhaseEnv(cfg, seed=7)
    obs_a, info_a = env_a.reset(seed=0)
    obs_b, info_b = env_b.reset(seed=0)
    assert np.allclose(obs_a, obs_b)
    assert info_a["ao_rate"] == info_b["ao_rate"]

    action = np.asarray([0.1] * cfg.M)
    _, reward_a, *_ = env_a.step(action)
    _, reward_b, *_ = env_b.step(action)
    assert reward_a == reward_b


def test_drl_env_action_and_observation_spaces_are_well_formed():
    """Action space bounds match the spec ([-pi, pi] for phases); the
    observation space dimensionality matches 2*K*M; a sampled action is
    always accepted by step() without shape errors."""
    cfg = Config(K=1, Mx=4, My=4)
    env = RISPhaseEnv(cfg, seed=0)
    env.reset(seed=0)
    assert env.action_space.shape == (cfg.M,)
    assert np.isclose(env.action_space.low[0], -np.pi) and np.isclose(env.action_space.high[0], np.pi)
    assert env.observation_space.shape == (2 * cfg.K * cfg.M,)
    env.step(env.action_space.sample())  # must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
