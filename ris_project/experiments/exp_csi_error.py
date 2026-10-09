"""
Phase 2, Milestone 5: imperfect CSI.

Rate vs normalised CSI error epsilon in {0, 0.01, 0.05, 0.1, 0.2, 0.5}
(K=1 and K=4), AO run using the noisy channel ESTIMATES only, evaluated
against the TRUE channels. AO-RIS (plain) and AF relay (K=1); AO-RZF
(plain) vs AO-RZF ("robust", regulariser inflated by the estimated error
power) for K=4. Perfect-CSI AO plotted as a horizontal reference line.
Also: rate vs M at a fixed epsilon, showing how CSI error erodes the M^2
gain established in Milestone 0.
Run: python -m experiments.exp_csi_error
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from ao import run_ao
from benchmarks import af_relay_rate, random_phase_rate
from metrics import effective_channel, sum_rate
from csi_error import add_csi_error
from experiments._common import new_fig, style_and_save, save_results, COLORS

N_DRAWS = 200
EPSILON_VALUES = [0, 0.01, 0.05, 0.1, 0.2, 0.5]
M_FIXED_EPS = [0.1]


def sweep_k1_vs_epsilon():
    names = ["AO-RIS", "Random-phase", "AF relay"]
    rates = {k: np.zeros(len(EPSILON_VALUES)) for k in names}
    for ei, eps in enumerate(tqdm(EPSILON_VALUES, desc="exp_csi K=1")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            Hhat, Ghat = add_csi_error(H, G, eps, cfg.rng(i + 30000))
            theta, W, _ = run_ao(Hhat, Ghat, P, sigma2, mode="mrt", max_iter=15)
            true_rate = sum_rate(effective_channel(G, H, theta), W, sigma2)
            acc["AO-RIS"].append(true_rate)

            acc["Random-phase"].append(random_phase_rate(H, G, P, sigma2, "mrt", cfg.rng(i + 10000)))
            acc["AF relay"].append(af_relay_rate(cfg, cfg.rng(i + 20000), ue_pos, P, sigma2))
        for k in names:
            rates[k][ei] = np.mean(acc[k])
    return rates


def sweep_k4_vs_epsilon():
    names = ["AO-RZF (plain)", "AO-RZF (robust)", "Random-phase"]
    colors = {"AO-RZF (plain)": "#9467bd", "AO-RZF (robust)": "#1f77b4", "Random-phase": "#ff7f0e"}
    rates = {k: np.zeros(len(EPSILON_VALUES)) for k in names}
    for ei, eps in enumerate(tqdm(EPSILON_VALUES, desc="exp_csi K=4")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=4)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            Hhat, Ghat = add_csi_error(H, G, eps, cfg.rng(i + 30000))

            theta_p, W_p, _ = run_ao(Hhat, Ghat, P, sigma2, mode="rzf", max_iter=10, manifold_iters=15)
            acc["AO-RZF (plain)"].append(sum_rate(effective_channel(G, H, theta_p), W_p, sigma2))

            theta_r, W_r, _ = run_ao(Hhat, Ghat, P, sigma2, mode="rzf_robust", max_iter=10,
                                      manifold_iters=15, epsilon=eps)
            acc["AO-RZF (robust)"].append(sum_rate(effective_channel(G, H, theta_r), W_r, sigma2))

            acc["Random-phase"].append(random_phase_rate(H, G, P, sigma2, "rzf", cfg.rng(i + 10000)))
        for k in names:
            rates[k][ei] = np.mean(acc[k])
    return rates, colors


def sweep_rate_vs_M_fixed_eps(eps):
    M_CONFIGS = [(16, (4, 4)), (32, (8, 4)), (64, (8, 8)), (128, (8, 16)), (256, (16, 16))]
    M_VALUES = np.array([m for m, _ in M_CONFIGS])
    rate_perfect = np.zeros(len(M_CONFIGS))
    rate_noisy = np.zeros(len(M_CONFIGS))
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc=f"exp_csi rate-vs-M (eps={eps})")):
        acc_perfect, acc_noisy = [], []
        for i in range(N_DRAWS):
            cfg = Config(K=1, Mx=Mx, My=My)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            acc_perfect.append(hist[-1])

            Hhat, Ghat = add_csi_error(H, G, eps, cfg.rng(i + 30000))
            theta, W, _ = run_ao(Hhat, Ghat, P, sigma2, mode="mrt", max_iter=15)
            acc_noisy.append(sum_rate(effective_channel(G, H, theta), W, sigma2))
        rate_perfect[mi] = np.mean(acc_perfect)
        rate_noisy[mi] = np.mean(acc_noisy)
    return M_VALUES, rate_perfect, rate_noisy


def main():
    print("Running exp_csi_error: K=1 rate vs epsilon ...")
    k1 = sweep_k1_vs_epsilon()
    perfect_k1 = k1["AO-RIS"][0]  # epsilon=0 IS perfect CSI

    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "AF relay"]:
        ax.plot(EPSILON_VALUES, k1[name], marker="o", color=COLORS.get(name, None), label=name)
    style_and_save(fig, ax, "exp_csi_error_vs_epsilon_K1", "Normalised CSI error $\\epsilon$",
                   "Sum rate [bit/s/Hz]", title="Rate vs CSI error (K=1, M=64, Ptx=30 dBm)")
    save_results("exp_csi_error_vs_epsilon_K1", epsilon=np.array(EPSILON_VALUES), **k1)

    print("Running exp_csi_error: K=4 rate vs epsilon (plain vs robust RZF) ...")
    k4, colors4 = sweep_k4_vs_epsilon()
    fig, ax = new_fig()
    for name in ["AO-RZF (plain)", "AO-RZF (robust)", "Random-phase"]:
        ax.plot(EPSILON_VALUES, k4[name], marker="o", color=colors4[name], label=name)
    style_and_save(fig, ax, "exp_csi_error_vs_epsilon_K4", "Normalised CSI error $\\epsilon$",
                   "Sum rate [bit/s/Hz]", title="Rate vs CSI error (K=4, M=64, Ptx=30 dBm)")
    save_results("exp_csi_error_vs_epsilon_K4", epsilon=np.array(EPSILON_VALUES), **k4)

    for eps in M_FIXED_EPS:
        print(f"Running exp_csi_error: rate vs M at epsilon={eps} ...")
        M_VALUES, rate_perfect, rate_noisy = sweep_rate_vs_M_fixed_eps(eps)
        fig, ax = new_fig()
        ax.plot(M_VALUES, rate_perfect, marker="o", color="#2ca02c", label="Perfect CSI")
        ax.plot(M_VALUES, rate_noisy, marker="s", color="#d62728", label=f"CSI error $\\epsilon$={eps}")
        ax.set_xscale("log", base=2)
        style_and_save(fig, ax, f"exp_csi_error_rate_vs_M_eps{eps}", "Number of RIS elements M",
                       "Sum rate [bit/s/Hz]", title=f"How CSI error erodes the M^2 gain (K=1, epsilon={eps})")
        save_results(f"exp_csi_error_rate_vs_M_eps{eps}", M=M_VALUES, rate_perfect=rate_perfect,
                     rate_noisy=rate_noisy)

    print("\nSanity check -- rate with CSI error should be <= perfect-CSI rate (on average):")
    for ei, eps in enumerate(EPSILON_VALUES):
        print(f"  eps={eps}: K=1 rate={k1['AO-RIS'][ei]:.3f} (perfect={perfect_k1:.3f}), "
              f"ok={k1['AO-RIS'][ei] <= perfect_k1 + 1e-6}")
    print("exp_csi_error done.")


if __name__ == "__main__":
    main()
