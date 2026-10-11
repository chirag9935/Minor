"""
Phase 2, Milestone 6(b,c): SDR vs manifold optimization for the max-min-SINR
RIS phase problem (W fixed), at M in {16, 32, 64} (K=4). Per the spec: do
NOT run SDR above M=64 (the SDP scales badly -- confirmed in development,
see docs/ASSUMPTIONS.md for timing). Random-phase (best of 200 draws) is
included as a third reference point.

Finding (see docs/ASSUMPTIONS.md for the full writeup and verification):
the SDR relaxation bound itself is correct (independently cross-checked
against the manifold result and against direct constraint evaluation),
but Gaussian randomization recovers a visibly looser solution in practice
-- manifold optimization (operating on the true, non-relaxed objective)
matches or beats SDR+randomization at every tested M, while also being
much faster.
Run: python -m experiments.exp_sdr_vs_manifold
"""
import time

import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from beamforming import rzf
from metrics import effective_channel
from sdr import sdr_max_min_sinr, manifold_max_min_sinr, build_u, actual_min_sinr
from experiments._common import new_fig, style_and_save, save_results

N_DRAWS = 20  # SDP solves are expensive; a smaller Monte-Carlo count is a deliberate, documented trade-off
M_CONFIGS = [(16, (4, 4)), (32, (8, 4)), (64, (8, 8))]


def sweep():
    M_values = np.array([m for m, _ in M_CONFIGS])
    sinr_sdr = np.zeros(len(M_CONFIGS))
    sinr_manifold = np.zeros(len(M_CONFIGS))
    sinr_random = np.zeros(len(M_CONFIGS))
    time_sdr = np.zeros(len(M_CONFIGS))
    time_manifold = np.zeros(len(M_CONFIGS))

    for mi, (M, (Mx, My)) in enumerate(M_CONFIGS):
        acc_sdr, acc_manifold, acc_random = [], [], []
        t_sdr, t_manifold = [], []
        for i in tqdm(range(N_DRAWS), desc=f"exp_sdr_vs_manifold M={M}"):
            cfg = Config(K=4, Mx=Mx, My=My)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            sigma2 = cfg.noise_power_w

            theta0 = np.ones(cfg.M, dtype=complex)
            H_eff0 = effective_channel(G, H, theta0)
            W = rzf(H_eff0, cfg.Pmax_w, sigma2)
            u = build_u(G, H, W)

            _, s_sdr, el_sdr = sdr_max_min_sinr(G, H, W, sigma2, n_randomizations=200,
                                                 rng=cfg.rng(i + 1000))
            acc_sdr.append(s_sdr)
            t_sdr.append(el_sdr)

            theta_init = np.exp(1j * cfg.rng(i + 2000).uniform(0, 2 * np.pi, cfg.M))
            t0 = time.time()
            _, s_manifold, el_manifold = manifold_max_min_sinr(G, H, W, sigma2, theta_init, iters=100)
            acc_manifold.append(s_manifold)
            t_manifold.append(el_manifold)

            rng_r = cfg.rng(i + 3000)
            best_r = -1.0
            for _ in range(200):
                tr = np.exp(1j * rng_r.uniform(0, 2 * np.pi, cfg.M))
                best_r = max(best_r, actual_min_sinr(tr, u, sigma2))
            acc_random.append(best_r)

        sinr_sdr[mi] = np.mean(acc_sdr)
        sinr_manifold[mi] = np.mean(acc_manifold)
        sinr_random[mi] = np.mean(acc_random)
        time_sdr[mi] = np.mean(t_sdr)
        time_manifold[mi] = np.mean(t_manifold)

    return M_values, sinr_sdr, sinr_manifold, sinr_random, time_sdr, time_manifold


def main():
    M_values, sinr_sdr, sinr_manifold, sinr_random, time_sdr, time_manifold = sweep()

    fig, ax = new_fig()
    ax.plot(M_values, 10 * np.log10(sinr_sdr), marker="o", color="#d62728", label="SDR + randomization")
    ax.plot(M_values, 10 * np.log10(sinr_manifold), marker="s", color="#1f77b4", label="Manifold optimization")
    ax.plot(M_values, 10 * np.log10(sinr_random), marker="^", color="#ff7f0e", label="Random phase (best of 200)")
    style_and_save(fig, ax, "exp_sdr_vs_manifold_sinr", "Number of RIS elements M",
                   "Max-min SINR [dB]", title=f"SDR vs manifold optimization (K=4, {N_DRAWS} draws)")

    fig, ax = new_fig()
    ax.plot(M_values, time_sdr, marker="o", color="#d62728", label="SDR + randomization")
    ax.plot(M_values, time_manifold, marker="s", color="#1f77b4", label="Manifold optimization")
    ax.set_yscale("log")
    style_and_save(fig, ax, "exp_sdr_vs_manifold_runtime", "Number of RIS elements M",
                   "Run time [s]", title="SDR vs manifold optimization: run time")

    save_results("exp_sdr_vs_manifold", M=M_values, sinr_sdr=sinr_sdr, sinr_manifold=sinr_manifold,
                 sinr_random=sinr_random, time_sdr=time_sdr, time_manifold=time_manifold)

    print(f"\n{'M':>5} {'SDR SINR':>10} {'Manifold SINR':>15} {'Random SINR':>13} "
          f"{'SDR time[s]':>12} {'Manifold time[s]':>17}")
    for mi, M in enumerate(M_values):
        print(f"{M:5d} {sinr_sdr[mi]:10.4f} {sinr_manifold[mi]:15.4f} {sinr_random[mi]:13.4f} "
              f"{time_sdr[mi]:12.3f} {time_manifold[mi]:17.3f}")
    print("\nNote: SDR is not run above M=64 (scales badly) -- see docs/ASSUMPTIONS.md.")
    print("exp_sdr_vs_manifold done.")


if __name__ == "__main__":
    main()
