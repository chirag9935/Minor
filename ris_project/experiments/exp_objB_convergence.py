"""
Phase 2, Milestone 1: Objective B convergence. Required power vs AO
iteration for a single representative channel realisation, from 5 random
RIS-phase initialisations (light lines) plus their mean (bold), K=4 (ZF,
min-power AO). Mirrors exp3_convergence.py's style for Objective A.
Run: python -m experiments.exp_objB_convergence
"""
import numpy as np

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from objective_b import min_power_zf
from experiments._common import new_fig, style_and_save, save_results

N_INIT = 5
GAMMA_DB = 10.0


def main():
    print("Running exp_objB_convergence: power vs AO iteration (K=4, ZF) ...")
    cfg = Config(K=4)
    gamma = 10 ** (GAMMA_DB / 10)
    rng = cfg.rng(0)
    ue_pos = get_ue_positions(cfg, rng)
    H = sv_channel_H(cfg, rng)
    G = sv_channel_g(cfg, rng, ue_pos)
    sigma2 = cfg.noise_power_w

    curves = []
    for i in range(N_INIT):
        rng_i = cfg.rng(100 + i)
        theta0 = np.exp(1j * rng_i.uniform(0, 2 * np.pi, cfg.M))
        _, _, _, hist = min_power_zf(H, G, gamma, sigma2, theta0=theta0, max_iter=15, manifold_iters=30)
        hist = np.array(hist[1::2])  # one point per full (W, theta) update pair, like exp3
        curves.append(hist)

    min_len = min(len(c) for c in curves)
    curves = np.stack([c[:min_len] for c in curves])
    iters = np.arange(1, min_len + 1)

    fig, ax = new_fig()
    for c in curves:
        ax.plot(iters, 10 * np.log10(c) + 30, color="#9467bd", alpha=0.3, linewidth=1)
    ax.plot(iters, 10 * np.log10(curves.mean(axis=0)) + 30, color="#9467bd", linewidth=2.5, label="mean (5 inits)")
    style_and_save(fig, ax, "exp_objB_convergence", "AO iteration", "Required $P_{tx}$ [dBm]",
                   title=f"Objective B convergence (K=4, ZF, target SINR={GAMMA_DB:.0f} dB)")
    save_results("exp_objB_convergence", iters=iters, curves=curves)
    print("exp_objB_convergence done.")


if __name__ == "__main__":
    main()
