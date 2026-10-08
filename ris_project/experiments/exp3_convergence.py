"""
Experiment 3: AO convergence. Sum rate vs AO iteration, for a single
representative channel realisation, from 5 random RIS-phase initialisations
(light lines) plus their mean (bold). One panel for K=1 (MRT + closed-form
phase update), one for K=4 (RZF + manifold phase update).
Run: python -m experiments.exp3_convergence
"""
import numpy as np

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from ao import run_ao
from experiments._common import new_fig, style_and_save, save_results

N_INIT = 5


def per_iteration_rates(H, G, P, sigma2, mode, theta0, max_iter, manifold_iters=50):
    # tol=-1 disables early stopping so every curve has the same length.
    _, _, hist = run_ao(H, G, P, sigma2, mode, theta0=theta0, max_iter=max_iter, tol=-1.0,
                         manifold_iters=manifold_iters)
    return np.array(hist[1::2])  # one point per full (W, theta) update pair


def run_panel(cfg: Config, mode: str, max_iter: int, manifold_iters: int = 50):
    rng = cfg.rng(0)
    ue_pos = get_ue_positions(cfg, rng)
    H = sv_channel_H(cfg, rng)
    G = sv_channel_g(cfg, rng, ue_pos)
    P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

    curves = []
    for i in range(N_INIT):
        rng_i = cfg.rng(100 + i)
        theta0 = np.exp(1j * rng_i.uniform(0, 2 * np.pi, cfg.M))
        curves.append(per_iteration_rates(H, G, P, sigma2, mode, theta0, max_iter, manifold_iters))
    curves = np.stack(curves)
    return curves


def main():
    print("Running exp3: AO convergence (K=1) ...")
    cfg1 = Config(K=1)
    curves1 = run_panel(cfg1, mode="mrt", max_iter=15)
    iters1 = np.arange(1, curves1.shape[1] + 1)

    fig, ax = new_fig()
    for c in curves1:
        ax.plot(iters1, c, color="#1f77b4", alpha=0.3, linewidth=1)
    ax.plot(iters1, curves1.mean(axis=0), color="#1f77b4", linewidth=2.5, label="mean (5 inits)")
    style_and_save(fig, ax, "exp3_convergence_K1", "AO iteration", "Sum rate [bit/s/Hz]",
                   title="AO convergence (K=1, MRT)")
    save_results("exp3_convergence_K1", iters=iters1, curves=curves1)

    print("Running exp3: AO convergence (K=4) ...")
    cfg4 = Config(K=4)
    curves4 = run_panel(cfg4, mode="rzf", max_iter=15, manifold_iters=30)
    iters4 = np.arange(1, curves4.shape[1] + 1)

    fig, ax = new_fig()
    for c in curves4:
        ax.plot(iters4, c, color="#9467bd", alpha=0.3, linewidth=1)
    ax.plot(iters4, curves4.mean(axis=0), color="#9467bd", linewidth=2.5, label="mean (5 inits)")
    style_and_save(fig, ax, "exp3_convergence_K4", "AO iteration", "Sum rate [bit/s/Hz]",
                   title="AO convergence (K=4, RZF + manifold)")
    save_results("exp3_convergence_K4", iters=iters4, curves=curves4)
    print("exp3 done.")


if __name__ == "__main__":
    main()
