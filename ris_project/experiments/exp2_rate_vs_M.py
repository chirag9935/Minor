"""
Experiment 2: Achievable rate vs number of RIS elements M, at Ptx = 30 dBm.

M in {16, 32, 64, 128, 256} (128 uses an 8x16 UPA). Same four curves per K
as exp1. 200 Monte-Carlo channel realisations per M point.
Run: python -m experiments.exp2_rate_vs_M
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate, af_relay_rate
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
PTX_DBM = 30.0
M_CONFIGS = [(16, (4, 4)), (32, (8, 4)), (64, (8, 8)), (128, (8, 16)), (256, (16, 16))]
M_VALUES = np.array([m for m, _ in M_CONFIGS])


def sweep_k1():
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc="exp2 K=1")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=PTX_DBM)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            acc["AO-RIS"].append(hist[-1])
            acc["Random-phase"].append(random_phase_rate(H, G, P, sigma2, "mrt", cfg.rng(i + 10000)))
            acc["No-RIS"].append(no_ris_rate(d, P, sigma2))
            acc["AF relay"].append(af_relay_rate(cfg, cfg.rng(i + 20000), ue_pos, P, sigma2))
        for k in names:
            rates[k][mi] = np.mean(acc[k])
    return rates


def sweep_k4():
    names = ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]
    rates = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc="exp2 K=4")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=4, Mx=Mx, My=My, Pmax_dBm=PTX_DBM)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            _, _, hist_zf = run_ao(H, G, P, sigma2, mode="zf", max_iter=10, manifold_iters=15)
            acc["AO-ZF"].append(hist_zf[-1])
            _, _, hist_rzf = run_ao(H, G, P, sigma2, mode="rzf", max_iter=10, manifold_iters=15)
            acc["AO-RZF"].append(hist_rzf[-1])
            acc["Random-phase"].append(random_phase_rate(H, G, P, sigma2, "rzf", cfg.rng(i + 10000)))
            acc["No-RIS"].append(no_ris_rate(d, P, sigma2))
        for k in names:
            rates[k][mi] = np.mean(acc[k])
    return rates


def main():
    print("Running exp2: rate vs M (K=1) ...")
    k1 = sweep_k1()
    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
        ax.plot(M_VALUES, k1[name], marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, "exp2_rate_vs_M_K1", "Number of RIS elements M", "Sum rate [bit/s/Hz]",
                   title="Rate vs M (K=1, Ptx=30 dBm)")
    save_results("exp2_rate_vs_M_K1", M=M_VALUES, **k1)

    print("Running exp2: rate vs M (K=4) ...")
    k4 = sweep_k4()
    fig, ax = new_fig()
    for name in ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]:
        ax.plot(M_VALUES, k4[name], marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, "exp2_rate_vs_M_K4", "Number of RIS elements M", "Sum rate [bit/s/Hz]",
                   title="Rate vs M (K=4, Ptx=30 dBm)")
    save_results("exp2_rate_vs_M_K4", M=M_VALUES, **k4)
    print("exp2 done.")


if __name__ == "__main__":
    main()
