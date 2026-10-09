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


def _make_cfg(preset: str, **kwargs) -> Config:
    """preset in {"default", "realistic"} -- see Config.realistic(). Phase 2 Milestone 0(e)."""
    return Config.realistic(**kwargs) if preset == "realistic" else Config(**kwargs)


def sweep_k1(preset: str = "default"):
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc=f"exp2 K=1 ({preset})")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = _make_cfg(preset, K=1, Mx=Mx, My=My, Pmax_dBm=PTX_DBM)
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


def sweep_k4(preset: str = "default"):
    names = ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]
    rates = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc=f"exp2 K=4 ({preset})")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = _make_cfg(preset, K=4, Mx=Mx, My=My, Pmax_dBm=PTX_DBM)
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


def main(preset: str = "default"):
    """preset in {"default", "realistic"}; see exp1_rate_vs_power.main() for
    the same convention. Output files suffixed "_realistic" for that preset."""
    suffix = "" if preset == "default" else f"_{preset}"
    title_suffix = "" if preset == "default" else f", {preset} gains"

    print(f"Running exp2: rate vs M (K=1, {preset}) ...")
    k1 = sweep_k1(preset)
    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
        ax.plot(M_VALUES, k1[name], marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, f"exp2_rate_vs_M_K1{suffix}", "Number of RIS elements M", "Sum rate [bit/s/Hz]",
                   title=f"Rate vs M (K=1, Ptx=30 dBm{title_suffix})")
    save_results(f"exp2_rate_vs_M_K1{suffix}", M=M_VALUES, **k1)

    print(f"Running exp2: rate vs M (K=4, {preset}) ...")
    k4 = sweep_k4(preset)
    fig, ax = new_fig()
    for name in ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]:
        ax.plot(M_VALUES, k4[name], marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, f"exp2_rate_vs_M_K4{suffix}", "Number of RIS elements M", "Sum rate [bit/s/Hz]",
                   title=f"Rate vs M (K=4, Ptx=30 dBm{title_suffix})")
    save_results(f"exp2_rate_vs_M_K4{suffix}", M=M_VALUES, **k4)
    print(f"exp2 ({preset}) done.")


if __name__ == "__main__":
    main()
