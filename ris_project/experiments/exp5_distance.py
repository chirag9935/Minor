"""
Experiment 5: Distance sensitivity (K=1).

Panel (a): UE-to-RIS distance swept 2..30 m (RIS-to-BS geometry fixed at the
default). Panel (b): RIS-to-BS distance swept 10..40 m (RIS-to-UE distance
held fixed at 5 m). AO-RIS, Random-phase, No-RIS, AF relay curves.

Assumption: for a clean, monotone distance sweep the UE is placed at the
*same height* as the RIS in this experiment only (so horizontal distance
equals 3D distance); elsewhere the UE height is offset from the RIS for a
more realistic default 3-8 m placement (see config.py / docs/ASSUMPTIONS.md).
200 Monte-Carlo channel realisations per distance point.
Run: python -m experiments.exp5_distance
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate, af_relay_rate
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
D_UE_RIS = np.array([2, 5, 10, 15, 20, 25, 30], dtype=float)
D_RIS_BS = np.array([10, 15, 20, 25, 30, 35, 40], dtype=float)
SCHEMES = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]


def _sweep(make_cfg_and_pos):
    rates = {k: [] for k in SCHEMES}
    for i in range(N_DRAWS):
        cfg, ue_pos = make_cfg_and_pos(i)
        rng = cfg.rng(i)
        H = sv_channel_H(cfg, rng)
        G = sv_channel_g(cfg, rng, ue_pos)
        d = direct_channel(cfg, rng, ue_pos)
        P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

        _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
        rates["AO-RIS"].append(hist[-1])
        rates["Random-phase"].append(random_phase_rate(H, G, P, sigma2, "mrt", cfg.rng(i + 10000)))
        rates["No-RIS"].append(no_ris_rate(d, P, sigma2))
        rates["AF relay"].append(af_relay_rate(cfg, cfg.rng(i + 20000), ue_pos, P, sigma2))
    return {k: float(np.mean(v)) for k, v in rates.items()}


def sweep_ue_ris_distance():
    out = {k: np.zeros(len(D_UE_RIS)) for k in SCHEMES}
    for di, dist in enumerate(tqdm(D_UE_RIS, desc="exp5 UE-RIS distance")):
        def make(i, dist=dist):
            cfg = Config(K=1, ue_height=10.0)  # match RIS height -> horizontal dist = 3D dist
            ue_pos = np.array([[cfg.ris_pos[0] + dist, cfg.ris_pos[1], cfg.ris_pos[2]]])
            return cfg, ue_pos
        means = _sweep(make)
        for k in SCHEMES:
            out[k][di] = means[k]
    return out


def sweep_ris_bs_distance():
    out = {k: np.zeros(len(D_RIS_BS)) for k in SCHEMES}
    for di, d_rb in enumerate(tqdm(D_RIS_BS, desc="exp5 RIS-BS distance")):
        def make(i, d_rb=d_rb):
            cfg = Config(K=1, ris_pos=(d_rb, 0.0, 10.0), ue_height=10.0)
            ue_pos = np.array([[d_rb + 5.0, 0.0, 10.0]])  # UE fixed 5 m from RIS
            return cfg, ue_pos
        means = _sweep(make)
        for k in SCHEMES:
            out[k][di] = means[k]
    return out


def main():
    print("Running exp5: UE-RIS distance sweep ...")
    a = sweep_ue_ris_distance()
    fig, ax = new_fig()
    for name in SCHEMES:
        ax.plot(D_UE_RIS, a[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp5_distance_ue_ris", "RIS-UE distance [m]", "Sum rate [bit/s/Hz]",
                   title="Rate vs RIS-UE distance (K=1, Ptx=30 dBm)")
    save_results("exp5_distance_ue_ris", distance=D_UE_RIS, **a)

    print("Running exp5: RIS-BS distance sweep ...")
    b = sweep_ris_bs_distance()
    fig, ax = new_fig()
    for name in SCHEMES:
        ax.plot(D_RIS_BS, b[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp5_distance_ris_bs", "BS-RIS distance [m]", "Sum rate [bit/s/Hz]",
                   title="Rate vs BS-RIS distance (K=1, Ptx=30 dBm, RIS-UE=5 m)")
    save_results("exp5_distance_ris_bs", distance=D_RIS_BS, **b)
    print("exp5 done.")


if __name__ == "__main__":
    main()
