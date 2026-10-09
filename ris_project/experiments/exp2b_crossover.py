"""
Experiment 2b (BONUS, not required by the assignment): extends exp2's
M-sweep for K=1 beyond the assignment's own M=16..256 range, out to
M=1024. Originally added (pre-Phase-2-fixes) to find the AO-RIS/no-RIS
crossover point, which at the time sat beyond M=256. After Phase 2
Milestone 0's blockage (80 dB default) and MRT-precoder fixes (see
docs/ASSUMPTIONS.md), AO-RIS already beats no-RIS well within the
assignment's own M=16..256 range -- this script's remaining purpose is
to confirm the ~M^2 growth trend keeps holding, and quantify how much
further the margin grows, beyond the assignment's own M cap.

K=1 only: the closed-form phase update has no manifold-optimization cost,
so M=1024 is cheap (no torch), unlike the K=4 manifold solver.
Run: python -m experiments.exp2b_crossover
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
M_CONFIGS = [(16, (4, 4)), (32, (8, 4)), (64, (8, 8)), (128, (8, 16)), (256, (16, 16)),
             (512, (16, 32)), (1024, (32, 32))]
M_VALUES = np.array([m for m, _ in M_CONFIGS])


def _make_cfg(preset: str, **kwargs) -> Config:
    return Config.realistic(**kwargs) if preset == "realistic" else Config(**kwargs)


def sweep_k1(preset: str = "default"):
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc=f"exp2b K=1 ({preset})")):
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


def main(preset: str = "default"):
    suffix = "" if preset == "default" else f"_{preset}"
    print(f"Running exp2b: extended M-sweep, K=1, M=16..1024, {preset} ...")
    k1 = sweep_k1(preset)

    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
        ax.plot(M_VALUES, k1[name], marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, f"exp2b_crossover_K1{suffix}", "Number of RIS elements M", "Sum rate [bit/s/Hz]",
                   title=f"Extended M-sweep (bonus; K=1, Ptx=30 dBm{'' if preset=='default' else ', '+preset})")
    save_results(f"exp2b_crossover_K1{suffix}", M=M_VALUES, **k1)

    for M, ao, noris in zip(M_VALUES, k1["AO-RIS"], k1["No-RIS"]):
        print(f"  M={M:5d}: AO-RIS={ao:7.3f}  No-RIS={noris:7.3f}  {'<-- AO-RIS wins' if ao >= noris else ''}")
    print(f"exp2b ({preset}) done.")


if __name__ == "__main__":
    main()
