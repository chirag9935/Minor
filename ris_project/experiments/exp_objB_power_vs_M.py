"""
Phase 2, Milestone 1: Objective B. Required BS transmit power (dBm) vs
number of RIS elements M (16..256, the assignment's own range), K=1,
fixed target SINR gamma=10 dB. AO-RIS only (random-phase/no-RIS/AF relay
don't depend on M the same way -- no-RIS and AF relay are M-independent by
construction, random-phase is shown for reference). 200 Monte-Carlo draws
per M point. Sanity check (Phase 2 spec): required power must be
non-increasing in M.
Run: python -m experiments.exp_objB_power_vs_M
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from metrics import effective_channel
from objective_b import min_power_single_user, required_power_single_user
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
GAMMA_DB = 10.0
M_CONFIGS = [(16, (4, 4)), (32, (8, 4)), (64, (8, 8)), (128, (8, 16)), (256, (16, 16))]
M_VALUES = np.array([m for m, _ in M_CONFIGS])


def to_dbm(p_watts):
    return 10 * np.log10(p_watts) + 30


def sweep_k1():
    gamma = 10 ** (GAMMA_DB / 10)
    names = ["AO-RIS", "Random-phase"]
    power_w = {k: np.zeros(len(M_CONFIGS)) for k in names}
    for mi, (M, (Mx, My)) in enumerate(tqdm(M_CONFIGS, desc="exp_objB power vs M")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1, Mx=Mx, My=My)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            sigma2 = cfg.noise_power_w

            _, _, P_ao = min_power_single_user(H, G, gamma, sigma2)
            acc["AO-RIS"].append(P_ao)

            rng_r = cfg.rng(i + 10000)
            theta_r = np.exp(1j * rng_r.uniform(0, 2 * np.pi, cfg.M))
            h_eff_r = effective_channel(G, H, theta_r)[0]
            acc["Random-phase"].append(required_power_single_user(h_eff_r, gamma, sigma2))
        for k in names:
            power_w[k][mi] = np.mean(acc[k])
    return power_w


def main():
    power_w = sweep_k1()
    names = ["AO-RIS", "Random-phase"]

    fig, ax = new_fig()
    for name in names:
        ax.plot(M_VALUES, to_dbm(power_w[name]), marker=MARKERS[name], color=COLORS[name], label=name)
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, "exp_objB_power_vs_M", "Number of RIS elements M", "Required $P_{tx}$ [dBm]",
                   title=f"Objective B: required power vs M (K=1, target SINR={GAMMA_DB:.0f} dB)")
    save_results("exp_objB_power_vs_M", M=M_VALUES, **{f"{k}_power_w": v for k, v in power_w.items()})

    diffs = np.diff(power_w["AO-RIS"])
    print("AO-RIS required power (W) vs M:", power_w["AO-RIS"])
    print("non-increasing in M:", bool(np.all(diffs <= 1e-9)))
    print("exp_objB_power_vs_M done.")


if __name__ == "__main__":
    main()
