"""
Phase 2, Milestone 1: Objective B. Outage probability vs target SINR gamma
at a fixed power budget Pmax (K=1, M=64, the default Pmax=30 dBm): the
fraction of 200 Monte-Carlo channel draws where the required power
exceeds Pmax (infeasible at that gamma), for AO-RIS, random-phase RIS,
no-RIS, and the AF relay.
Run: python -m experiments.exp_objB_outage
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from metrics import effective_channel
from objective_b import (
    min_power_single_user, required_power_single_user, af_relay_required_power, is_outage,
)
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
GAMMA_DB = np.arange(0, 41, 4)


def sweep_k1():
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    outage = {k: np.zeros(len(GAMMA_DB)) for k in names}
    cfg0 = Config(K=1)
    Pmax = cfg0.Pmax_w

    for gi, gamma_db in enumerate(tqdm(GAMMA_DB, desc="exp_objB outage")):
        gamma = 10 ** (gamma_db / 10)
        n_out = {k: 0 for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            sigma2 = cfg.noise_power_w

            _, _, P_ao = min_power_single_user(H, G, gamma, sigma2)
            n_out["AO-RIS"] += is_outage(P_ao, Pmax)

            rng_r = cfg.rng(i + 10000)
            theta_r = np.exp(1j * rng_r.uniform(0, 2 * np.pi, cfg.M))
            h_eff_r = effective_channel(G, H, theta_r)[0]
            P_rand = required_power_single_user(h_eff_r, gamma, sigma2)
            n_out["Random-phase"] += is_outage(P_rand, Pmax)

            P_noris = required_power_single_user(d[0], gamma, sigma2)
            n_out["No-RIS"] += is_outage(P_noris, Pmax)

            P_relay = af_relay_required_power(cfg, cfg.rng(i + 20000), ue_pos, gamma, sigma2)
            n_out["AF relay"] += np.isinf(P_relay) or is_outage(P_relay, Pmax)

        for k in names:
            outage[k][gi] = n_out[k] / N_DRAWS
    return outage, Pmax


def main():
    outage, Pmax = sweep_k1()
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]

    fig, ax = new_fig()
    for name in names:
        ax.plot(GAMMA_DB, outage[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp_objB_outage", "Target SINR $\\gamma$ [dB]", "Outage probability",
                   title=f"Objective B: outage vs target SINR (K=1, M=64, Pmax={10*np.log10(Pmax)+30:.0f} dBm)")
    save_results("exp_objB_outage", gamma_db=GAMMA_DB, pmax_w=np.array([Pmax]), **outage)

    for name in names:
        print(f"  {name}: outage = {np.round(outage[name], 3)}")
    print("exp_objB_outage done.")


if __name__ == "__main__":
    main()
