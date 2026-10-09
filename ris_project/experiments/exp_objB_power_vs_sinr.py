"""
Phase 2, Milestone 1: Objective B. Required BS transmit power (dBm) vs
target per-user SINR (0-30 dB), K=1: AO-RIS, random-phase RIS, no-RIS,
AF relay. 200 Monte-Carlo channel realisations per SINR point (mean of
the *required power*, converted to dBm; infeasible/outage draws, where
the AF relay's second hop alone cannot reach the target, are excluded
from the relay's mean and counted separately as an outage rate).
Run: python -m experiments.exp_objB_power_vs_sinr
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from metrics import effective_channel
from objective_b import min_power_single_user, required_power_single_user, af_relay_required_power
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
SINR_DB = np.arange(0, 31, 5)


def to_dbm(p_watts):
    return 10 * np.log10(p_watts) + 30


def sweep_k1():
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    power_w = {k: np.zeros(len(SINR_DB)) for k in names}
    outage_rate = {k: np.zeros(len(SINR_DB)) for k in names}
    for si, sinr_db in enumerate(tqdm(SINR_DB, desc="exp_objB power vs SINR")):
        gamma = 10 ** (sinr_db / 10)
        acc = {k: [] for k in names}
        n_outage = {k: 0 for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1)
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            d = direct_channel(cfg, rng, ue_pos)
            sigma2 = cfg.noise_power_w

            _, _, P_ao = min_power_single_user(H, G, gamma, sigma2)
            acc["AO-RIS"].append(P_ao)

            rng_r = cfg.rng(i + 10000)
            theta_r = np.exp(1j * rng_r.uniform(0, 2 * np.pi, cfg.M))
            h_eff_r = effective_channel(G, H, theta_r)[0]
            acc["Random-phase"].append(required_power_single_user(h_eff_r, gamma, sigma2))

            acc["No-RIS"].append(required_power_single_user(d[0], gamma, sigma2))

            P_relay = af_relay_required_power(cfg, cfg.rng(i + 20000), ue_pos, gamma, sigma2)
            if np.isinf(P_relay):
                n_outage["AF relay"] += 1
            else:
                acc["AF relay"].append(P_relay)

        for k in names:
            power_w[k][si] = np.mean(acc[k]) if acc[k] else np.nan
            outage_rate[k][si] = n_outage[k] / N_DRAWS
    return power_w, outage_rate


def main():
    power_w, outage_rate = sweep_k1()
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]

    fig, ax = new_fig()
    for name in names:
        p_dbm = to_dbm(power_w[name])
        ax.plot(SINR_DB, p_dbm, marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp_objB_power_vs_sinr", "Target SINR $\\gamma$ [dB]",
                   "Required $P_{tx}$ [dBm]", title="Objective B: required power vs target SINR (K=1)")
    results = {f"{k}_power_w": v for k, v in power_w.items()}
    results.update({f"{k}_outage_rate": v for k, v in outage_rate.items()})
    save_results("exp_objB_power_vs_sinr", sinr_db=SINR_DB, **results)

    for name in names:
        print(f"  {name}: outage rate vs SINR = {np.round(outage_rate[name], 3)}")
    print("exp_objB_power_vs_sinr done.")


if __name__ == "__main__":
    main()
