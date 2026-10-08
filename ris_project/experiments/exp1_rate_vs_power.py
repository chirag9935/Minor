"""
Experiment 1: Achievable rate vs BS transmit power (Ptx = 0..30 dBm, step 5).

Figure 1 (K=1): AO-RIS (MRT + closed-form phase), Random-phase RIS, No-RIS, AF relay.
Figure 2 (K=4): AO-ZF, AO-RZF, Random-phase RIS, No-RIS.

200 Monte-Carlo channel realisations per Ptx point, mean sum rate plotted.
Run: python -m experiments.exp1_rate_vs_power
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate, af_relay_rate
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
PTX_DBM = np.arange(0, 31, 5)


def sweep_k1():
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(PTX_DBM)) for k in names}
    for pi, ptx_dbm in enumerate(tqdm(PTX_DBM, desc="exp1 K=1")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1, Pmax_dBm=float(ptx_dbm))
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
            rates[k][pi] = np.mean(acc[k])
    return rates


def sweep_k4():
    names = ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]
    rates = {k: np.zeros(len(PTX_DBM)) for k in names}
    for pi, ptx_dbm in enumerate(tqdm(PTX_DBM, desc="exp1 K=4")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=4, Pmax_dBm=float(ptx_dbm))
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
            rates[k][pi] = np.mean(acc[k])
    return rates


def main():
    print("Running exp1: rate vs Ptx (K=1) ...")
    k1 = sweep_k1()
    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
        ax.plot(PTX_DBM, k1[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp1_rate_vs_power_K1", "Transmit power $P_{tx}$ [dBm]", "Sum rate [bit/s/Hz]",
                   title="Rate vs Ptx (K=1)")
    save_results("exp1_rate_vs_power_K1", ptx_dbm=PTX_DBM, **k1)

    print("Running exp1: rate vs Ptx (K=4) ...")
    k4 = sweep_k4()
    fig, ax = new_fig()
    for name in ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]:
        ax.plot(PTX_DBM, k4[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, "exp1_rate_vs_power_K4", "Transmit power $P_{tx}$ [dBm]", "Sum rate [bit/s/Hz]",
                   title="Rate vs Ptx (K=4)")
    save_results("exp1_rate_vs_power_K4", ptx_dbm=PTX_DBM, **k4)
    print("exp1 done.")


if __name__ == "__main__":
    main()
