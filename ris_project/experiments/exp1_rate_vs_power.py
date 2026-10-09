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


def _make_cfg(preset: str, **kwargs) -> Config:
    """preset in {"default", "realistic"} -- see Config.realistic() for what
    changes (lower, more typical antenna gains). Phase 2 Milestone 0(e)."""
    return Config.realistic(**kwargs) if preset == "realistic" else Config(**kwargs)


def sweep_k1(preset: str = "default"):
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(PTX_DBM)) for k in names}
    for pi, ptx_dbm in enumerate(tqdm(PTX_DBM, desc=f"exp1 K=1 ({preset})")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = _make_cfg(preset, K=1, Pmax_dBm=float(ptx_dbm))
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


def sweep_k4(preset: str = "default"):
    names = ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]
    rates = {k: np.zeros(len(PTX_DBM)) for k in names}
    for pi, ptx_dbm in enumerate(tqdm(PTX_DBM, desc=f"exp1 K=4 ({preset})")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = _make_cfg(preset, K=4, Pmax_dBm=float(ptx_dbm))
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


def main(preset: str = "default"):
    """preset in {"default", "realistic"}. The default preset is the one run
    by `python -m experiments.exp1_rate_vs_power`; run the realistic preset
    separately (e.g. `python -c "import experiments.exp1_rate_vs_power as e; e.main('realistic')"`)
    -- kept as an opt-in second run rather than doubling every default run's
    cost. Output files are suffixed "_realistic" for that preset."""
    suffix = "" if preset == "default" else f"_{preset}"
    title_suffix = "" if preset == "default" else f", {preset} gains"

    print(f"Running exp1: rate vs Ptx (K=1, {preset}) ...")
    k1 = sweep_k1(preset)
    fig, ax = new_fig()
    for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
        ax.plot(PTX_DBM, k1[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, f"exp1_rate_vs_power_K1{suffix}", "Transmit power $P_{tx}$ [dBm]", "Sum rate [bit/s/Hz]",
                   title=f"Rate vs Ptx (K=1{title_suffix})")
    save_results(f"exp1_rate_vs_power_K1{suffix}", ptx_dbm=PTX_DBM, **k1)

    print(f"Running exp1: rate vs Ptx (K=4, {preset}) ...")
    k4 = sweep_k4(preset)
    fig, ax = new_fig()
    for name in ["AO-ZF", "AO-RZF", "Random-phase", "No-RIS"]:
        ax.plot(PTX_DBM, k4[name], marker=MARKERS[name], color=COLORS[name], label=name)
    style_and_save(fig, ax, f"exp1_rate_vs_power_K4{suffix}", "Transmit power $P_{tx}$ [dBm]", "Sum rate [bit/s/Hz]",
                   title=f"Rate vs Ptx (K=4{title_suffix})")
    save_results(f"exp1_rate_vs_power_K4{suffix}", ptx_dbm=PTX_DBM, **k4)
    print(f"exp1 ({preset}) done.")


if __name__ == "__main__":
    main()
