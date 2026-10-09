"""
Phase 2, Milestone 3: realistic ITU-R P.676 molecular absorption.

1. Specific attenuation vs frequency (20-350 GHz) at the default condition
   (T=15C, P=1013 hPa, rho=7.5 g/m^3), marking 28/140/300 GHz.
2. Humidity sweep: specific attenuation vs frequency for
   rho in {2, 7.5, 15} g/m^3.
3. 300 GHz distance study: AO-RIS, random-phase, no-RIS, AF relay vs
   RIS-UE distance (K=1), comparing the "itu" (real) and "placeholder"
   (Phase 1) absorption models side by side, and reporting how each
   model's AO-RIS/no-RIS and AO-RIS/AF-relay crossover distances differ
   -- "effect on the RIS-vs-relay crossover size" (Phase 2 spec).

Run: python -m experiments.exp_absorption_itu
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate, af_relay_rate
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
FREQ_GHZ_FINE = np.linspace(20, 350, 100)
MARK_FREQ_GHZ = [28, 140, 300]
RHO_VALUES = [2, 7.5, 15]
DISTANCES_300 = np.array([2, 5, 10, 15, 20, 30, 40], dtype=float)


def plot_attenuation_vs_frequency():
    cfg = Config()
    fig, ax = new_fig()
    atten = np.array([cfg.absorption_coeff(f * 1e9) for f in FREQ_GHZ_FINE])
    ax.plot(FREQ_GHZ_FINE, atten, color="#1f77b4")
    for f in MARK_FREQ_GHZ:
        k = cfg.absorption_coeff(f * 1e9)
        ax.scatter([f], [k], color="#d62728", zorder=5)
        ax.annotate(f"{f} GHz\n{k:.2f} dB/km", (f, k), textcoords="offset points",
                   xytext=(8, 8), fontsize=8)
    style_and_save(fig, ax, "exp_absorption_vs_frequency", "Frequency [GHz]",
                   "Specific attenuation [dB/km]",
                   title="ITU-R P.676 attenuation (15C, 1013 hPa, 7.5 g/m^3)",
                   legend=False)
    save_results("exp_absorption_vs_frequency", freq_ghz=FREQ_GHZ_FINE, atten_db_per_km=atten)


def plot_humidity_sweep():
    fig, ax = new_fig()
    results = {}
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]
    for rho, color in zip(RHO_VALUES, colors):
        cfg = Config(water_vapour_density=rho)
        atten = np.array([cfg.absorption_coeff(f * 1e9) for f in FREQ_GHZ_FINE])
        results[f"atten_rho{rho}"] = atten
        ax.plot(FREQ_GHZ_FINE, atten, color=color, label=f"rho={rho} g/m^3")
    style_and_save(fig, ax, "exp_absorption_humidity_sweep", "Frequency [GHz]",
                   "Specific attenuation [dB/km]",
                   title="ITU-R P.676 gaseous attenuation vs water vapour density")
    save_results("exp_absorption_humidity_sweep", freq_ghz=FREQ_GHZ_FINE, **results)


def _sweep_300ghz(absorption_model: str):
    names = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]
    rates = {k: np.zeros(len(DISTANCES_300)) for k in names}
    for di, dist in enumerate(tqdm(DISTANCES_300, desc=f"300GHz dist ({absorption_model})")):
        acc = {k: [] for k in names}
        for i in range(N_DRAWS):
            cfg = Config(K=1, fc=300e9, ue_height=10.0, absorption_model=absorption_model)
            ue_pos = np.array([[cfg.ris_pos[0] + dist, cfg.ris_pos[1], cfg.ris_pos[2]]])
            rng = cfg.rng(i)
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
            rates[k][di] = np.mean(acc[k])
    return rates


def _first_crossover(x, winner, loser):
    idx = np.where(winner >= loser)[0]
    return float(x[idx[0]]) if len(idx) else None


def distance_study_300ghz():
    results = {}
    crossovers = {}
    fig, ax = new_fig()
    linestyles = {"itu": "-", "placeholder": "--"}
    for model in ["itu", "placeholder"]:
        print(f"Running 300 GHz distance study, absorption_model={model} ...")
        rates = _sweep_300ghz(model)
        for name in ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]:
            results[f"{name}_{model}"] = rates[name]
            label = f"{name} ({model})"
            ax.plot(DISTANCES_300, rates[name], marker=MARKERS[name], color=COLORS[name],
                    linestyle=linestyles[model], label=label, alpha=1.0 if model == "itu" else 0.6)
        crossovers[f"noris_{model}"] = _first_crossover(DISTANCES_300, rates["AO-RIS"], rates["No-RIS"])
        crossovers[f"relay_{model}"] = _first_crossover(DISTANCES_300, rates["AO-RIS"], rates["AF relay"])

    style_and_save(fig, ax, "exp_absorption_300ghz_distance", "RIS-UE distance [m]", "Sum rate [bit/s/Hz]",
                   title="300 GHz: ITU-R (solid) vs placeholder (dashed) absorption model")
    save_results("exp_absorption_300ghz_distance", distance=DISTANCES_300, **results)

    print("\nAO-RIS crossover distances (m), ITU vs placeholder absorption model:")
    for key, val in crossovers.items():
        print(f"  {key}: {val}")
    print("exp_absorption_300ghz_distance done.")


def main():
    print("Plotting specific attenuation vs frequency ...")
    plot_attenuation_vs_frequency()
    print("Plotting humidity sweep ...")
    plot_humidity_sweep()
    distance_study_300ghz()
    print("exp_absorption_itu done.")


if __name__ == "__main__":
    main()
