"""
Experiment 6: Frequency comparison (28 GHz vs 140 GHz vs 300 GHz), AO-RIS
sum rate vs RIS-UE distance (K=1), isolating the effect of the molecular
absorption loss Lmol(fc,d) on top of the (dominant) free-space path loss.

A second, supplementary panel plots Lmol itself (in dB) vs distance for the
three frequencies -- the free-space path loss difference between bands is
large (>10 dB) and would otherwise swamp the comparatively small absorption
effect in the main rate plot, so this panel makes the absorption model's
contribution directly visible.
Run: python -m experiments.exp6_frequency
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import sv_channel_H, sv_channel_g, molecular_absorption
from ao import run_ao
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
# Kept short enough that the 300 GHz curve stays meaningfully above zero
# (at THz, usable range shrinks fast -- see the absorption panel below for
# the longer-range view of the absorption loss itself).
DISTANCES = np.array([3, 6, 10, 15, 20, 30, 40], dtype=float)
FREQS = [28e9, 140e9, 300e9]
FREQ_LABEL = {28e9: "28 GHz", 140e9: "140 GHz", 300e9: "300 GHz"}


def sweep_frequency(fc):
    rates = np.zeros(len(DISTANCES))
    for di, dist in enumerate(tqdm(DISTANCES, desc=f"exp6 {FREQ_LABEL[fc]}")):
        vals = []
        for i in range(N_DRAWS):
            cfg = Config(K=1, fc=fc, ue_height=10.0)  # match RIS height -> horizontal dist = 3D dist
            ue_pos = np.array([[cfg.ris_pos[0] + dist, cfg.ris_pos[1], cfg.ris_pos[2]]])
            rng = cfg.rng(i)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w
            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            vals.append(hist[-1])
        rates[di] = np.mean(vals)
    return rates


def main():
    print("Running exp6: AO-RIS rate vs distance, per frequency ...")
    fig, ax = new_fig()
    results = {}
    for fc in FREQS:
        label = FREQ_LABEL[fc]
        rates = sweep_frequency(fc)
        results[label] = rates
        ax.plot(DISTANCES, rates, marker=MARKERS[label], color=COLORS[label], label=label)
    style_and_save(fig, ax, "exp6_rate_vs_frequency", "RIS-UE distance [m]", "Sum rate [bit/s/Hz]",
                   title="AO-RIS rate vs distance, by carrier frequency (K=1, Ptx=30 dBm)")
    save_results("exp6_rate_vs_frequency", distance=DISTANCES, **results)

    print("Running exp6: molecular absorption factor vs distance ...")
    cfg = Config()
    fig, ax = new_fig()
    d_fine = np.linspace(1, 150, 100)
    abs_results = {}
    for fc in FREQS:
        label = FREQ_LABEL[fc]
        k = cfg.absorption_coeff(fc)
        lmol_db = 10 * np.log10(molecular_absorption(k, d_fine))
        abs_results[label] = lmol_db
        ax.plot(d_fine, lmol_db, color=COLORS[label], label=f"{label} (k={k} dB/km)")
    style_and_save(fig, ax, "exp6_absorption_vs_distance", "Distance [m]", "Molecular absorption loss [dB]",
                   title="Molecular absorption loss Lmol(fc, d) (placeholder coefficients)")
    save_results("exp6_absorption_vs_distance", distance=d_fine, **abs_results)
    print("exp6 done.")


if __name__ == "__main__":
    main()
