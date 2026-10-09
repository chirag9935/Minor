"""
Phase 2, Milestone 4: hardware limitations, discrete RIS phase shifts.

For M in {64, 256} (K=1, Ptx=30 dBm): rate vs number of phase-control
bits (1,2,3,4), both plain quantization (round the AO-converged continuous
solution to the nearest discrete level) and quantization-aware refinement
(coordinate descent over the discrete levels, see hardware.py), against
the continuous-phase bound. Also compares the quantized AO-RIS rate
against the AF relay to see whether/how quantization shifts the
RIS-vs-relay crossover established in exp2b (continuous phases).
Run: python -m experiments.exp_hw_quantization
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from ao import run_ao
from benchmarks import af_relay_rate
from hardware import rate_with_quantization
from experiments._common import new_fig, style_and_save, save_results

N_DRAWS = 200
BITS_VALUES = [1, 2, 3, 4]
M_VALUES = [64, 256]
M_TO_MXMY = {64: (8, 8), 256: (16, 16)}


def sweep_M(M, Mx, My):
    rate_cont = np.zeros(N_DRAWS)
    rate_plain = {b: np.zeros(N_DRAWS) for b in BITS_VALUES}
    rate_refined = {b: np.zeros(N_DRAWS) for b in BITS_VALUES}
    rate_relay = np.zeros(N_DRAWS)

    for i in tqdm(range(N_DRAWS), desc=f"exp_hw_quantization M={M}"):
        cfg = Config(K=1, Mx=Mx, My=My)
        rng = cfg.rng(i)
        ue_pos = get_ue_positions(cfg, rng)
        H = sv_channel_H(cfg, rng)
        G = sv_channel_g(cfg, rng, ue_pos)
        P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

        theta, W, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
        rate_cont[i] = hist[-1]
        rate_relay[i] = af_relay_rate(cfg, cfg.rng(i + 20000), ue_pos, P, sigma2)

        for b in BITS_VALUES:
            r_plain, _ = rate_with_quantization(G, H, W, theta, b, sigma2, refine=False)
            r_refined, _ = rate_with_quantization(G, H, W, theta, b, sigma2, refine=True, max_rounds=3)
            rate_plain[b][i] = r_plain
            rate_refined[b][i] = r_refined

    return rate_cont, rate_plain, rate_refined, rate_relay


def main():
    results = {}
    crossover_notes = []

    for M in M_VALUES:
        Mx, My = M_TO_MXMY[M]
        rate_cont, rate_plain, rate_refined, rate_relay = sweep_M(M, Mx, My)

        mean_cont = np.mean(rate_cont)
        mean_relay = np.mean(rate_relay)
        mean_plain = np.array([np.mean(rate_plain[b]) for b in BITS_VALUES])
        mean_refined = np.array([np.mean(rate_refined[b]) for b in BITS_VALUES])

        results[f"M{M}_rate_cont"] = rate_cont
        results[f"M{M}_rate_relay"] = rate_relay
        for b in BITS_VALUES:
            results[f"M{M}_rate_plain_{b}bit"] = rate_plain[b]
            results[f"M{M}_rate_refined_{b}bit"] = rate_refined[b]

        fig, ax = new_fig()
        ax.plot(BITS_VALUES, mean_plain, marker="o", color="#ff7f0e", label="Plain quantization")
        ax.plot(BITS_VALUES, mean_refined, marker="s", color="#1f77b4", label="Quantization-aware refine")
        ax.axhline(mean_cont, color="#2ca02c", linestyle="--", label="Continuous-phase bound")
        ax.axhline(mean_relay, color="#d62728", linestyle=":", label="AF relay (continuous)")
        style_and_save(fig, ax, f"exp_hw_quantization_M{M}", "RIS phase control bits",
                       "Sum rate [bit/s/Hz]", title=f"Rate vs phase-control bits (K=1, M={M}, Ptx=30 dBm)")

        print(f"\nM={M}: continuous bound = {mean_cont:.3f} bit/s/Hz, AF relay = {mean_relay:.3f} bit/s/Hz")
        for b, p, r in zip(BITS_VALUES, mean_plain, mean_refined):
            gap_plain = mean_cont - p
            gap_refined = mean_cont - r
            beats_relay = ">=" if r >= mean_relay else "<"
            crossover_notes.append(
                f"M={M}, {b}-bit refined rate {r:.3f} {beats_relay} AF relay {mean_relay:.3f}")
            print(f"  {b}-bit: plain={p:.3f} (gap {gap_plain:.3f}), "
                  f"refined={r:.3f} (gap {gap_refined:.3f})")

    save_results("exp_hw_quantization", bits=np.array(BITS_VALUES), **results)
    print("\nRIS-vs-relay crossover under quantization:")
    for line in crossover_notes:
        print(f"  {line}")
    print("exp_hw_quantization done.")


if __name__ == "__main__":
    main()
