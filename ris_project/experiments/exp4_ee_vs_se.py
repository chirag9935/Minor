"""
Experiment 4: Energy efficiency (EE) vs spectral efficiency (SE) trade-off.

Sweeps Ptx = 0..30 dBm (step 5) for M in {16, 64, 256}, AO-RIS scheme, K=1
(the K=1 AO-MRT scheme is used as the representative single-user operating
point; see docs/ASSUMPTIONS.md). One EE-vs-SE curve per M, one marker per
Ptx point. 200 Monte-Carlo channel realisations per point.
Run: python -m experiments.exp4_ee_vs_se
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from ao import run_ao
from metrics import energy_efficiency
from experiments._common import new_fig, style_and_save, save_results

N_DRAWS = 200
PTX_DBM = np.arange(0, 31, 5)
M_VALUES = [16, 64, 256]
M_TO_MXMY = {16: (4, 4), 64: (8, 8), 256: (16, 16)}
COLORS_M = {16: "#2ca02c", 64: "#1f77b4", 256: "#9467bd"}


def sweep_M(M):
    Mx, My = M_TO_MXMY[M]
    se = np.zeros(len(PTX_DBM))
    ee = np.zeros(len(PTX_DBM))
    for pi, ptx_dbm in enumerate(tqdm(PTX_DBM, desc=f"exp4 M={M}")):
        se_acc, ee_acc = [], []
        for i in range(N_DRAWS):
            cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=float(ptx_dbm))
            rng = cfg.rng(i)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w

            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            rate = hist[-1]
            EE, SE = energy_efficiency(cfg, rate, P)
            se_acc.append(SE)
            ee_acc.append(EE)
        se[pi] = np.mean(se_acc)
        ee[pi] = np.mean(ee_acc)
    return se, ee


def main():
    fig, ax = new_fig()
    results = {}
    for M in M_VALUES:
        print(f"Running exp4: EE vs SE, M={M} ...")
        se, ee = sweep_M(M)
        results[f"se_M{M}"] = se
        results[f"ee_M{M}"] = ee
        ax.plot(se, ee / 1e6, marker="o", color=COLORS_M[M], label=f"M={M}")
    style_and_save(fig, ax, "exp4_ee_vs_se", "Spectral efficiency [bit/s/Hz]", "Energy efficiency [Mbit/J]",
                   title="EE vs SE trade-off (AO-RIS, K=1)")
    save_results("exp4_ee_vs_se", ptx_dbm=PTX_DBM, **results)
    print("exp4 done.")


if __name__ == "__main__":
    main()
