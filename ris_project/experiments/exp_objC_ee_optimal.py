"""
Phase 2, Milestone 2: Objective C, energy-efficiency optimization.

For each M, sweeps Ptx over a fine grid (41 points, -10..30 dBm) with AO
(K=1), computes EE at every point via metrics.energy_efficiency, and picks
the Ptx that maximises EE (1D grid search -- "no new solver needed", per
the Phase 2 spec; refined with a golden-section search around the grid
optimum for a touch more precision). Reuses exp4_ee_vs_se.sweep_M (Phase 1
code, extended not rewritten -- see its ptx_dbm_grid parameter).

Plots: EE-optimal Ptx vs M; max EE vs M; EE-SE trade-off curve with the
EE-optimal point marked per M (extends exp4's plot); sum-rate-optimal
(always Ptx=30 dBm, since rate is monotonic in Ptx) vs EE-optimal
comparison table (printed).

Sensitivity: EE vs RIS element power P_RIS (1, 5, 10, 20 mW) and vs BS
static power, at the EE-optimal Ptx for the default M=64.
Run: python -m experiments.exp_objC_ee_optimal
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g
from ao import run_ao
from metrics import energy_efficiency
from experiments import exp4_ee_vs_se as exp4
from experiments._common import new_fig, style_and_save, save_results, COLORS

PTX_DBM_FINE = np.linspace(-10, 30, 41)
M_VALUES = [16, 32, 64, 128, 256]
M_TO_MXMY = {16: (4, 4), 32: (8, 4), 64: (8, 8), 128: (8, 16), 256: (16, 16)}
M_COLOR = {16: "#2ca02c", 32: "#17becf", 64: "#1f77b4", 128: "#ff7f0e", 256: "#9467bd"}


def _golden_section_refine(se_fn, lo, hi, tol=0.05, max_iter=25):
    """Golden-section search for the Ptx (dBm) maximising EE, refining
    around a bracket [lo, hi] found by the coarse grid. se_fn(ptx_dbm)
    returns (EE, SE)."""
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = se_fn(c)[0]
    fd = se_fn(d)[0]
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = se_fn(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = se_fn(d)[0]
    best_ptx = (a + b) / 2
    ee, se = se_fn(best_ptx)
    return best_ptx, ee, se


def _ee_at_ptx(M, Mx, My, ptx_dbm, n_draws=200):
    """Single-point EE/SE at a given Ptx (used by the golden-section refine)."""
    ee_acc, se_acc = [], []
    for i in range(n_draws):
        cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=float(ptx_dbm))
        rng = cfg.rng(i)
        ue_pos = get_ue_positions(cfg, rng)
        H = sv_channel_H(cfg, rng)
        G = sv_channel_g(cfg, rng, ue_pos)
        P, sigma2 = cfg.Pmax_w, cfg.noise_power_w
        _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
        EE, SE = energy_efficiency(cfg, hist[-1], P)
        ee_acc.append(EE)
        se_acc.append(SE)
    return np.mean(ee_acc), np.mean(se_acc)


def main():
    ee_optimal_ptx = np.zeros(len(M_VALUES))
    ee_max = np.zeros(len(M_VALUES))
    se_at_ee_optimal = np.zeros(len(M_VALUES))
    se_curves, ee_curves = {}, {}

    fig_tradeoff, ax_tradeoff = new_fig()
    for mi, M in enumerate(M_VALUES):
        print(f"Running exp_objC: EE vs Ptx, M={M} ...")
        Mx, My = M_TO_MXMY[M]
        se, ee = exp4.sweep_M(M, ptx_dbm_grid=PTX_DBM_FINE, mx_my=(Mx, My))
        se_curves[f"se_M{M}"] = se
        ee_curves[f"ee_M{M}"] = ee

        i_best = int(np.argmax(ee))
        lo = PTX_DBM_FINE[max(i_best - 1, 0)]
        hi = PTX_DBM_FINE[min(i_best + 1, len(PTX_DBM_FINE) - 1)]
        best_ptx, best_ee, best_se = _golden_section_refine(
            lambda p, M=M, Mx=Mx, My=My: _ee_at_ptx(M, Mx, My, p), lo, hi)
        ee_optimal_ptx[mi] = best_ptx
        ee_max[mi] = best_ee
        se_at_ee_optimal[mi] = best_se

        color = M_COLOR[M]
        ax_tradeoff.plot(se, ee / 1e6, color=color, label=f"M={M}")
        ax_tradeoff.scatter([best_se], [best_ee / 1e6], color=color, marker="*", s=200,
                            edgecolor="black", zorder=5)

    style_and_save(fig_tradeoff, ax_tradeoff, "exp_objC_ee_se_tradeoff",
                   "Spectral efficiency [bit/s/Hz]", "Energy efficiency [Mbit/J]",
                   title="EE vs SE trade-off with EE-optimal point marked (*) per M")

    fig, ax = new_fig()
    ax.plot(M_VALUES, ee_optimal_ptx, marker="o", color="#1f77b4")
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, "exp_objC_ee_optimal_ptx_vs_M", "Number of RIS elements M",
                   "EE-optimal $P_{tx}$ [dBm]", title="EE-optimal transmit power vs M", legend=False)

    fig, ax = new_fig()
    ax.plot(M_VALUES, ee_max / 1e6, marker="o", color="#2ca02c")
    ax.set_xscale("log", base=2)
    style_and_save(fig, ax, "exp_objC_ee_max_vs_M", "Number of RIS elements M",
                   "Maximum EE [Mbit/J]", title="Maximum achievable EE vs M", legend=False)

    save_results("exp_objC_ee_optimal", M=np.array(M_VALUES), ee_optimal_ptx_dbm=ee_optimal_ptx,
                 ee_max=ee_max, se_at_ee_optimal=se_at_ee_optimal, **se_curves, **ee_curves)

    print("\nSum-rate-optimal vs EE-optimal operating point (K=1):")
    print(f"{'M':>5} {'EE-opt Ptx [dBm]':>18} {'EE-opt SE [b/s/Hz]':>20} {'Sum-rate-opt Ptx [dBm]':>24}")
    for mi, M in enumerate(M_VALUES):
        print(f"{M:5d} {ee_optimal_ptx[mi]:18.2f} {se_at_ee_optimal[mi]:20.3f} {30.0:24.1f}")
    print("exp_objC_ee_optimal done.")


def sensitivity():
    """EE vs RIS element power P_RIS (1,5,10,20 mW) and vs BS static power,
    at the EE-optimal Ptx for the default M=64 (assumptions: see
    docs/ASSUMPTIONS.md, cites Huang et al. TWC 2019 for the power-model
    structure, docs/REFERENCES.md [5])."""
    M, Mx, My = 64, 8, 8
    p_ris_mw_values = [1, 5, 10, 20]
    p_static_w_values = [0.2, 1.0, 2.0, 5.0]

    ee_vs_pris = np.zeros(len(p_ris_mw_values))
    for i, p_ris_mw in enumerate(p_ris_mw_values):
        ee_acc = []
        for di in range(200):
            cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=20.0, p_ris_elem_w=p_ris_mw * 1e-3)
            rng = cfg.rng(di)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w
            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            EE, _ = energy_efficiency(cfg, hist[-1], P)
            ee_acc.append(EE)
        ee_vs_pris[i] = np.mean(ee_acc)

    ee_vs_pstatic = np.zeros(len(p_static_w_values))
    for i, p_static in enumerate(p_static_w_values):
        ee_acc = []
        for di in range(200):
            cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=20.0, p_bs_static_w=p_static)
            rng = cfg.rng(di)
            ue_pos = get_ue_positions(cfg, rng)
            H = sv_channel_H(cfg, rng)
            G = sv_channel_g(cfg, rng, ue_pos)
            P, sigma2 = cfg.Pmax_w, cfg.noise_power_w
            _, _, hist = run_ao(H, G, P, sigma2, mode="mrt", max_iter=15)
            EE, _ = energy_efficiency(cfg, hist[-1], P)
            ee_acc.append(EE)
        ee_vs_pstatic[i] = np.mean(ee_acc)

    fig, ax = new_fig()
    ax.plot(p_ris_mw_values, ee_vs_pris / 1e6, marker="o", color="#1f77b4")
    style_and_save(fig, ax, "exp_objC_ee_vs_pris", "RIS element power $P_{RIS}$ [mW]",
                   "Energy efficiency [Mbit/J]", title="EE sensitivity to RIS element power (K=1, M=64, Ptx=20 dBm)",
                   legend=False)

    fig, ax = new_fig()
    ax.plot(p_static_w_values, ee_vs_pstatic / 1e6, marker="o", color="#2ca02c")
    style_and_save(fig, ax, "exp_objC_ee_vs_pstatic", "BS static power $P_{BS,static}$ [W]",
                   "Energy efficiency [Mbit/J]", title="EE sensitivity to BS static power (K=1, M=64, Ptx=20 dBm)",
                   legend=False)

    save_results("exp_objC_ee_sensitivity", p_ris_mw=np.array(p_ris_mw_values), ee_vs_pris=ee_vs_pris,
                 p_static_w=np.array(p_static_w_values), ee_vs_pstatic=ee_vs_pstatic)
    print("exp_objC sensitivity done.")


if __name__ == "__main__":
    main()
    sensitivity()
