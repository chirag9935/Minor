"""
Phase 2, Milestone 0(a): blockage-loss sensitivity sweep.

Investigates the finding flagged after the Phase 1 mid-evaluation: at the
Phase 1 default of blockage_loss_db=40, the un-optimized blocked direct
link (no-RIS) beat both AO-RIS and the AF relay, which is inconsistent
with a "completely blocked" link. Sweeps blockage_loss_db in
{0,20,40,60,80,100} dB at M in {64, 256} (K=1, Ptx=30 dBm) and reports the
blockage level at which AO-RIS first overtakes no-RIS and the AF relay.

This is the evidence behind config.py's default blockage_loss_db=80 (see
docs/ASSUMPTIONS.md, "Milestone 0").
Run: python -m experiments.exp_blockage_sweep
"""
import numpy as np
from tqdm import tqdm

from config import Config
from channel import get_ue_positions, sv_channel_H, sv_channel_g, direct_channel
from ao import run_ao
from benchmarks import no_ris_rate, random_phase_rate, af_relay_rate
from experiments._common import new_fig, style_and_save, save_results, COLORS, MARKERS

N_DRAWS = 200
PTX_DBM = 30.0
BLOCKAGE_DB = np.array([0, 20, 40, 60, 80, 100], dtype=float)
M_PANELS = [(64, (8, 8)), (256, (16, 16))]
SCHEMES = ["AO-RIS", "Random-phase", "No-RIS", "AF relay"]


def sweep(Mx, My, desc):
    rates = {k: np.zeros(len(BLOCKAGE_DB)) for k in SCHEMES}
    for bi, blk in enumerate(tqdm(BLOCKAGE_DB, desc=desc)):
        acc = {k: [] for k in SCHEMES}
        for i in range(N_DRAWS):
            cfg = Config(K=1, Mx=Mx, My=My, Pmax_dBm=PTX_DBM, blockage_loss_db=float(blk))
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
        for k in SCHEMES:
            rates[k][bi] = np.mean(acc[k])
    return rates


def first_crossover(blockage_db, winner, loser):
    """First blockage_db value (from the swept grid) where winner >= loser."""
    idx = np.where(winner >= loser)[0]
    return float(blockage_db[idx[0]]) if len(idx) else None


def main():
    for M, (Mx, My) in M_PANELS:
        print(f"Running exp_blockage_sweep: M={M} ...")
        rates = sweep(Mx, My, desc=f"blockage M={M}")

        fig, ax = new_fig()
        for name in SCHEMES:
            ax.plot(BLOCKAGE_DB, rates[name], marker=MARKERS[name], color=COLORS[name], label=name)
        style_and_save(fig, ax, f"exp_blockage_sweep_M{M}", "Blockage loss [dB]", "Sum rate [bit/s/Hz]",
                       title=f"Rate vs blockage loss (K=1, M={M}, Ptx=30 dBm)")
        save_results(f"exp_blockage_sweep_M{M}", blockage_db=BLOCKAGE_DB, **rates)

        x_noris = first_crossover(BLOCKAGE_DB, rates["AO-RIS"], rates["No-RIS"])
        x_relay = first_crossover(BLOCKAGE_DB, rates["AO-RIS"], rates["AF relay"])
        print(f"  M={M}: AO-RIS first beats No-RIS at blockage >= "
              f"{x_noris if x_noris is not None else 'never (within 0-100 dB)'} dB")
        print(f"  M={M}: AO-RIS first beats AF relay at blockage >= "
              f"{x_relay if x_relay is not None else 'never (within 0-100 dB)'} dB "
              f"(relay bypasses blockage entirely, so this is expected to rarely/never trigger)")
    print("exp_blockage_sweep done.")


if __name__ == "__main__":
    main()
