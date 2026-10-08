"""
Draws the scenario geometry (top-down view): BS, RIS, a sample of UEs, and
the illustrative blocker rectangle between BS and the UE area. Purely a
visualisation -- the direct link is not geometrically ray-traced (see
channel.py), it is penalised by a fixed blockage_loss_db instead.
Run: python -m experiments.plot_geometry
"""
import numpy as np
import matplotlib.patches as patches

from config import Config
from channel import get_ue_positions
from experiments._common import new_fig, style_and_save


def main():
    cfg = Config(K=4)
    rng = cfg.rng(0)
    ue_pos = get_ue_positions(cfg, rng, K=12)  # a representative scatter of UE positions

    fig, ax = new_fig(figsize=(6.5, 6))
    bx0, bx1, by0, by1, _, _ = cfg.blocker_box
    ax.add_patch(patches.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                    facecolor="0.6", edgecolor="0.3", alpha=0.5, label="Blocker"))

    ax.scatter(*cfg.bs_pos[:2], marker="^", s=150, color="#1f77b4", label="BS", zorder=5)
    ax.scatter(*cfg.ris_pos[:2], marker="s", s=150, color="#9467bd", label="RIS", zorder=5)
    ax.scatter(ue_pos[:, 0], ue_pos[:, 1], marker="o", s=40, color="#d62728", label="UE (sample)", zorder=5)

    ax.annotate("", xy=cfg.ris_pos[:2], xytext=cfg.bs_pos[:2],
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.annotate("BS->RIS (H)", xy=((cfg.bs_pos[0] + cfg.ris_pos[0]) / 2, (cfg.bs_pos[1] + cfg.ris_pos[1]) / 2 + 1))

    ax.set_xlim(-5, 35)
    ax.set_ylim(-10, 20)
    ax.set_aspect("equal")
    style_and_save(fig, ax, "geometry", "x [m]", "y [m]",
                   title="Scenario geometry (top-down view)")


if __name__ == "__main__":
    main()
