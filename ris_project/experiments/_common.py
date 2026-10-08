"""
Shared plotting/result-saving helpers for the experiments/ scripts, so every
figure uses the same colour per scheme, the same style, and results land in
results/ as .npz (data) next to plots/ as .png (dpi 200).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PLOTS_DIR = os.path.join(ROOT, "plots")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# One consistent colour per scheme, used in every figure.
COLORS = {
    "AO-RIS": "#1f77b4",
    "AO-MRT": "#1f77b4",
    "AO-ZF": "#1f77b4",
    "AO-RZF": "#9467bd",
    "Random-phase": "#ff7f0e",
    "No-RIS": "#2ca02c",
    "AF relay": "#d62728",
    "28 GHz": "#2ca02c",
    "140 GHz": "#1f77b4",
    "300 GHz": "#d62728",
}
MARKERS = {
    "AO-RIS": "o", "AO-MRT": "o", "AO-ZF": "o", "AO-RZF": "s",
    "Random-phase": "^", "No-RIS": "x", "AF relay": "d",
    "28 GHz": "^", "140 GHz": "o", "300 GHz": "s",
}


def new_fig(figsize=(6, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3)
    return fig, ax


def style_and_save(fig, ax, name, xlabel, ylabel, title=None, legend=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


def save_results(name, **arrays):
    path = os.path.join(RESULTS_DIR, f"{name}.npz")
    np.savez(path, **arrays)
    print(f"  saved {path}")
