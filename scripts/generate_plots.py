"""Render the committed evidence with consistent, deterministic figure styling.

Context and cache figures read CSVs; decay curves are analytic envelopes.
Run from the repository root: python -m scripts.generate_plots
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#182b36"
TEAL = "#007f79"
ORANGE = "#a64b19"
BLUE = "#4266a6"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#819099", "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "savefig.facecolor": "white",
    })


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig, name, out_dir, note):
    fig.text(0.06, 0.035, note, fontsize=9, color=INK)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.86, bottom=0.30, wspace=0.32)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / name
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {path}")


def plot_context_sweep(csv_path="results/context_sweep_results.csv", out_dir="results/figures"):
    rows = read_rows(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.suptitle("Context extrapolation depends on the baseline", x=0.06, ha="left", fontsize=19, fontweight="bold")
    series = [
        ("ResonatorLM (Paper Model)", "ResonatorLM", TEAL, "o", "-"),
        ("Transformer (Learned PE - Paper)", "Transformer · learned positions", ORANGE, "s", "--"),
        ("Transformer (RoPE - Strong Baseline)", "Transformer · RoPE", BLUE, "^", "-."),
    ]
    for key, label, color, marker, line in series:
        selected = sorted((r for r in rows if r["model"] == key), key=lambda r: int(r["context"]))
        xs = [int(r["context"]) for r in selected]
        for ax, metric in zip(axes, ("ppl", "acc")):
            ax.plot(xs, [float(r[metric]) for r in selected], label=label,
                    color=color, marker=marker, linestyle=line, linewidth=2.3, markersize=6)
    for ax, title, ylabel in zip(axes, ("Test perplexity ↓", "Next-character accuracy ↑"), ("Perplexity", "Accuracy (%)")):
        ax.set(title=title, xlabel="Context length (characters)", ylabel=ylabel, xticks=[256, 512, 1024])
        ax.grid(axis="y", color="#dce2e6", linewidth=0.7)
    axes[0].set_ylim(2, 22)
    axes[1].set_ylim(25, 70)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.055, 0.09), ncol=3, fontsize=10)
    finish(fig, "context_sweep.png", out_dir,
           "WikiText-2 raw characters · trained at 256 · 2,000 steps · seed 0\nSource: context_sweep_results.csv. Single-seed measurements; no variance estimate.")


def plot_memory_scaling(out_dir="results/figures", csv_path="results/benchmark_memory.csv"):
    rows = read_rows(csv_path)
    xs = [int(r["seq_len"]) for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.suptitle("Constant decode cache, growing attention cache", x=0.06, ha="left", fontsize=18, fontweight="bold")
    ax.plot(xs, [float(r["trans_kv_kib"]) for r in rows], color=ORANGE, marker="s", linewidth=2.3, label="Transformer KV cache")
    ax.plot(xs, [30.0] * len(xs), color=TEAL, marker="o", linewidth=2.3, label="Full ResonatorLM decode cache · 30 KiB")
    ax.plot(xs, [float(r["res_state_kib"]) for r in rows], color=BLUE, linestyle="--", linewidth=2, label="Resonant state only · 12 KiB")
    ax.set(yscale="log", xlabel="Context length (characters)", ylabel="Cache size (KiB, log scale)", xticks=xs)
    ax.set_xticklabels([f"{x // 1024}K" for x in xs])
    ax.grid(axis="y", color="#dce2e6", linewidth=0.7)
    ax.legend(loc="center right", fontsize=10)
    finish(fig, "memory_scaling.png", out_dir,
           "Batch 1 · float32 · six layers. Cache storage only; excludes weights and activations.\nSource: benchmark_memory.csv. Full cache includes 18 KiB of local buffers (tests/test_model.py).")


def plot_half_life_decay(out_dir="results/figures"):
    t = np.linspace(0, 10000, 1000)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    fig.suptitle("Storage size and information retention are different", x=0.06, ha="left", fontsize=18, fontweight="bold")
    colors = ["#3c444b", "#677b89", BLUE, "#775b91", ORANGE, TEAL]
    for hl, color in zip([2, 16, 64, 256, 1024, 2048], colors):
        ax.plot(t, 2.0 ** (-t / hl), label=f"Half-life {hl:,}", color=color, linewidth=1.8)
    ax.axvline(2048, color=ORANGE, linestyle=":", linewidth=1.4)
    ax.axvline(np.log(2) / 1e-4, color=INK, linestyle="--", linewidth=1.4)
    ax.text(2180, 0.82, "Initialization\nceiling: 2,048", color=ORANGE, fontsize=10)
    ax.text(7070, 0.6, "Architectural\nceiling: ≈6,931", color=INK, fontsize=10)
    ax.set(xlabel="Lag (characters)", ylabel="Remaining impulse amplitude", ylim=(0, 1.03), xlim=(0, 10000))
    ax.grid(axis="y", color="#dce2e6", linewidth=0.7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.055, 0.09), ncol=3, fontsize=10)
    finish(fig, "half_life_decay.png", out_dir,
           "Analytic envelopes: 2^(−lag / half-life), for selected half-lives. Not fitted learned-mode curves.\nArchitectural ceiling: ln(2) / α_min, with α_min = 10⁻⁴. This is not a retrieval-task benchmark.")


if __name__ == "__main__":
    style()
    plot_context_sweep()
    plot_memory_scaling()
    plot_half_life_decay()
