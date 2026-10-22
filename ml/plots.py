"""Offline evaluation plots (Matplotlib, no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAVY = "#14213d"
BLUE = "#3b82f6"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def category_distribution(counts: dict, path: Path) -> None:
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE)
    ax.set_xlabel("Articles")
    ax.set_title("Articles per category")
    _save(fig, path)


def length_distributions(headline_lengths, description_lengths, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(headline_lengths, bins=40, color=NAVY)
    axes[0].set_title("Headline length (words)")
    axes[1].hist(description_lengths, bins=40, color=BLUE)
    axes[1].set_title("Description length (words)")
    for ax in axes:
        ax.set_ylabel("Articles")
    _save(fig, path)


def model_comparison(best_per_algorithm: list[dict], path: Path) -> None:
    names = [r["model"] for r in best_per_algorithm]
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, [r[m] for r in best_per_algorithm], width, label=m)
    ax.set_xticks(x + width)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Validation performance by algorithm")
    ax.legend()
    _save(fig, path)


def confusion_matrix_plot(cm, labels: list[str], path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    size = max(6, 0.28 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalised, test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)
