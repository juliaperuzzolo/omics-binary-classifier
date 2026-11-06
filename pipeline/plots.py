"""ROC curve and confusion matrix figures (style from the project specification).

One figure per file. Each figure is saved in every format of `figure_formats`
at 300 dpi, then closed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (Docker, servers)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def save_all(fig_path_stem: Path, formats: list[str]) -> list[Path]:
    saved = []
    for fmt in formats:
        path = fig_path_stem.with_suffix(f".{fmt}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)
    plt.close()
    return saved


def plot_roc_curve(
    fpr,
    tpr,
    auc_value: float,
    *,
    plot_color: str,
    positive_label: str,
    negative_label: str,
    set_name: str,
    output_stem: Path,
    formats: list[str],
) -> list[Path]:
    """ROC curve for one set (train or test). `set_name` goes into the title and file name."""
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="--", lw=4, color="gray",
             label="Chance (AUC = 0.50)", alpha=0.8)
    plt.plot(fpr, tpr, color=plot_color,
             label=f"{positive_label} (AUC = {auc_value:.2f})",
             lw=6, alpha=0.9)

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False positive rate", fontsize=20)
    plt.ylabel("True positive rate", fontsize=20)
    plt.title(f"ROC curve ({set_name} set)\n{positive_label} vs {negative_label}",
              fontsize=24, pad=20)
    plt.legend(loc="lower right", fontsize=23, frameon=True, borderpad=0.7, labelspacing=0.3)
    plt.tight_layout()
    return save_all(output_stem, formats)


def plot_confusion_matrix(
    cm,
    *,
    positive_label: str,
    negative_label: str,
    set_name: str,
    output_stem: Path,
    formats: list[str],
) -> list[Path]:
    """Confusion matrix for one set. `cm` is [[TN, FP], [FN, TP]]."""
    with plt.rc_context({"font.size": 20}):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[negative_label, positive_label],
            yticklabels=[negative_label, positive_label],
            cbar=False, linewidths=2, linecolor="white",
            annot_kws={"size": 35}, ax=ax,
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.title(f"Confusion matrix ({set_name} set)\n{positive_label} vs {negative_label}",
                  pad=20)
        plt.tight_layout()
        return save_all(output_stem, formats)


def save_figure(fig, fig_path_stem: Path, formats: list[str]) -> list[Path]:
    """Save a specific figure object in every format (300 dpi), then close it."""
    saved = []
    for fmt in formats:
        path = fig_path_stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved
