"""Stage 5: interpretation of the elastic net (coefficients, relative importance, lollipops).

Coefficients come from the evaluation model (fit on the training split).
Sign convention: positive coefficient = higher values point to group 1;
negative coefficient = higher values point to group 0.
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .plots import save_all  # noqa: E402
from .qc import ensure_writable  # noqa: E402
from .tuning import TuningResult  # noqa: E402

log = logging.getLogger(__name__)

INTERP_DIR_NAME = "04_interpretation"
LOLLIPOP_MAX = 10


def coefficient_table(tuning: TuningResult) -> pd.DataFrame:
    """One row per feature, with the original names and the signed coefficient."""
    model = tuning.best_estimator.named_steps["model"]
    coefs = np.ravel(model.coef_)
    if len(coefs) != len(tuning.feature_names_original):
        raise ValueError("number of coefficients does not match the number of features")
    return pd.DataFrame(
        {"feature": tuning.feature_names_original, "coefficient": coefs}
    )


def importance_table(coefs: pd.DataFrame, group0_name: str, group1_name: str) -> pd.DataFrame:
    """Relative importance computed separately for each direction.

    Only non-zero coefficients are kept.
    Positive: importance = coef / max(positive coefs) * 100.
    Negative: importance = |coef| / max(|negative coefs|) * 100.
    So the strongest feature in each direction is exactly 100%.
    """
    nonzero = coefs.loc[coefs["coefficient"] != 0].copy()
    nonzero["direction"] = np.where(nonzero["coefficient"] > 0, "positive", "negative")
    nonzero["points_to"] = np.where(
        nonzero["coefficient"] > 0, group1_name, group0_name
    )
    nonzero["abs_coefficient"] = nonzero["coefficient"].abs()

    parts = []
    for direction in ("positive", "negative"):
        block = nonzero.loc[nonzero["direction"] == direction].copy()
        if block.empty:
            continue
        block["importance_pct"] = 100.0 * block["abs_coefficient"] / block["abs_coefficient"].max()
        parts.append(block)
    if not parts:
        return nonzero.assign(importance_pct=pd.Series(dtype=float))

    table = pd.concat(parts)
    table["direction_order"] = table["direction"].map({"positive": 0, "negative": 1})
    table = table.sort_values(["direction_order", "importance_pct"], ascending=[True, False])
    return table[["feature", "coefficient", "direction", "points_to", "importance_pct"]].reset_index(
        drop=True
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _xlabel(config: Config) -> str:
    if config.data_type.strip().lower() in ("proteomica", "proteomics"):
        return "Protein importance (%)"
    return "Feature importance (%)"


def plot_lollipop(
    rows: pd.DataFrame,
    *,
    color: str,
    title: str,
    xlabel: str,
    output_stem: Path,
    formats: list[str],
) -> list[Path]:
    """Horizontal lollipop: most important feature at the top. Style follows the spec."""
    names = rows["feature"].tolist()
    values = rows["importance_pct"].to_numpy(dtype=float)
    y = np.arange(len(names))[::-1]  # first row (largest value) drawn at the top

    # Height grows with the number of features; width is fixed.
    height = max(3.5, 2.2 + 0.6 * len(names))
    fig, ax = plt.subplots(figsize=(9, height))
    ax.hlines(y, 0, values, color=color, lw=3.4, alpha=0.8)  # stem, ~ggplot size 1.2
    ax.scatter(values, y, color=color, s=200, alpha=0.9, zorder=3)  # head, ~ggplot size 5

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel, fontsize=22, labelpad=15)
    ax.set_ylabel("")
    ax.set_title("\n".join(textwrap.wrap(title, 40)), fontsize=24, pad=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=18, colors="black")
    ax.tick_params(axis="x", labelcolor="black")
    ax.tick_params(axis="y", labelcolor="black")
    fig.tight_layout()
    return save_all(output_stem, formats)


def run_elasticnet_interpretation(
    config: Config, tuning: TuningResult, out_dir: Path | None = None
) -> dict:
    interp_dir = out_dir if out_dir is not None else config.resolved_outdir() / INTERP_DIR_NAME
    ensure_writable(interp_dir)

    coefs = coefficient_table(tuning)
    n_total = len(coefs)
    n_zero = int((coefs["coefficient"] == 0).sum())
    n_pos = int((coefs["coefficient"] > 0).sum())
    n_neg = int((coefs["coefficient"] < 0).sum())

    # Three coefficient tables, sorted by magnitude (descending).
    def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.assign(
            abs_coefficient=frame["coefficient"].abs(),
            points_to=np.where(frame["coefficient"] > 0, config.group1_name, config.group0_name),
        )
        return out.sort_values("abs_coefficient", ascending=False)[
            ["feature", "coefficient", "points_to"]
        ].reset_index(drop=True)

    _sorted(coefs.loc[coefs["coefficient"] > 0]).to_csv(
        interp_dir / "coefficients_positive.csv", index=False)
    _sorted(coefs.loc[coefs["coefficient"] < 0]).to_csv(
        interp_dir / "coefficients_negative.csv", index=False)
    _sorted(coefs.loc[coefs["coefficient"] != 0]).to_csv(
        interp_dir / "coefficients_all.csv", index=False)

    # Relative importance table (both directions, consolidated).
    importance = importance_table(coefs, config.group0_name, config.group1_name)
    importance.round(4).to_csv(interp_dir / "importance_pct.csv", index=False)

    # Lollipops: one figure per direction, skipped with a warning if the direction is empty.
    warnings: list[str] = []
    figures: list[Path] = []
    xlabel = _xlabel(config)
    specs = [
        ("positive", config.group1_name, config.colors.group1, "lollipop_positive"),
        ("negative", config.group0_name, config.colors.group0, "lollipop_negative"),
    ]
    for direction, group_name, color, stem in specs:
        block = importance.loc[importance["direction"] == direction].head(LOLLIPOP_MAX)
        if block.empty:
            msg = (f"no features with a {direction} non-zero coefficient; "
                   f"skipping the {direction} lollipop plot")
            log.warning(msg)
            warnings.append(msg)
            continue
        title = f"{group_name}"
        figures += plot_lollipop(
            block,
            color=color,
            title=title,
            xlabel=xlabel,
            output_stem=interp_dir / f"{stem}_{_slug(group_name)}",
            formats=config.figure_formats,
        )

    summary_lines = [
        "ELASTIC NET INTERPRETATION (stage 5)",
        f"Model: evaluation model fit on the training split",
        f"Features used by the model: {n_total}",
        f"Non-zero coefficients: {n_pos + n_neg} "
        f"({n_pos} point to {config.group1_name}, {n_neg} point to {config.group0_name})",
        f"Zeroed by regularization: {n_zero}",
        "Importance: computed separately for each direction; the strongest feature in "
        "each direction is 100%.",
        *[f"WARNING: {w}" for w in warnings],
    ]
    (interp_dir / "interpretation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "output_dir": interp_dir,
        "n_nonzero": n_pos + n_neg,
        "n_zero": n_zero,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "warnings": warnings,
        "figures": figures,
    }
