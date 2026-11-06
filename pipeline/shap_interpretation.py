"""Stage 6: SHAP interpretation for the xgboost model (evaluation model, test set).

SHAP values are computed on the test set after the pipeline transform
(pipeline[:-1]), never on the raw test matrix. For display and LOWESS, the values
are the imputed values brought back to the original log2 scale with
scaler.inverse_transform (LOWESS does not accept NaN on the x axis).
Units: SHAP values are in log-odds. Positive SHAP pushes toward group 1.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

from .config import Config  # noqa: E402
from .plots import save_all, save_figure  # noqa: E402
from .qc import QCResult, ensure_writable  # noqa: E402
from .tuning import TuningResult  # noqa: E402

log = logging.getLogger(__name__)

INTERP_DIR_NAME = "04_interpretation"

# Limits for the interaction heatmap (see _interaction_plan).
MAX_BATCH_BYTES = 256 * 1024 * 1024   # memory per batch of interaction tensors
MAX_INTERACTION_MINUTES = 15.0        # estimated run time above this skips the heatmap
CONSISTENCY_ATOL = 1e-3


class ShapError(RuntimeError):
    """Raised when a SHAP consistency check fails."""


def _points_to(rho: float, group0: str, group1: str) -> str:
    if math.isnan(rho):
        return "undefined (constant feature)"
    return group1 if rho > 0 else group0


def importance_table(
    sv: np.ndarray, X_show: pd.DataFrame, group0: str, group1: str
) -> pd.DataFrame:
    """Mean |SHAP| and direction (Spearman between feature value and SHAP), all features."""
    rows = []
    for j, feature in enumerate(X_show.columns):
        rho = spearmanr(X_show[feature].to_numpy(), sv[:, j]).correlation
        rows.append(
            {
                "feature": feature,
                "mean_abs_shap": float(np.mean(np.abs(sv[:, j]))),
                "spearman_rho": float(rho) if rho is not None else float("nan"),
                "points_to": _points_to(float(rho), group0, group1),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.reset_index(drop=True)


def lowess_zero_crossings(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS curve and the x positions where the curve crosses SHAP = 0 (linear interpolation)."""
    fitted = lowess(y, x, return_sorted=True)
    xs, ys = fitted[:, 0], fitted[:, 1]
    crossings = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y0 == 0:
            crossings.append(xs[i])
        elif y0 * y1 < 0:
            crossings.append(xs[i] + (0 - y0) * (xs[i + 1] - xs[i]) / (y1 - y0))
    return fitted, np.array(crossings)


def _dependence_figure(
    top_features: list[str],
    X_show: pd.DataFrame,
    sv: np.ndarray,
    columns: list[str],
    group0: str,
    group1: str,
    set_name: str = "test",
) -> tuple[plt.Figure, pd.DataFrame]:
    k = len(top_features)
    ncols = min(3, k)
    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    crossing_rows = []
    for idx, feature in enumerate(top_features):
        ax = axes[idx // ncols][idx % ncols]
        j = columns.index(feature)
        x = X_show[feature].to_numpy()
        y = sv[:, j]
        ax.scatter(x, y, s=28, alpha=0.7, color="#4c72b0", edgecolor="none")
        fitted, cross = lowess_zero_crossings(x, y)
        ax.plot(fitted[:, 0], fitted[:, 1], color="#d62728", lw=2.5, label="LOWESS")
        ax.axhline(0, color="gray", ls="--", lw=1)
        if cross.size:
            ax.scatter(cross, np.zeros_like(cross), marker="^", s=90,
                       color="black", zorder=4, label="LOWESS = 0")
        for c in cross:
            crossing_rows.append({"feature": feature, "x_at_shap_zero": float(c)})
        ax.set_title(feature, fontsize=13)
        ax.set_xlabel(f"{feature} (log2, imputed)")
        ax.set_ylabel("SHAP value (log-odds)")
        ax.legend(fontsize=9, frameon=False)
    for idx in range(k, nrows * ncols):  # hide empty axes
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle(
        f"SHAP dependence with LOWESS ({set_name} set)\n"
        f"Positive SHAP pushes toward {group1}; negative toward {group0}",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, pd.DataFrame(crossing_rows, columns=["feature", "x_at_shap_zero"])


def _interaction_plan(n_samples: int, n_features: int, booster, Z: np.ndarray) -> dict:
    """Estimate memory and time for the interaction run before doing it."""
    per_row_bytes = (n_features + 1) ** 2 * 4  # float32 tensor per sample
    batch = max(1, MAX_BATCH_BYTES // per_row_bytes)
    batch = min(batch, n_samples)
    probe = min(5, n_samples)
    t0 = time.perf_counter()
    booster.predict(xgb.DMatrix(Z[:probe]), pred_interactions=True)
    seconds_per_row = max((time.perf_counter() - t0) / probe, 1e-6)
    est_minutes = seconds_per_row * n_samples / 60.0
    return {
        "batch_size": int(batch),
        "batch_mb": round(batch * per_row_bytes / 1024 / 1024, 2),
        "estimated_minutes": round(est_minutes, 2),
        "per_row_kb": round(per_row_bytes / 1024, 2),
    }


def _interaction_heatmap(
    booster, Z_df: pd.DataFrame, sv: np.ndarray, top_idx: list[int], names: list[str],
    interp_dir: Path, formats: list[str], group0: str, group1: str,
    config: Config, set_name: str = "test",
) -> tuple[Path | None, list[str]]:
    """Mean |interaction| for the top features, accumulated batch by batch.

    Only the submatrix of the top features is kept, and the bias term is removed.
    The consistency check runs on the first small batch only.
    """
    warnings: list[str] = []
    Z = Z_df.to_numpy()
    n, p = Z.shape
    plan = _interaction_plan(n, p, booster, Z)
    log.info(
        "Interaction estimate: %d samples x %d features, batch %d (%.2f MB), ~%.2f min",
        n, p, plan["batch_size"], plan["batch_mb"], plan["estimated_minutes"],
    )
    if plan["estimated_minutes"] > MAX_INTERACTION_MINUTES:
        msg = (f"interaction heatmap skipped: estimated {plan['estimated_minutes']} min "
               f"exceeds the {MAX_INTERACTION_MINUTES:g} min limit")
        return None, [msg]

    k = len(top_idx)
    abs_sum = np.zeros((k, k))
    signed_sum = np.zeros((k, k))
    first = True
    for start in range(0, n, plan["batch_size"]):
        stop = min(start + plan["batch_size"], n)
        inter = booster.predict(xgb.DMatrix(Z[start:stop]), pred_interactions=True)
        features_only = inter[:, :-1, :-1]  # drop the bias term
        if first:
            # Row sums of the interaction tensor must equal the SHAP values.
            row_sums = features_only.sum(axis=2)
            max_err = float(np.abs(row_sums - sv[start:stop]).max())
            if max_err > CONSISTENCY_ATOL:
                raise ShapError(
                    f"interaction consistency check failed (max error {max_err:.2e}); "
                    "the heatmap was not written"
                )
            log.info("Interaction consistency check passed (max error %.2e)", max_err)
            first = False
        sub = features_only[:, top_idx][:, :, top_idx]
        abs_sum += np.abs(sub).sum(axis=0)
        signed_sum += sub.sum(axis=0)

    mean_abs = pd.DataFrame(abs_sum / n, index=names, columns=names)
    mean_abs.round(6).to_csv(interp_dir / "shap_interaction_matrix_mean_abs.csv")

    mask = np.triu(np.ones((k, k), dtype=bool), k=1)  # hide the upper triangle; diagonal shown
    fig, ax = plt.subplots(figsize=(1.0 * k + 3, 0.9 * k + 2))
    sns.heatmap(
        mean_abs, mask=mask, annot=True, fmt=".3f", cmap="Reds",
        linewidths=1, linecolor="white", cbar_kws={"label": "mean |SHAP interaction|"},
        annot_kws={"size": 16}, ax=ax,
    )
    ax.set_title(
        f"SHAP interaction heatmap (mean |value|, {set_name} set)\n"
        "Diagonal = main effect of each feature",
        fontsize=14, pad=15,
    )
    fig.tight_layout()
    path = save_figure(fig, interp_dir / f"shap_interaction_heatmap_{set_name}", formats)[0]
    return path, warnings


def run_shap_interpretation(
    config: Config, qc: QCResult, tuning: TuningResult, out_dir: Path | None = None,
    set_name: str = "test",
) -> dict:
    interp_dir = out_dir if out_dir is not None else config.resolved_outdir() / INTERP_DIR_NAME
    ensure_writable(interp_dir)
    formats = config.figure_formats
    g0, g1 = config.group0_name, config.group1_name

    pipe = tuning.best_estimator
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]
    safe = tuning.feature_names_safe
    original = tuning.feature_names_original

    X_all = qc.X.copy()
    X_all.columns = safe
    X_test = X_all.loc[qc.test_ids]

    # Transform with the pipeline minus the final model (never the raw test matrix).
    Z = pipe[:-1].transform(X_test)
    Z_df = pd.DataFrame(Z, columns=safe, index=qc.test_ids)
    # Display and LOWESS: imputed values back on the original log2 scale.
    X_show = pd.DataFrame(scaler.inverse_transform(Z), columns=original, index=qc.test_ids)

    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(Z))
    margin = model.predict(Z, output_margin=True)
    additivity = float(np.abs(sv.sum(axis=1) + explainer.expected_value - margin).max())
    log.info("SHAP additivity check: max error %.2e", additivity)
    if additivity > 1e-3:
        raise ShapError(f"SHAP additivity check failed (max error {additivity:.2e})")

    # Full table (all features), sorted by mean |SHAP|.
    importance = importance_table(sv, X_show, g0, g1)
    importance.round(6).to_csv(interp_dir / "shap_importance.csv", index=False)

    # Beeswarm and bar plots (traditional SHAP look, title added).
    n_feat = len(original)
    plt.figure()
    shap.summary_plot(sv, X_show, show=False, max_display=n_feat)
    plt.gcf().set_size_inches(10, max(4.0, 0.45 * n_feat + 2))
    plt.title(f"SHAP beeswarm ({set_name} set)\nPositive SHAP pushes toward {g1}", fontsize=13)
    save_all(interp_dir / f"shap_beeswarm_{set_name}", formats)

    plt.figure()
    shap.summary_plot(sv, X_show, plot_type="bar", show=False, max_display=n_feat)
    plt.gcf().set_size_inches(10, max(4.0, 0.45 * n_feat + 2))
    plt.title(f"SHAP feature importance (mean |SHAP|, {set_name} set)", fontsize=13)
    save_all(interp_dir / f"shap_bar_{set_name}", formats)

    # Dependence plots for the top-K features.
    top_k = min(config.shap.top_k, n_feat)
    top_features = importance["feature"].head(top_k).tolist()
    fig, crossings = _dependence_figure(top_features, X_show, sv, original, g0, g1, set_name)
    save_figure(fig, interp_dir / f"shap_dependence_lowess_{set_name}", formats)
    crossings.round(6).to_csv(interp_dir / "shap_dependence_zero_crossings.csv", index=False)

    # Interaction heatmap (triangular), optional.
    warnings: list[str] = []
    heatmap_path = None
    if not config.shap.compute_interactions:
        warnings.append("interaction heatmap disabled in params (shap.compute_interactions: false)")
    else:
        k_h = min(config.shap.interaction_top_k, n_feat)
        if k_h < 3:
            warnings.append(f"interaction heatmap skipped: only {k_h} feature(s) available (minimum 3)")
        else:
            top_idx = [original.index(f) for f in importance["feature"].head(k_h)]
            names = [original[i] for i in top_idx]
            heatmap_path, extra = _interaction_heatmap(
                model.get_booster(), Z_df, sv, top_idx, names,
                interp_dir, formats, g0, g1, config, set_name,
            )
            warnings += extra
    for w in warnings:
        log.warning(w)

    return {
        "output_dir": interp_dir,
        "top_k": top_k,
        "top_features": top_features,
        "additivity_error": additivity,
        "heatmap": heatmap_path,
        "warnings": warnings,
        "importance": importance,
    }
