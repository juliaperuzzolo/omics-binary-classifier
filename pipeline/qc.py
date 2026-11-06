"""Stage 2: data QC, missing-value feature filter and stratified train/test split.

The feature filter uses all samples and never looks at the labels. The split
is done after the filter, and it is stratified on the labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config
from .data import Dataset

log = logging.getLogger(__name__)

QC_DIR_NAME = "01_qc"


class QCError(RuntimeError):
    """Raised when QC or the split cannot proceed. The message says what to change."""


@dataclass
class FilterResult:
    kept: list[str]
    table: pd.DataFrame  # one row per feature: counts, percentages, kept, reason


@dataclass
class QCResult:
    X: pd.DataFrame          # features kept after the filter, all samples
    y: pd.Series
    train_ids: list[str]
    test_ids: list[str]
    kept_features: list[str]
    removed_features: pd.DataFrame
    report: str


def ensure_writable(directory: Path) -> None:
    """Create the folder if needed and check that files can be written into it."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise QCError(f"output folder is not writable: {directory} ({exc})") from exc


def filter_features(X: pd.DataFrame, min_filled_percent: float) -> FilterResult:
    """Keep features that have at least `min_filled_percent` % of samples with a value.

    Example: min_filled_percent = 80 keeps only features filled in at least 80% of samples.
    Features with no values at all are always removed.
    """
    n = len(X)
    rows = []
    for feature in X.columns:
        filled = int(X[feature].notna().sum())
        if filled == 0:
            reason = "no values in any sample"
        elif filled * 100 < min_filled_percent * n:
            reason = (
                f"filled in {100.0 * filled / n:.1f}% of samples "
                f"(minimum {min_filled_percent:g}%)"
            )
        else:
            reason = ""
        rows.append(
            {
                "feature": feature,
                "n_filled": filled,
                "filled_percent": round(100.0 * filled / n, 2),
                "null_percent": round(100.0 * (n - filled) / n, 2),
                "kept": reason == "",
                "reason": reason,
            }
        )
    table = pd.DataFrame(rows)
    kept = table.loc[table["kept"], "feature"].tolist()
    return FilterResult(kept=kept, table=table)


def split_samples(
    y: pd.Series, test_size: float, seed: int, cv_folds: int
) -> tuple[list[str], list[str]]:
    """Stratified train/test split. Returns the sample IDs of each set."""
    counts = y.value_counts()
    try:
        train_pos, test_pos = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            stratify=y.to_numpy(),
            random_state=seed,
        )
    except ValueError as exc:
        raise QCError(f"could not split the data: {exc}") from exc

    train_counts = y.iloc[train_pos].value_counts()
    for label in (0, 1):
        n_train = int(train_counts.get(label, 0))
        if n_train < cv_folds:
            raise QCError(
                f"the training set has only {n_train} samples of label {label} "
                f"(total {int(counts.get(label, 0))}); cv_folds={cv_folds} needs at least "
                f"{cv_folds} training samples per label. Use more samples, a smaller "
                "cv_folds, or a smaller test_size."
            )

    ids = y.index
    return [str(ids[i]) for i in train_pos], [str(ids[i]) for i in test_pos]


def _group_counts(y: pd.Series, config: Config) -> str:
    total = len(y)
    parts = []
    for label, name in ((0, config.group0_name), (1, config.group1_name)):
        n = int((y == label).sum())
        parts.append(f"{name} (label {label}): {n} ({100.0 * n / total:.1f}%)")
    return "; ".join(parts)


def _null_summary_by_group(X: pd.DataFrame, y: pd.Series, config: Config) -> list[str]:
    lines = []
    for label, name in ((0, config.group0_name), (1, config.group1_name)):
        block = X.loc[y == label]
        cells = block.size
        nulls = int(block.isna().sum().sum())
        pct = 100.0 * nulls / cells if cells else 0.0
        lines.append(f"  {name}: {nulls} of {cells} values missing ({pct:.1f}%)")
    return lines


def overview_lines(config: Config, dataset: Dataset, filt: FilterResult) -> list[str]:
    """Data overview: samples, missing values, feature filter. Used by stage 2 and --validate-only."""
    X, y = dataset.X, dataset.y
    total_cells = X.size
    total_nulls = int(X.isna().sum().sum())
    features_with_nulls = int((X.isna().sum() > 0).sum())
    removed = filt.table.loc[~filt.table["kept"]]
    return [
        f"Input: {config.resolved_input()}",
        f"Data type: {config.data_type}",
        "",
        f"Samples: {dataset.n_samples}",
        f"  {_group_counts(y, config)}",
        "",
        f"Features before filter: {dataset.n_features}",
        f"Missing values: {total_nulls} of {total_cells} cells "
        f"({100.0 * total_nulls / total_cells:.1f}%)",
        f"Features with at least one missing value: {features_with_nulls}",
        "Missing values by group:",
        *_null_summary_by_group(X, y, config),
        "",
        f"Feature filter: keep features with at least {config.min_filled_percent:g}% "
        "of samples filled in. Computed on all samples; labels are not used.",
        f"Features removed: {len(removed)}",
        *[
            f"  {row.feature}: {row.null_percent:.2f}% missing ({row.reason})"
            for row in removed.itertuples()
        ],
        f"Features kept: {len(filt.kept)}",
    ]


def build_report(
    config: Config,
    dataset: Dataset,
    filt: FilterResult,
    train_ids: list[str],
    test_ids: list[str],
) -> str:
    y = dataset.y
    in_train = y.index.astype(str).isin(train_ids)
    y_train, y_test = y[in_train], y[~in_train]

    lines = [
        "DATA SUMMARY (stage 2: QC and split)",
        "",
        *overview_lines(config, dataset, filt),
        "",
        f"Train/test split: stratified by label, test_size={config.test_size}, "
        f"seed={config.seed}",
        f"  Train: {len(train_ids)} samples; {_group_counts(y_train, config)}",
        f"  Test:  {len(test_ids)} samples; {_group_counts(y_test, config)}",
    ]
    return "\n".join(lines) + "\n"


def run_qc(config: Config, dataset: Dataset) -> QCResult:
    """Run the filter and the split, and write the QC outputs to results/01_qc/."""
    qc_dir = config.resolved_outdir() / QC_DIR_NAME
    ensure_writable(qc_dir)

    filt = filter_features(dataset.X, config.min_filled_percent)
    if not filt.kept:
        raise QCError(
            "no features left after the missing-value filter. "
            f"Lower min_filled_percent (currently {config.min_filled_percent:g})."
        )

    train_ids, test_ids = split_samples(
        dataset.y, config.test_size, config.seed, config.cv_folds
    )
    report = build_report(config, dataset, filt, train_ids, test_ids)

    removed = filt.table.loc[~filt.table["kept"], ["feature", "null_percent", "reason"]]
    filt.table.to_csv(qc_dir / "features_filter_summary.csv", index=False)
    removed.to_csv(qc_dir / "removed_features.csv", index=False)
    (qc_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (qc_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n", encoding="utf-8")
    (qc_dir / "data_summary.txt").write_text(report, encoding="utf-8")
    log.info("QC outputs written to %s", qc_dir)

    return QCResult(
        X=dataset.X[filt.kept],
        y=dataset.y,
        train_ids=train_ids,
        test_ids=test_ids,
        kept_features=filt.kept,
        removed_features=removed,
        report=report,
    )
