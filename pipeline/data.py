"""Read the input CSV and validate it before any computation.

Every check runs before the heavy steps. All problems found are reported
together, so the user can fix them in one pass.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


class DataError(ValueError):
    """Raised when the input CSV fails validation. The message lists every problem."""


@dataclass
class Dataset:
    """Validated data. `X` holds the feature columns only; `y` holds 0/1 labels."""

    sample_ids: pd.Index
    X: pd.DataFrame
    y: pd.Series
    n_samples: int
    n_features: int


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration:
            return []


def load_dataset(config: Config, path: Path | None = None) -> Dataset:
    """Read a CSV (default: config.input) and validate it. Raises DataError listing all problems."""
    path = path if path is not None else config.resolved_input()
    id_col, label_col = config.id_col, config.label_col

    if not path.is_file():
        raise DataError(f"input file not found: {path}")

    header = _read_header(path)
    if not header:
        raise DataError(f"input file has no header row: {path}")

    problems: list[str] = []

    # Duplicate or empty column names (checked on the raw header, before pandas renames them)
    duplicated = sorted({name for name in header if header.count(name) > 1})
    if duplicated:
        problems.append(f"duplicate column names: {', '.join(duplicated)}")
    if any(name.strip() == "" for name in header):
        problems.append("the header contains an empty column name")

    missing = [col for col in (id_col, label_col) if col not in header]
    if missing:
        problems.append(f"required column(s) not found in the CSV: {', '.join(missing)}")
        raise DataError(_join(path, problems))

    df = pd.read_csv(path, dtype=None)
    if df.empty:
        raise DataError(f"input file has no data rows: {path}")

    # Sample IDs: no missing, no duplicates
    ids = df[id_col]
    if ids.isna().any():
        problems.append(f"'{id_col}' has missing values")
    dup_ids = ids[ids.duplicated(keep=False)].astype(str).unique().tolist()
    if dup_ids:
        shown = ", ".join(dup_ids[:10]) + (" ..." if len(dup_ids) > 10 else "")
        problems.append(f"duplicated sample IDs in '{id_col}': {shown}")

    # Labels: no missing, only 0 and 1, both classes present
    labels = df[label_col]
    if labels.isna().any():
        problems.append(f"'{label_col}' has missing values")
    else:
        bad = sorted({str(v) for v in labels.unique() if not _is_zero_or_one(v)})
        if bad:
            problems.append(
                f"'{label_col}' must contain only 0 and 1; found: {', '.join(bad[:10])}"
            )
        elif labels.nunique() < 2:
            problems.append(f"'{label_col}' has only one class; both 0 and 1 are required")

    # Features: everything except the ID and label columns must be numeric
    feature_cols = [c for c in header if c not in (id_col, label_col)]
    if not feature_cols:
        problems.append("no feature columns found")
    non_numeric = [c for c in feature_cols if not _is_numeric_column(df[c])]
    if non_numeric:
        shown = ", ".join(non_numeric[:10]) + (" ..." if len(non_numeric) > 10 else "")
        problems.append(
            "non-numeric columns found (only sample ID, label and numeric features "
            f"are allowed): {shown}"
        )

    # Infinite values are not valid input
    numeric_cols = [c for c in feature_cols if c not in non_numeric]
    if numeric_cols:
        values = df[numeric_cols].to_numpy(dtype=float)
        if np.isinf(values).any():
            problems.append("feature columns contain infinite values")

    if problems:
        raise DataError(_join(path, problems))

    X = df[feature_cols].astype(float)
    y = labels.astype(int)
    sample_ids = pd.Index(ids.astype(str), name=id_col)
    X.index = sample_ids
    y.index = sample_ids

    counts = y.value_counts().to_dict()
    log.info("Input validated: %d samples, %d features", len(df), len(feature_cols))
    log.info("Samples per group: %s=%d, %s=%d",
             config.group0_name, counts.get(0, 0), config.group1_name, counts.get(1, 0))

    return Dataset(
        sample_ids=sample_ids,
        X=X,
        y=y,
        n_samples=len(df),
        n_features=len(feature_cols),
    )


def _is_zero_or_one(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number in (0.0, 1.0)


def _is_numeric_column(series: pd.Series) -> bool:
    # Booleans are rejected on purpose: they are not valid numeric features here.
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _join(path: Path, problems: list[str]) -> str:
    return f"invalid input {path}:\n  - " + "\n  - ".join(problems)
