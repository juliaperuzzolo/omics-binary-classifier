"""--only-interpret: rebuild the QC split and the evaluation model from saved checkpoints.

No tuning or refit happens here. The evaluation model and the split IDs were written
by an earlier run into the output folder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd

from .config import Config
from .data import Dataset, load_dataset
from .qc import QCResult
from .tuning import EVALUATION_MODEL_FILE, TUNING_DIR_NAME, TuningResult
from .qc import QC_DIR_NAME


class ResumeError(RuntimeError):
    """Raised when a required checkpoint file is missing."""


def _read_ids(path) -> list[str]:
    if not path.is_file():
        raise ResumeError(f"checkpoint missing: {path}. Run the full pipeline first.")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_checkpoints(config: Config) -> tuple[Dataset, QCResult, TuningResult]:
    outdir = config.resolved_outdir()
    qc_dir = outdir / QC_DIR_NAME
    tune_dir = outdir / TUNING_DIR_NAME
    checkpoint = tune_dir / EVALUATION_MODEL_FILE
    if not checkpoint.is_file():
        raise ResumeError(f"checkpoint missing: {checkpoint}. Run the full pipeline first.")

    bundle = joblib.load(checkpoint)
    train_ids = _read_ids(qc_dir / "train_ids.txt")
    test_ids = _read_ids(qc_dir / "test_ids.txt")
    kept = bundle["feature_names_original"]

    dataset = load_dataset(config)
    X = dataset.X[kept]
    qc = QCResult(
        X=X,
        y=dataset.y,
        train_ids=train_ids,
        test_ids=test_ids,
        kept_features=kept,
        removed_features=pd.read_csv(qc_dir / "removed_features.csv"),
        report="",
    )
    best = json.loads((tune_dir / "best_params.json").read_text(encoding="utf-8"))
    tuning = TuningResult(
        search=None,
        best_estimator=bundle["pipeline"],
        cv_results=pd.DataFrame(),
        best_params=best["best_params"],
        cv_auc_mean=float(best["cv_auc_mean"]),
        cv_auc_std=float(best["cv_auc_std"]),
        feature_names_original=kept,
        feature_names_safe=bundle["feature_names_safe"],
        uses_imputer=bool(bundle.get("uses_imputer", False)),
        checkpoint_path=checkpoint,
    )
    return dataset, qc, tuning
