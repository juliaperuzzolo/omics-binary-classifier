"""Stage 4: evaluation on train and test, figures, checkpoint 2 and the final 100% model.

Evaluation model: the search's best estimator, fit on the training split (70% or 80%).
Final model: the same pipeline and hyperparameters, refit on 100% of the samples.
Its metrics are not reported; it has no independent test set (external validation).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .artifacts import build_bundle, save_bundle
from .config import DECISION_THRESHOLD, Config
from .plots import plot_confusion_matrix, plot_roc_curve
from .qc import QCResult, ensure_writable
from .tuning import TuningResult

EVAL_DIR_NAME = "03_final_model"
EVALUATION_MODEL_FILE = "evaluation_model.joblib"
FINAL_MODEL_FILE = "final_model.joblib"

FINAL_MODEL_NOTE = (
    "FINAL MODEL for external validation. Fit on 100% of the samples (training + test) "
    "with the selected hyperparameters. The test metrics reported by this run come from "
    "the evaluation model (training split only). This model has no independent test set; "
    "validation must be done with external data."
)


@dataclass
class EvaluationResult:
    metrics: dict[str, dict[str, float]]   # {"train": {...}, "test": {...}}
    confusion: dict[str, dict[str, int]]   # {"train": {TN, FP, FN, TP}, "test": {...}}
    output_dir: Path
    final_model_path: Path


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """All metrics for one set. Labels: 1 = positive (target group), 0 = negative."""
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, proba),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
        "proba": proba,
    }


def _predictions_frame(ids: list[str], y: np.ndarray, out: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ids,
            "true_label": y,
            "predicted_probability": np.round(out["proba"], 6),
            "predicted_label": out["y_pred"],
        }
    )


def run_evaluation(config: Config, qc: QCResult, tuning: TuningResult) -> EvaluationResult:
    eval_dir = config.resolved_outdir() / EVAL_DIR_NAME
    ensure_writable(eval_dir)

    safe = tuning.feature_names_safe
    X_all = qc.X.copy()
    X_all.columns = safe
    y_all = qc.y

    X_train, y_train = X_all.loc[qc.train_ids], y_all.loc[qc.train_ids].to_numpy()
    X_test, y_test = X_all.loc[qc.test_ids], y_all.loc[qc.test_ids].to_numpy()

    model = tuning.best_estimator  # evaluation model: fit on the training split
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    out_train = binary_metrics(y_train, proba_train, DECISION_THRESHOLD)
    out_test = binary_metrics(y_test, proba_test, DECISION_THRESHOLD)

    # ----- Metrics table (train, test, and CV AUROC of the best combination) -----
    keys = ["accuracy", "precision", "recall", "f1", "auroc", "specificity", "balanced_accuracy"]
    labels = {
        "accuracy": "Accuracy", "precision": "Precision", "recall": "Sensitivity (recall)",
        "f1": "F1", "auroc": "AUROC", "specificity": "Specificity",
        "balanced_accuracy": "Balanced accuracy",
    }
    rows = []
    for key in keys:
        row = {"metric": labels[key], "train": out_train[key], "test": out_test[key]}
        if key == "auroc":
            row["cv_mean_std"] = f"{tuning.cv_auc_mean:.3f} +/- {tuning.cv_auc_std:.3f}"
        else:
            row["cv_mean_std"] = ""
        rows.append(row)
    metrics_table = pd.DataFrame(rows)
    metrics_table.round(4).to_csv(eval_dir / "metrics_summary.csv", index=False)

    confusion_table = pd.DataFrame(
        [
            {"set": "train", **out_train["confusion"]},
            {"set": "test", **out_test["confusion"]},
        ]
    )
    confusion_table.to_csv(eval_dir / "confusion_matrices.csv", index=False)

    # ----- Predictions -----
    _predictions_frame(qc.train_ids, y_train, out_train).to_csv(
        eval_dir / "predictions_train.csv", index=False)
    _predictions_frame(qc.test_ids, y_test, out_test).to_csv(
        eval_dir / "predictions_test.csv", index=False)

    # ----- Figures (one per file, in every requested format) -----
    formats = config.figure_formats
    common = {
        "positive_label": config.group1_name,
        "negative_label": config.group0_name,
    }
    for name, out in (("train", out_train), ("test", out_test)):
        plot_roc_curve(
            out["fpr"], out["tpr"], out["auroc"],
            plot_color=config.colors.plot_color, set_name=name,
            output_stem=eval_dir / f"roc_curve_{name}", formats=formats, **common,
        )
        cm = np.array([[out["confusion"]["TN"], out["confusion"]["FP"]],
                       [out["confusion"]["FN"], out["confusion"]["TP"]]])
        plot_confusion_matrix(
            cm, set_name=name,
            output_stem=eval_dir / f"confusion_matrix_{name}", formats=formats, **common,
        )

    # ----- Checkpoint 2 summary -----
    summary = {
        "threshold": DECISION_THRESHOLD,
        "cv_auroc_mean": tuning.cv_auc_mean,
        "cv_auroc_std": tuning.cv_auc_std,
        "train": {k: out_train[k] for k in keys} | {"confusion": out_train["confusion"]},
        "test": {k: out_test[k] for k in keys} | {"confusion": out_test["confusion"]},
    }
    (eval_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ----- Evaluation model, kept separately in the final folder -----
    shutil.copy2(tuning.checkpoint_path, eval_dir / EVALUATION_MODEL_FILE)

    # ----- CHECKPOINT 3: final model refit on 100% of the samples -----
    final_model = clone(model)
    final_model.fit(X_all, y_all.to_numpy())
    kept = tuning.feature_names_original
    removed = qc.removed_features
    bundle = build_bundle(
        config=config,
        estimator=final_model,
        feature_names_original=kept,
        feature_names_safe=safe,
        removed_features=[
            f"{row.feature} ({row.null_percent:.2f}% missing)"
            for row in removed.itertuples()
        ],
        trained_on=f"100% of samples ({len(qc.y)} samples: training + test)",
        best_params=tuning.best_params,
        note=FINAL_MODEL_NOTE,
    )
    bundle["uses_imputer"] = tuning.uses_imputer
    final_path = save_bundle(bundle, eval_dir / FINAL_MODEL_FILE)

    return EvaluationResult(
        metrics={"train": out_train, "test": out_test},
        confusion={"train": out_train["confusion"], "test": out_test["confusion"]},
        output_dir=eval_dir,
        final_model_path=final_path,
    )
