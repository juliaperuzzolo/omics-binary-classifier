"""External validation: apply the saved final model to new labeled data (no retraining).

Input: validation_input_data (CSV inside data/), same format as the training input:
sample_id, group (0/1) and the features. Output: 05_validation/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DECISION_THRESHOLD, Config
from .data import DataError, load_dataset
from .evaluation import FINAL_MODEL_FILE, EVAL_DIR_NAME, binary_metrics
from .interpretation_elasticnet import run_elasticnet_interpretation
from .plots import plot_confusion_matrix, plot_roc_curve
from .qc import QCResult, ensure_writable, filter_features
from .shap_interpretation import run_shap_interpretation
from .tuning import TuningResult

log = logging.getLogger(__name__)

VALIDATION_DIR_NAME = "05_validation"
INTERPRETATION_SUBDIR = "interpretation"


class ValidationStageError(RuntimeError):
    """Raised when the validation data cannot be used with the saved model."""


@dataclass
class ValidationResult:
    metrics: dict
    output_dir: Path
    missing_features: list[str]
    ignored_features: list[str]


def final_model_path(config: Config) -> Path:
    return config.resolved_outdir() / EVAL_DIR_NAME / FINAL_MODEL_FILE


def run_validation(config: Config) -> ValidationResult:
    """Apply the final model to the validation data. Raises on any problem before scoring."""
    path = config.resolved_validation_input()
    if path is None:
        raise ValidationStageError("validation_input_data is not set in params.yaml")

    model_file = final_model_path(config)
    if not model_file.is_file():
        raise ValidationStageError(
            f"final model not found: {model_file}. Run the full pipeline first."
        )
    bundle = joblib.load(model_file)

    val_dir = config.resolved_outdir() / VALIDATION_DIR_NAME
    ensure_writable(val_dir)

    try:
        ds = load_dataset(config, path=path)
    except DataError as exc:
        raise ValidationStageError(str(exc)) from exc

    kept = list(bundle["feature_names_original"])
    missing = [f for f in kept if f not in ds.X.columns]
    if missing:
        raise ValidationStageError(
            "validation data is missing features used by the model: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    ignored = [c for c in ds.X.columns if c not in kept]
    for col in ignored:
        log.warning("validation feature not used by the model, ignored: %s", col)

    # Same missing-value filter as the training data, computed on the validation samples.
    filt = filter_features(ds.X, config.min_filled_percent)
    failing = [f for f in kept if f not in filt.kept]
    if failing:
        table = filt.table.set_index("feature")
        details = "; ".join(
            f"{f} ({table.loc[f, 'filled_percent']:.1f}% filled)" for f in failing[:20]
        )
        raise ValidationStageError(
            f"features used by the model do not pass the missing-value filter "
            f"(min_filled_percent={config.min_filled_percent:g}) in the validation data: "
            f"{details}. The model cannot be applied without them."
        )
    filt.table.to_csv(val_dir / "validation_features_filter.csv", index=False)

    X = ds.X[kept]
    if not bundle.get("uses_imputer", False) and X.isna().to_numpy().any():
        raise ValidationStageError(
            "validation data has missing values, but the model was trained without "
            "missing values and has no imputer. Remove the missing values or retrain."
        )
    if ds.y.nunique() < 2:
        raise ValidationStageError("validation data must contain both labels (0 and 1)")

    X_safe = X.copy()
    X_safe.columns = bundle["feature_names_safe"]
    model = bundle["pipeline"]
    proba = model.predict_proba(X_safe)[:, 1]
    out = binary_metrics(ds.y.to_numpy(), proba, DECISION_THRESHOLD)

    # Tables and predictions
    keys = ["accuracy", "precision", "recall", "f1", "auroc", "specificity", "balanced_accuracy"]
    metrics_row = {k: round(float(out[k]), 4) for k in keys}
    pd.DataFrame([metrics_row]).T.rename(columns={0: "validation"}).to_csv(
        val_dir / "metrics_summary.csv")
    (val_dir / "metrics_summary.json").write_text(
        json.dumps({**metrics_row, "confusion": out["confusion"],
                    "threshold": DECISION_THRESHOLD}, indent=2),
        encoding="utf-8")
    pd.DataFrame([{"set": "validation", **out["confusion"]}]).to_csv(
        val_dir / "confusion_matrix.csv", index=False)
    pd.DataFrame({
        "sample_id": ds.sample_ids,
        "true_label": ds.y.to_numpy(),
        "predicted_probability": np.round(out["proba"], 6),
        "predicted_label": out["y_pred"],
    }).to_csv(val_dir / "predictions_validation.csv", index=False)

    # QC of the validation data: same filter as training, applied to validation samples
    removed_val = filt.table.loc[~filt.table["kept"]]
    qc_lines = [
        "VALIDATION DATA QC",
        f"Input: {path}",
        f"Samples: {len(ds.y)}; labels: {ds.y.value_counts().to_dict()} "
        f"(0 = {config.group0_name}, 1 = {config.group1_name})",
        f"Missing-value filter: min_filled_percent={config.min_filled_percent:g}, "
        "computed on the validation samples (same rule as training)",
        f"Features ignored (not used by the model): {len(ignored)}",
        f"Features used by the model, all passed the filter: {len(kept)}",
        f"Features removed by the filter in the validation data (not used by the model): "
        f"{len(removed_val)}",
        *[f"  {row.feature}: {row.null_percent:.2f}% missing" for row in removed_val.itertuples()],
        f"Missing values in model features: {int(X.isna().sum().sum())} of {X.size} cells",
        "Missing values by group:",
        *[
            f"  {name}: {int(X.loc[ds.y == lab].isna().sum().sum())} of "
            f"{X.loc[ds.y == lab].size} values missing"
            for lab, name in ((0, config.group0_name), (1, config.group1_name))
        ],
        "Imputation (KNN) and scaling use the training statistics only.",
    ]
    (val_dir / "validation_data_qc.txt").write_text("\n".join(qc_lines) + "\n", encoding="utf-8")

    # Checkpoint: validation samples after scaling and imputation (training statistics)
    transformed = model[:-1].transform(X_safe)
    t_df = pd.DataFrame(transformed, columns=kept, index=X.index)
    t_df.insert(0, config.label_col, ds.y.to_numpy())
    t_df.insert(0, config.id_col, X.index.astype(str))
    t_df.round(6).to_csv(val_dir / "transformed_validation_samples.csv", index=False)

    # Figures
    formats = config.figure_formats
    common = {"positive_label": config.group1_name, "negative_label": config.group0_name,
              "set_name": "validation"}
    plot_roc_curve(out["fpr"], out["tpr"], out["auroc"],
                   plot_color=config.colors.plot_color,
                   output_stem=val_dir / "roc_curve_validation", formats=formats, **common)
    cm = np.array([[out["confusion"]["TN"], out["confusion"]["FP"]],
                   [out["confusion"]["FN"], out["confusion"]["TP"]]])
    plot_confusion_matrix(cm, output_stem=val_dir / "confusion_matrix_validation",
                          formats=formats, **common)

    # Summary
    counts = ds.y.value_counts().to_dict()
    lines = [
        "EXTERNAL VALIDATION",
        f"Input: {path}",
        f"Model: {config.model}, final model trained on 100% of the training data",
        "No retraining, scaling or imputation was fit on the validation data.",
        f"Samples: {len(ds.y)} ({config.group0_name}: {counts.get(0, 0)}, "
        f"{config.group1_name}: {counts.get(1, 0)})",
        f"Features used by the model: {len(kept)}",
        f"Features ignored (not used by the model): {len(ignored)}"
        + (f" ({', '.join(ignored)})" if ignored else ""),
        f"Threshold: {DECISION_THRESHOLD}",
    ]
    (val_dir / "validation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Interpretation on the validation data
    tuning = TuningResult(
        search=None,
        best_estimator=model,
        cv_results=pd.DataFrame(),
        best_params=dict(bundle.get("best_params", {})),
        cv_auc_mean=float("nan"),
        cv_auc_std=float("nan"),
        feature_names_original=kept,
        feature_names_safe=list(bundle["feature_names_safe"]),
        uses_imputer=bool(bundle.get("uses_imputer", False)),
        checkpoint_path=model_file,
    )
    qc = QCResult(
        X=X, y=ds.y, train_ids=[], test_ids=[str(i) for i in X.index],
        kept_features=kept,
        removed_features=pd.DataFrame(columns=["feature", "null_percent", "reason"]),
        report="",
    )
    interp_dir = val_dir / INTERPRETATION_SUBDIR
    if config.model == "xgboost":
        run_shap_interpretation(config, qc, tuning, out_dir=interp_dir, set_name="validation")
    else:
        run_elasticnet_interpretation(config, tuning, out_dir=interp_dir)

    return ValidationResult(
        metrics=metrics_row,
        output_dir=val_dir,
        missing_features=missing,
        ignored_features=ignored,
    )
