"""Stage 3: sklearn pipeline and hyperparameter search on the training split only.

Pipeline (same for both models): StandardScaler -> KNNImputer -> model.
The imputer is included only when the filtered features contain missing values.
Everything that learns from data is fitted inside the search, on the training split.
"""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .artifacts import build_bundle, sanitize_feature_names, save_bundle
from .config import Config
from .qc import QCResult, ensure_writable

log = logging.getLogger(__name__)

TUNING_DIR_NAME = "02_tuning"
EVALUATION_MODEL_FILE = "evaluation_model.joblib"


class TuningError(RuntimeError):
    """Raised by the pre-flight checks before the search starts."""


@dataclass
class TuningResult:
    search: Any
    best_estimator: Pipeline
    cv_results: pd.DataFrame
    best_params: dict[str, Any]
    cv_auc_mean: float
    cv_auc_std: float
    feature_names_original: list[str]
    feature_names_safe: list[str]
    uses_imputer: bool
    checkpoint_path: Path


def build_pipeline(config: Config, uses_imputer: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if uses_imputer:
        steps.append(("imputer", KNNImputer()))
    if config.model == "elasticnet":
        estimator = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=config.elasticnet.max_iter,
            random_state=config.seed,
        )
    else:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=config.seed,
            n_jobs=1,
        )
    steps.append(("model", estimator))
    return Pipeline(steps)


def search_space(config: Config) -> dict[str, list[Any]]:
    """Hyperparameter lists for the chosen model, with the `model__` prefix for the pipeline."""
    if config.model == "elasticnet":
        return {
            "model__C": config.elasticnet.C,
            "model__l1_ratio": config.elasticnet.l1_ratio,
        }
    xgb = config.xgboost
    return {
        "model__n_estimators": xgb.n_estimators,
        "model__max_depth": xgb.max_depth,
        "model__learning_rate": xgb.learning_rate,
        "model__subsample": xgb.subsample,
        "model__colsample_bytree": xgb.colsample_bytree,
        "model__min_child_weight": xgb.min_child_weight,
        "model__reg_lambda": xgb.reg_lambda,
    }


def preflight(config: Config, qc: QCResult, tuning_dir: Path) -> None:
    """Checks that fail early, before the search."""
    ensure_writable(tuning_dir)

    if config.model == "xgboost":
        try:
            importlib.import_module("shap")
        except ImportError as exc:
            raise TuningError(
                f"model is xgboost, but the 'shap' package cannot be imported: {exc}"
            ) from exc

    train_X = qc.X.loc[qc.train_ids]
    all_missing = train_X.columns[train_X.isna().all()].tolist()
    if all_missing:
        raise TuningError(
            "these features have no values in the training set, so they cannot be "
            f"imputed: {', '.join(all_missing[:10])}. Raise min_filled_percent or "
            "use a smaller test_size."
        )


def run_tuning(config: Config, qc: QCResult) -> TuningResult:
    tuning_dir = config.resolved_outdir() / TUNING_DIR_NAME
    preflight(config, qc, tuning_dir)

    # Safe names for XGBoost, with a mapping back to the original names for outputs.
    original = list(qc.kept_features)
    safe, _ = sanitize_feature_names(original)
    X_all = qc.X.copy()
    X_all.columns = safe

    X_train = X_all.loc[qc.train_ids]
    y_train = qc.y.loc[qc.train_ids].to_numpy()

    uses_imputer = bool(qc.X.isna().to_numpy().any())
    pipeline = build_pipeline(config, uses_imputer)
    cv = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)

    if config.model == "elasticnet":
        search = GridSearchCV(
            pipeline,
            search_space(config),
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=config.n_jobs,
            error_score="raise",
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            search_space(config),
            n_iter=config.n_combinations(),
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=config.n_jobs,
            random_state=config.seed,
            error_score="raise",
        )

    log.info("Starting search: %s", config.search_summary())
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    best_idx = search.best_index_
    cv_mean = float(cv_results.loc[best_idx, "mean_test_score"])
    cv_std = float(cv_results.loc[best_idx, "std_test_score"])
    best_params = {k: _plain(v) for k, v in search.best_params_.items()}

    # CHECKPOINT 1: full CV table and the evaluation model (best_estimator_, fit on the training split)
    tuning_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(tuning_dir / "cv_results.csv", index=False)
    checkpoint_path = tuning_dir / EVALUATION_MODEL_FILE
    bundle = build_bundle(
        config=config,
        estimator=search.best_estimator_,
        feature_names_original=original,
        feature_names_safe=safe,
        removed_features=[],
        trained_on=f"training split only ({len(qc.train_ids)} samples, "
                   f"{100 - round(100 * config.test_size)}%)",
        best_params=best_params,
        note=(
            "EVALUATION MODEL. Fit on the training split only. Use it to recompute test "
            "metrics, confusion matrices and figures. Its test metrics are the real "
            "results of this run. It is not the model for external validation."
        ),
    )
    bundle["uses_imputer"] = uses_imputer
    save_bundle(bundle, checkpoint_path)

    # CHECKPOINT 1b: every sample after standardization (and imputation, if used).
    # Scaler and imputer statistics come from the training split only.
    transformed = search.best_estimator_[:-1].transform(X_all)
    transformed_df = pd.DataFrame(transformed, columns=original, index=X_all.index)
    transformed_df.insert(0, config.label_col, qc.y.loc[X_all.index].to_numpy())
    transformed_df.insert(0, config.id_col, X_all.index.astype(str))
    transformed_df.round(6).to_csv(tuning_dir / "transformed_samples.csv", index=False)

    summary = {
        "model": config.model,
        "best_params": best_params,
        "cv_folds": config.cv_folds,
        "cv_auc_mean": cv_mean,
        "cv_auc_std": cv_std,
        "n_combinations": config.n_combinations(),
        "uses_imputer": uses_imputer,
    }
    (tuning_dir / "best_params.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Checkpoint 1 saved: %s", checkpoint_path)

    return TuningResult(
        search=search,
        best_estimator=search.best_estimator_,
        cv_results=cv_results,
        best_params=best_params,
        cv_auc_mean=cv_mean,
        cv_auc_std=cv_std,
        feature_names_original=original,
        feature_names_safe=safe,
        uses_imputer=uses_imputer,
        checkpoint_path=checkpoint_path,
    )


def _plain(value: Any) -> Any:
    """Convert numpy scalars to plain Python values for JSON and joblib bundles."""
    if isinstance(value, np.generic):
        return value.item()
    return value
