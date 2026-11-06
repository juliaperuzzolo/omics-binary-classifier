import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline.artifacts import sanitize_feature_names
from pipeline.config import load_config
from pipeline.data import load_dataset
from pipeline.qc import run_qc
from pipeline.tuning import build_pipeline, run_tuning
from tests.conftest import TEST_CSV


def _setup(write_config, tmp_path, extra=""):
    cfg_path = write_config(
        f"input: {TEST_CSV}\nlabel_col: group\ngroup0_name: Control\n"
        f"group1_name: Cancer\noutdir: {tmp_path / 'out'}\n" + extra
    )
    cfg = load_config(cfg_path)
    qc = run_qc(cfg, load_dataset(cfg))
    return cfg, qc


def test_sanitize_feature_names_is_safe_and_reversible():
    safe, mapping = sanitize_feature_names(["miR-21[5p]", "IL6<receptor>", "A", "A_1", "A"])
    assert all(not any(c in s for c in "[]<") for s in safe)
    assert len(set(safe)) == len(safe)
    assert [mapping[s] for s in safe] == ["miR-21[5p]", "IL6<receptor>", "A", "A_1", "A"]


def test_scaler_and_imputer_are_inside_pipeline_and_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(10, 2, size=(30, 3)))
    X_train.iloc[0, 0] = np.nan
    X_test = pd.DataFrame(rng.normal(100, 50, size=(10, 3)))  # very different on purpose
    y = pd.Series([0, 1] * 15)

    from xgboost import XGBClassifier  # noqa: F401  (import check)
    from pipeline.config import Config

    cfg = Config.model_validate({"input": "x.csv", "label_col": "g",
                                 "group0_name": "a", "group1_name": "b"})
    pipe = build_pipeline(cfg, uses_imputer=True)
    assert [name for name, _ in pipe.steps] == ["scaler", "imputer", "model"]
    pipe.set_params(model__C=1.0, model__l1_ratio=0.5)  # normally set by the search
    pipe.fit(X_train, y)
    scaler = pipe.named_steps["scaler"]
    # Scaler statistics come from the training data only.
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy(), equal_nan=True)
    pipe.predict_proba(X_test)  # transforming test data must not refit anything
    assert np.allclose(scaler.mean_, X_train.mean().to_numpy(), equal_nan=True)


def test_imputer_is_skipped_when_no_missing_values():
    from pipeline.config import Config

    cfg = Config.model_validate({"input": "x.csv", "label_col": "g",
                                 "group0_name": "a", "group1_name": "b"})
    pipe = build_pipeline(cfg, uses_imputer=False)
    assert [name for name, _ in pipe.steps] == ["scaler", "model"]


def test_elasticnet_checkpoint_is_loadable_and_predicts(write_config, tmp_path):
    cfg, qc = _setup(write_config, tmp_path)
    result = run_tuning(cfg, qc)
    bundle = joblib.load(result.checkpoint_path)
    assert bundle["threshold"] == 0.5
    assert bundle["trained_on"].startswith("training split only")
    assert "EVALUATION MODEL" in bundle["note"]
    assert bundle["feature_names_original"] == qc.kept_features
    X_test = qc.X.loc[qc.test_ids]
    X_test.columns = bundle["feature_names_safe"]
    proba = bundle["pipeline"].predict_proba(X_test)[:, 1]
    assert proba.shape == (len(qc.test_ids),)
    assert (result.cv_results["rank_test_score"] == 1).sum() >= 1
    assert "split0_test_score" in result.cv_results.columns  # per-fold scores


def test_xgboost_smoke_run(write_config, tmp_path):
    pytest.importorskip("shap")
    cfg, qc = _setup(
        write_config, tmp_path,
        extra="model: xgboost\ncv_folds: 3\nxgboost:\n  n_iter: 3\n",
    )
    result = run_tuning(cfg, qc)
    assert len(result.cv_results) == 3
    assert result.best_params


def test_transformed_samples_checkpoint_is_written(write_config, tmp_path):
    cfg, qc = _setup(write_config, tmp_path)
    run_tuning(cfg, qc)
    table = pd.read_csv(tmp_path / "out" / "02_tuning" / "transformed_samples.csv")
    assert list(table.columns[:2]) == ["sample_id", "group"]
    assert list(table.columns[2:]) == qc.kept_features
    assert len(table) == len(qc.y)
    assert not table[qc.kept_features].isna().any().any()  # imputed (KNN present in test data)
