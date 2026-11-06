import shutil

import pandas as pd
import pytest

from pipeline.__main__ import main
from pipeline.config import ConfigError, load_config
from pipeline.validation import ValidationStageError, run_validation
from tests.conftest import TEST_CSV

SMALL_GRID = "elasticnet:\n  C: [1]\n  l1_ratio: [0.5]\ncv_folds: 3\n"


def _project(tmp_path, validation_line):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(TEST_CSV, data / "train.csv")
    df = pd.read_csv(TEST_CSV)
    # Test-only stand-in for new data: rows from both groups, same format.
    controls = df.index[df["group"] == 0][:10]
    cases = df.index[df["group"] == 1][:10]
    df.loc[list(controls) + list(cases)].to_csv(data / "validation.csv", index=False)
    params = tmp_path / "params.yaml"
    params.write_text(
        "input: data/train.csv\nlabel_col: group\ngroup0_name: Control\n"
        "group1_name: Cancer\n"
        f"outdir: {tmp_path / 'results'}\n"
        f"{validation_line}\n{SMALL_GRID}",
        encoding="utf-8",
    )
    return params, data


def test_no_validation_stops_after_final_model(tmp_path):
    params, _ = _project(tmp_path, "validation_input_data: null")
    assert main(["--config", str(params)]) == 0
    assert (tmp_path / "results" / "03_final_model" / "final_model.joblib").is_file()
    assert not (tmp_path / "results" / "05_validation").exists()


def test_validation_runs_and_writes_outputs(tmp_path):
    params, _ = _project(tmp_path, "validation_input_data: data/validation.csv")
    assert main(["--config", str(params)]) == 0
    val = tmp_path / "results" / "05_validation"
    for name in ("metrics_summary.csv", "metrics_summary.json", "confusion_matrix.csv",
                 "predictions_validation.csv", "validation_summary.txt",
                 "roc_curve_validation.png", "confusion_matrix_validation.png",
                 "validation_data_qc.txt", "transformed_validation_samples.csv"):
        assert (val / name).is_file(), name
    assert (val / "interpretation" / "coefficients_all.csv").is_file()


def test_missing_model_feature_is_an_error(tmp_path):
    params, data = _project(tmp_path, "validation_input_data: data/validation.csv")
    # Train first (the final model is needed), then drop a feature from the validation file.
    assert main(["--config", str(params)]) == 0
    df = pd.read_csv(data / "validation.csv")
    dropped = df.columns[2]
    df.drop(columns=[dropped]).to_csv(data / "validation.csv", index=False)
    cfg = load_config(params)
    with pytest.raises(ValidationStageError, match="missing features") as exc:
        run_validation(cfg)
    assert dropped in str(exc.value)


def test_extra_feature_is_ignored_with_warning(tmp_path):
    params, data = _project(tmp_path, "validation_input_data: data/validation.csv")
    assert main(["--config", str(params)]) == 0
    df = pd.read_csv(data / "validation.csv")
    df["EXTRA_FEATURE"] = 1.5
    df.to_csv(data / "validation.csv", index=False)
    res = run_validation(load_config(params))
    assert "EXTRA_FEATURE" in res.ignored_features


def test_validation_path_outside_data_is_rejected(tmp_path):
    params, _ = _project(tmp_path, "validation_input_data: elsewhere/validation.csv")
    with pytest.raises(ConfigError, match="inside the data/ folder"):
        load_config(params)


def test_validate_only_prints_data_overview(tmp_path, capsys):
    params, _ = _project(tmp_path, "validation_input_data: null")
    assert main(["--config", str(params), "--validate-only"]) == 0
    out = capsys.readouterr().out
    assert "DATA SUMMARY" in out and "Features kept" in out
    assert not (tmp_path / "results").exists()  # nothing written


def test_model_feature_failing_null_filter_in_validation_is_an_error(tmp_path):
    params, data = _project(tmp_path, "validation_input_data: data/validation.csv")
    assert main(["--config", str(params)]) == 0
    df = pd.read_csv(data / "validation.csv")
    feature = df.columns[2]  # a feature the model uses
    df.loc[df.index[:-2], feature] = float("nan")  # only 2 of 20 samples keep a value
    df.to_csv(data / "validation.csv", index=False)
    with pytest.raises(ValidationStageError, match="missing-value filter") as exc:
        run_validation(load_config(params))
    assert feature in str(exc.value)
