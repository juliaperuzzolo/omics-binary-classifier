import logging

import pytest

from pipeline.config import ConfigError, load_config
from tests.conftest import EXAMPLE_PARAMS


def test_example_params_loads_with_expected_defaults():
    cfg = load_config(EXAMPLE_PARAMS)
    assert cfg.model == "elasticnet"
    assert cfg.cv_folds == 5
    assert cfg.test_size == 0.3
    assert cfg.colors.group1 == "#d62728"
    assert cfg.search_summary() == "30 combinations x 5 folds = 150 fits"


def test_unknown_key_is_rejected(write_config, minimal_yaml):
    path = write_config(minimal_yaml + "sead: 42\n")
    with pytest.raises(ConfigError, match="sead"):
        load_config(path)


def test_misspelled_model_is_rejected(write_config, minimal_yaml):
    path = write_config(minimal_yaml + "model: elasticnett\n")
    with pytest.raises(ConfigError, match="model"):
        load_config(path)


def test_invalid_hex_color_is_rejected(write_config, minimal_yaml):
    path = write_config(minimal_yaml + "colors:\n  group0: \"blue\"\n")
    with pytest.raises(ConfigError, match="colors.group0"):
        load_config(path)


def test_scalar_hyperparameter_is_converted_to_list_with_warning(
    write_config, minimal_yaml, caplog
):
    path = write_config(minimal_yaml + "elasticnet:\n  C: 1\n")
    with caplog.at_level(logging.WARNING, logger="pipeline.config"):
        cfg = load_config(path)
    assert cfg.elasticnet.C == [1.0]
    assert any("'C' is a scalar" in r.message for r in caplog.records)


def test_missing_required_key_is_reported(write_config, tmp_path):
    path = write_config(
        f"input: {tmp_path / 'data.csv'}\ngroup0_name: Control\ngroup1_name: Cancer\n"
    )
    with pytest.raises(ConfigError, match="label_col"):
        load_config(path)


def test_deleted_sections_fall_back_to_defaults(write_config, minimal_yaml):
    path = write_config(minimal_yaml)  # no colors, no elasticnet, no shap sections
    cfg = load_config(path)
    assert cfg.colors.plot_color == "#79c8b2"
    assert len(cfg.elasticnet.C) == 6
    assert cfg.shap.top_k == 10


def test_test_size_out_of_range_is_rejected(write_config, minimal_yaml):
    path = write_config(minimal_yaml + "test_size: 0.9\n")
    with pytest.raises(ConfigError, match="test_size"):
        load_config(path)


def test_xgboost_search_size_is_capped_by_n_iter(write_config, minimal_yaml):
    path = write_config(minimal_yaml + "model: xgboost\nxgboost:\n  n_iter: 7\n")
    cfg = load_config(path)
    assert cfg.search_space_size() == 4 * 4 * 4 * 3 * 4 * 3 * 3
    assert cfg.n_combinations() == 7


def test_empty_file_is_rejected(write_config):
    with pytest.raises(ConfigError, match="empty"):
        load_config(write_config(""))


def test_cv_folds_accepts_3_5_10_only(write_config, minimal_yaml):
    for folds in (3, 5, 10):
        cfg = load_config(write_config(minimal_yaml + f"cv_folds: {folds}\n"))
        assert cfg.cv_folds == folds
    with pytest.raises(ConfigError, match="cv_folds"):
        load_config(write_config(minimal_yaml + "cv_folds: 4\n"))


def test_min_filled_percent_range(write_config, minimal_yaml):
    cfg = load_config(write_config(minimal_yaml + "min_filled_percent: 80\n"))
    assert cfg.min_filled_percent == 80
    with pytest.raises(ConfigError, match="min_filled_percent"):
        load_config(write_config(minimal_yaml + "min_filled_percent: 120\n"))


def test_old_max_null_fraction_key_is_rejected(write_config, minimal_yaml):
    with pytest.raises(ConfigError, match="max_null_fraction"):
        load_config(write_config(minimal_yaml + "max_null_fraction: 0.5\n"))
