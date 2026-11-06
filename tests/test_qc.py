import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.data import load_dataset
from pipeline.qc import QCError, filter_features, run_qc, split_samples
from tests.conftest import TEST_CSV


def _frame():
    # 10 samples. f_full: 100% filled; f_80: 80% filled; f_50: 50%; f_10: 10%; f_none: 0%.
    values = {
        "f_full": [1.0] * 10,
        "f_80": [1.0] * 8 + [np.nan] * 2,
        "f_50": [1.0] * 5 + [np.nan] * 5,
        "f_10": [1.0] + [np.nan] * 9,
        "f_none": [np.nan] * 10,
    }
    return pd.DataFrame(values, index=[f"S{i}" for i in range(10)])


def test_filter_keeps_features_at_or_above_minimum():
    res = filter_features(_frame(), min_filled_percent=50)
    assert res.kept == ["f_full", "f_80", "f_50"]


def test_filter_80_keeps_only_features_filled_at_least_80_percent():
    res = filter_features(_frame(), min_filled_percent=80)
    assert res.kept == ["f_full", "f_80"]


def test_filter_0_keeps_everything_with_at_least_one_value():
    res = filter_features(_frame(), min_filled_percent=0)
    assert res.kept == ["f_full", "f_80", "f_50", "f_10"]


def test_feature_with_no_values_is_always_removed():
    res = filter_features(_frame(), min_filled_percent=0)
    row = res.table.set_index("feature").loc["f_none"]
    assert not row["kept"] and row["reason"] == "no values in any sample"


def test_removed_table_reports_null_percent():
    res = filter_features(_frame(), min_filled_percent=80)
    table = res.table.set_index("feature")
    assert table.loc["f_10", "null_percent"] == 90.0
    assert table.loc["f_50", "null_percent"] == 50.0


def test_split_is_stratified_and_reproducible():
    y = pd.Series([0] * 40 + [1] * 20, index=[f"S{i}" for i in range(60)])
    train1, test1 = split_samples(y, test_size=0.3, seed=42, cv_folds=5)
    train2, test2 = split_samples(y, test_size=0.3, seed=42, cv_folds=5)
    assert (train1, test1) == (train2, test2)
    assert set(train1).isdisjoint(test1)
    assert len(train1) + len(test1) == 60
    assert sum(y[t] == 1 for t in test1) == 6  # 30% of 20 cancer samples


def test_split_rejects_too_few_training_samples_for_cv():
    y = pd.Series([0] * 20 + [1] * 4, index=[f"S{i}" for i in range(24)])
    with pytest.raises(QCError, match="label 1"):
        split_samples(y, test_size=0.3, seed=42, cv_folds=5)


def test_run_qc_writes_outputs_on_test_data(write_config, tmp_path):
    cfg_path = write_config(
        f"input: {TEST_CSV}\nlabel_col: group\ngroup0_name: Control\n"
        f"group1_name: Cancer\noutdir: {tmp_path / 'out'}\nmin_filled_percent: 50\n"
    )
    cfg = load_config(cfg_path)
    ds = load_dataset(cfg)
    qc = run_qc(cfg, ds)
    qc_dir = tmp_path / "out" / "01_qc"
    for name in ("data_summary.txt", "features_filter_summary.csv",
                 "removed_features.csv", "train_ids.txt", "test_ids.txt"):
        assert (qc_dir / name).is_file()
    assert set(qc.train_ids).isdisjoint(qc.test_ids)
    assert "ABRACL" in qc.removed_features["feature"].tolist()


def test_unwritable_outdir_is_reported_before_work(write_config, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a folder", encoding="utf-8")
    cfg_path = write_config(
        f"input: {TEST_CSV}\nlabel_col: group\ngroup0_name: Control\n"
        f"group1_name: Cancer\noutdir: {blocker}\n"
    )
    cfg = load_config(cfg_path)
    ds = load_dataset(cfg)
    with pytest.raises(QCError, match="not writable"):
        run_qc(cfg, ds)
