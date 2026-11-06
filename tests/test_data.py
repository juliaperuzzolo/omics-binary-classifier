import pytest

from pipeline.config import load_config
from pipeline.data import DataError, load_dataset
from tests.conftest import TEST_CSV


def _config_for(write_config, csv_path):
    return load_config(
        write_config(
            f"input: {csv_path}\nlabel_col: group\ngroup0_name: Control\ngroup1_name: Cancer\n"
        )
    )


def _write_csv(tmp_path, text):
    path = tmp_path / "data.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_test_pipeline_csv_is_valid(write_config):
    # Real test file provided by the user (control = 0, cancer = 1).
    # Only structural properties are asserted, so the test survives data updates.
    ds = load_dataset(_config_for(write_config, TEST_CSV))
    assert ds.n_samples > 0 and ds.n_features > 0
    assert set(ds.y.unique()) == {0, 1}
    assert ds.X.isna().any().any()  # NaNs are allowed in features


def test_label_outside_0_1_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1\nA,0,1.0\nB,2,2.0\nC,1,3.0\n")
    with pytest.raises(DataError, match="only 0 and 1"):
        load_dataset(_config_for(write_config, csv))


def test_missing_label_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1\nA,0,1.0\nB,,2.0\nC,1,3.0\n")
    with pytest.raises(DataError, match="missing values"):
        load_dataset(_config_for(write_config, csv))


def test_duplicated_id_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1\nA,0,1.0\nA,1,2.0\n")
    with pytest.raises(DataError, match="duplicated sample IDs"):
        load_dataset(_config_for(write_config, csv))


def test_non_numeric_column_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1,batch\nA,0,1.0,b1\nB,1,2.0,b2\n")
    with pytest.raises(DataError, match="batch"):
        load_dataset(_config_for(write_config, csv))


def test_duplicated_feature_name_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1,f1\nA,0,1.0,2.0\nB,1,3.0,4.0\n")
    with pytest.raises(DataError, match="duplicate column names: f1"):
        load_dataset(_config_for(write_config, csv))


def test_missing_id_column_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "name,group,f1\nA,0,1.0\nB,1,2.0\n")
    with pytest.raises(DataError, match="sample_id"):
        load_dataset(_config_for(write_config, csv))


def test_all_problems_are_reported_together(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1\nA,0,1.0\nA,3,2.0\nC,1,3.0\n")
    with pytest.raises(DataError) as exc:
        load_dataset(_config_for(write_config, csv))
    assert "duplicated sample IDs" in str(exc.value)
    assert "only 0 and 1" in str(exc.value)


def test_single_class_is_rejected(write_config, tmp_path):
    csv = _write_csv(tmp_path, "sample_id,group,f1\nA,0,1.0\nB,0,2.0\n")
    with pytest.raises(DataError, match="only one class"):
        load_dataset(_config_for(write_config, csv))
