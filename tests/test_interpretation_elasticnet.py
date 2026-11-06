import numpy as np
import pandas as pd

from pipeline.interpretation_elasticnet import importance_table


def _coefs(values):
    return pd.DataFrame({"feature": [f"f{i}" for i in range(len(values))],
                         "coefficient": values})


def test_largest_positive_and_negative_are_exactly_100_percent():
    table = importance_table(_coefs([2.0, 0.5, -3.0, -1.5, 0.0]), "Control", "Cancer")
    pos = table[table["direction"] == "positive"].set_index("feature")
    neg = table[table["direction"] == "negative"].set_index("feature")
    assert pos.loc["f0", "importance_pct"] == 100.0
    assert np.isclose(pos.loc["f1", "importance_pct"], 25.0)
    assert neg.loc["f2", "importance_pct"] == 100.0
    assert np.isclose(neg.loc["f3", "importance_pct"], 50.0)


def test_importance_is_between_0_and_100_and_zeros_are_excluded():
    table = importance_table(_coefs([0.0, 1.2, -0.4, 0.0, 7.0, -9.0]), "Control", "Cancer")
    assert table["importance_pct"].between(0, 100).all()
    assert "f0" not in table["feature"].tolist()
    assert "f3" not in table["feature"].tolist()


def test_sign_is_kept_and_points_to_correct_group():
    table = importance_table(_coefs([1.0, -1.0]), "Control", "Cancer").set_index("feature")
    assert table.loc["f0", "coefficient"] == 1.0
    assert table.loc["f0", "points_to"] == "Cancer"
    assert table.loc["f1", "coefficient"] == -1.0
    assert table.loc["f1", "points_to"] == "Control"


def test_empty_direction_has_no_rows():
    table = importance_table(_coefs([1.0, 2.0]), "Control", "Cancer")
    assert set(table["direction"]) == {"positive"}
