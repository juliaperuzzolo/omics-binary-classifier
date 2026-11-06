from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PARAMS = ROOT / "params.yaml"
TEST_CSV = ROOT / "data" / "example_test_dataset.csv"


@pytest.fixture
def example_yaml_text() -> str:
    return EXAMPLE_PARAMS.read_text(encoding="utf-8")


@pytest.fixture
def write_config(tmp_path):
    """Write a params.yaml into tmp_path and return its path."""

    def _write(text: str) -> Path:
        path = tmp_path / "params.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def minimal_yaml(tmp_path) -> str:
    """The four required keys, with the input pointing at a CSV in tmp_path."""
    return (
        f"input: {tmp_path / 'data.csv'}\n"
        "label_col: group\n"
        "group0_name: Control\n"
        "group1_name: Cancer\n"
    )
