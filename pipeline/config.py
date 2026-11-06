"""Load and validate params.yaml.

All validation happens here, before any data is read or any model is fit.
Unknown keys are rejected, and missing optional keys fall back to defaults
defined in the code.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

log = logging.getLogger(__name__)

# Fixed by design, not configurable: the decision threshold for the classifier.
DECISION_THRESHOLD: float = 0.5

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


class ConfigError(ValueError):
    """Raised when params.yaml is invalid. The message lists every problem found."""


def _as_list(info: ValidationInfo, value: Any) -> Any:
    """Wrap a scalar hyperparameter into a one-item list, with a warning."""
    if isinstance(value, list):
        if not value:
            raise ValueError(f"'{info.field_name}' must not be an empty list")
        return value
    log.warning(
        "'%s' is a scalar (%r); converting it to a single-value list [%r]",
        info.field_name,
        value,
        value,
    )
    return [value]


class Colors(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group0: str = "#1f77b4"
    group1: str = "#d62728"
    plot_color: str = "#79c8b2"

    @field_validator("group0", "group1", "plot_color")
    @classmethod
    def _check_hex(cls, value: str) -> str:
        if not _HEX_COLOR.match(value):
            raise ValueError(f"invalid color {value!r}; expected #RRGGBB")
        return value


class ElasticNetParams(BaseModel):
    """GridSearchCV space. Every combination of C and l1_ratio is tested."""

    model_config = ConfigDict(extra="forbid")

    C: list[float] = [0.001, 0.01, 0.1, 1, 10, 100]
    l1_ratio: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_iter: int = Field(5000, ge=1)

    @field_validator("C", "l1_ratio", mode="before")
    @classmethod
    def _grid_as_list(cls, value: Any, info: ValidationInfo) -> Any:
        return _as_list(info, value)

    @field_validator("C")
    @classmethod
    def _check_c(cls, values: list[float]) -> list[float]:
        if any(v <= 0 for v in values):
            raise ValueError("'C' values must be > 0")
        return values

    @field_validator("l1_ratio")
    @classmethod
    def _check_l1(cls, values: list[float]) -> list[float]:
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError("'l1_ratio' values must be between 0 and 1")
        return values


class XGBoostParams(BaseModel):
    """RandomizedSearchCV space. n_iter combinations are sampled from it."""

    model_config = ConfigDict(extra="forbid")

    n_iter: int = Field(30, ge=1)
    n_estimators: list[int] = [100, 200, 300, 500]
    max_depth: list[int] = [2, 3, 4, 6]
    learning_rate: list[float] = [0.01, 0.05, 0.1, 0.2]
    subsample: list[float] = [0.6, 0.8, 1.0]
    colsample_bytree: list[float] = [0.4, 0.6, 0.8, 1.0]
    min_child_weight: list[float] = [1, 3, 5]
    reg_lambda: list[float] = [1, 5, 10]

    @field_validator(
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        mode="before",
    )
    @classmethod
    def _grid_as_list(cls, value: Any, info: ValidationInfo) -> Any:
        return _as_list(info, value)

    @field_validator("n_estimators", "max_depth")
    @classmethod
    def _check_positive_int(cls, values: list[int], info: ValidationInfo) -> list[int]:
        if any(v < 1 for v in values):
            raise ValueError(f"'{info.field_name}' values must be >= 1")
        return values

    @field_validator("learning_rate")
    @classmethod
    def _check_lr(cls, values: list[float]) -> list[float]:
        if any(v <= 0 for v in values):
            raise ValueError("'learning_rate' values must be > 0")
        return values

    @field_validator("subsample", "colsample_bytree")
    @classmethod
    def _check_fraction(cls, values: list[float], info: ValidationInfo) -> list[float]:
        if any(not 0 < v <= 1 for v in values):
            raise ValueError(f"'{info.field_name}' values must be in (0, 1]")
        return values

    @field_validator("min_child_weight", "reg_lambda")
    @classmethod
    def _check_non_negative(cls, values: list[float], info: ValidationInfo) -> list[float]:
        if any(v < 0 for v in values):
            raise ValueError(f"'{info.field_name}' values must be >= 0")
        return values


class ShapParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(10, ge=1)
    compute_interactions: bool = True
    interaction_top_k: int = Field(15, ge=1)


class Config(BaseModel):
    """Full pipeline configuration. Unknown keys are errors."""

    model_config = ConfigDict(extra="forbid")

    # Global
    seed: int = 42
    outdir: Path = Path("results")

    # Input (required keys have no default)
    input: Path
    id_col: str = "sample_id"
    label_col: str
    group0_name: str
    group1_name: str
    data_type: str = "omics"
    # Optional external validation data (same format as input). None = no validation.
    validation_input_data: Path | None = None

    # Figures
    colors: Colors = Field(default_factory=Colors)
    figure_formats: list[Literal["png", "pdf", "svg"]] = ["png"]

    # Model and validation
    model: Literal["elasticnet", "xgboost"] = "elasticnet"
    cv_folds: Literal[3, 5, 10] = 5
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    # Percent (0-100) of non-missing values a feature must have to be kept.
    # 80 means: keep only features with at least 80% of their values filled in.
    min_filled_percent: float = Field(50.0, ge=0.0, le=100.0)
    n_jobs: int = -1

    # Hyperparameters and SHAP
    elasticnet: ElasticNetParams = Field(default_factory=ElasticNetParams)
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)
    shap: ShapParams = Field(default_factory=ShapParams)

    # Set by load_config: folder of params.yaml, used to resolve relative paths.
    _config_dir: Path = PrivateAttr(default=Path("."))

    @model_validator(mode="after")
    def _check_cross_fields(self) -> Config:
        if self.group0_name.strip() == self.group1_name.strip():
            raise ValueError("'group0_name' and 'group1_name' must be different")
        if self.id_col == self.label_col:
            raise ValueError("'id_col' and 'label_col' must be different columns")
        if not self.figure_formats:
            raise ValueError("'figure_formats' must contain at least one format")
        return self

    def resolved_input(self) -> Path:
        return self.input if self.input.is_absolute() else self._config_dir / self.input

    def resolved_validation_input(self) -> Path | None:
        if self.validation_input_data is None:
            return None
        path = self.validation_input_data
        return path if path.is_absolute() else self._config_dir / path

    def resolved_outdir(self) -> Path:
        return self.outdir if self.outdir.is_absolute() else self._config_dir / self.outdir

    def search_space_size(self) -> int:
        """Number of distinct hyperparameter combinations in the chosen model's space."""
        if self.model == "elasticnet":
            return len(self.elasticnet.C) * len(self.elasticnet.l1_ratio)
        xgb = self.xgboost
        return (
            len(xgb.n_estimators)
            * len(xgb.max_depth)
            * len(xgb.learning_rate)
            * len(xgb.subsample)
            * len(xgb.colsample_bytree)
            * len(xgb.min_child_weight)
            * len(xgb.reg_lambda)
        )

    def n_combinations(self) -> int:
        """Combinations actually evaluated: all of them for the grid, min(n_iter, space) for xgboost."""
        space = self.search_space_size()
        if self.model == "elasticnet":
            return space
        return min(self.xgboost.n_iter, space)

    def search_summary(self) -> str:
        n = self.n_combinations()
        fits = n * self.cv_folds
        if self.model == "elasticnet":
            return f"{n} combinations x {self.cv_folds} folds = {fits} fits"
        return (
            f"{n} combinations sampled from {self.search_space_size()} possible "
            f"x {self.cv_folds} folds = {fits} fits"
        )


def _format_error(err: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in err["loc"]) or "<root>"
    message = err["msg"]
    if message == "Extra inputs are not permitted":
        message = "unknown key (check for a typo)"
    elif message == "Field required":
        message = "required key is missing"
    return f"{location}: {message}"


def load_config(path: str | Path) -> Config:
    """Read and validate params.yaml. Raises ConfigError listing every problem found."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping of keys to values")

    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        problems = [_format_error(e) for e in exc.errors()]
        raise ConfigError("invalid params.yaml:\n  - " + "\n  - ".join(problems)) from exc

    config._config_dir = path.resolve().parent
    if config.validation_input_data is not None:
        data_dir = (config._config_dir / "data").resolve()
        target = config.resolved_validation_input().resolve()
        if not target.is_relative_to(data_dir):
            raise ConfigError(
                f"validation_input_data must be a file inside the data/ folder "
                f"({data_dir}); got {target}"
            )
    return config
