"""Checkpoints and run records: feature-name sanitization, saved model bundles, run_info.json.

Every saved model is a self-contained bundle: it can be loaded later to recompute
metrics or draw new figures and tables without rerunning the search.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import re
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import yaml

from .config import DECISION_THRESHOLD, Config

# Characters that XGBoost rejects in feature names.
_FORBIDDEN = re.compile(r"[\[\]<]")

_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "shap", "numba",
    "statsmodels", "seaborn", "matplotlib", "pydantic", "PyYAML", "joblib",
]


def sanitize_feature_names(names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return safe unique names and a mapping {safe_name: original_name}."""
    safe_names: list[str] = []
    seen: set[str] = set()
    for name in names:
        base = _FORBIDDEN.sub("_", name)
        candidate, counter = base, 1
        while candidate in seen:  # keep names unique after replacement
            counter += 1
            candidate = f"{base}_{counter}"
        seen.add(candidate)
        safe_names.append(candidate)
    return safe_names, dict(zip(safe_names, names))


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def build_bundle(
    *,
    config: Config,
    estimator: Any,
    feature_names_original: list[str],
    feature_names_safe: list[str],
    removed_features: list[str],
    trained_on: str,
    best_params: dict[str, Any],
    note: str,
) -> dict[str, Any]:
    """Everything needed to reuse a trained model without the original code state."""
    return {
        "pipeline": estimator,
        "model_name": config.model,
        "threshold": DECISION_THRESHOLD,
        "feature_names_original": feature_names_original,
        "feature_names_safe": feature_names_safe,
        "removed_features": removed_features,
        "label_col": config.label_col,
        "id_col": config.id_col,
        "group0_name": config.group0_name,
        "group1_name": config.group1_name,
        "seed": config.seed,
        "best_params": best_params,
        "trained_on": trained_on,
        "versions": library_versions(),
        "note": note,
    }


def save_bundle(bundle: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def write_params_used(config: Config, path: Path) -> Path:
    """Copy of the configuration actually used, with every default filled in."""
    data = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_run_info(config: Config, path: Path) -> Path:
    """Seed, versions, date and input hash. Rewritten at every stage that runs."""
    info = {
        "seed": config.seed,
        "versions": library_versions(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(config.resolved_input()),
        "input_sha256": file_sha256(config.resolved_input()),
        "model": config.model,
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
