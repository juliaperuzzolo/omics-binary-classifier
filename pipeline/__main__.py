"""Command-line entry point: python -m pipeline --config params.yaml [--validate-only]."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, load_config
from .artifacts import write_params_used, write_run_info
from .data import DataError, load_dataset
from .qc import QCError, ensure_writable, filter_features, overview_lines, run_qc
from .evaluation import run_evaluation
from .interpretation_elasticnet import run_elasticnet_interpretation
from .shap_interpretation import ShapError, run_shap_interpretation
from .resume import ResumeError, load_checkpoints
from .validation import ValidationStageError, final_model_path, run_validation
from .tuning import TuningError, run_tuning

log = logging.getLogger("pipeline")


def _interpret(config, qc, tuning) -> int:
    """Stage 5 (elastic net) or stage 6 (xgboost). Shared by both run modes."""
    if config.model == "xgboost":
        try:
            shap_res = run_shap_interpretation(config, qc, tuning)
        except (QCError, ShapError) as exc:
            log.error("%s", exc)
            return 2
        print(
            f"\nInterpretation (SHAP, test set): dependence plots for top {shap_res['top_k']} "
            f"features; additivity error {shap_res['additivity_error']:.1e}"
            f"\nOutputs: {shap_res['output_dir']}"
        )
        for warning in shap_res["warnings"]:
            print(f"WARNING: {warning}")
        return 0

    try:
        interp = run_elasticnet_interpretation(config, tuning)
    except (QCError, ValueError) as exc:
        log.error("%s", exc)
        return 2

    print(
        f"\nInterpretation (elastic net): {interp['n_nonzero']} non-zero coefficients "
        f"({interp['n_pos']} -> {config.group1_name}, {interp['n_neg']} -> {config.group0_name}); "
        f"{interp['n_zero']} zeroed by regularization"
        f"\nOutputs: {interp['output_dir']}"
    )
    for warning in interp["warnings"]:
        print(f"WARNING: {warning}")
    return 0


def _validate_if_configured(config) -> int:
    """Stop after the final model if there is no validation data; otherwise validate."""
    if config.validation_input_data is None:
        print(
            "\nFinal model ready for external validation: "
            f"{final_model_path(config)}\n"
            "validation_input_data is null, so the pipeline stops here. "
            "Set it in params.yaml (a CSV inside data/) to validate."
        )
        return 0
    try:
        val = run_validation(config)
    except ValidationStageError as exc:
        log.error("%s", exc)
        return 2
    m = val.metrics
    print(
        "\nExternal validation (final model, no retraining)\n"
        f"AUROC {m['auroc']:.3f}  accuracy {m['accuracy']:.3f}  sensitivity {m['recall']:.3f}  "
        f"specificity {m['specificity']:.3f}  F1 {m['f1']:.3f}  "
        f"balanced accuracy {m['balanced_accuracy']:.3f}"
        f"\nOutputs: {val.output_dir}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Train and evaluate a binary omics classifier from params.yaml.",
    )
    parser.add_argument(
        "--config", default="params.yaml", help="path to params.yaml (default: params.yaml)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate params.yaml and the input data, print the search size, and exit",
    )
    parser.add_argument(
        "--only-interpret",
        action="store_true",
        help="reuse the saved checkpoints and run only the interpretation (no retraining)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    if args.only_interpret:
        try:
            _, qc, tuning = load_checkpoints(config)
        except (ResumeError, DataError) as exc:
            log.error("%s", exc)
            return 2
        code = _interpret(config, qc, tuning)
        return code if code else _validate_if_configured(config)

    try:
        dataset = load_dataset(config)
    except DataError as exc:
        log.error("%s", exc)
        return 2

    log.info("Model: %s", config.model)
    log.info("Hyperparameter search: %s", config.search_summary())

    if args.validate_only:
        filt = filter_features(dataset.X, config.min_filled_percent)
        print("\nDATA SUMMARY (preview, nothing written)")
        print("\n".join(overview_lines(config, dataset, filt)))
        log.info("Validation passed. Nothing was trained (--validate-only).")
        return 0

    try:
        ensure_writable(config.resolved_outdir())
        write_params_used(config, config.resolved_outdir() / "params_used.yaml")
        write_run_info(config, config.resolved_outdir() / "run_info.json")
        qc = run_qc(config, dataset)
    except QCError as exc:
        log.error("%s", exc)
        return 2

    # Plain text, no log prefix, so the report reads the same as data_summary.txt.
    print(qc.report, end="")

    try:
        tuning = run_tuning(config, qc)
    except TuningError as exc:
        log.error("%s", exc)
        return 2

    print(
        f"Best {config.model} parameters: {tuning.best_params}\n"
        f"CV AUROC (mean +/- std over {config.cv_folds} folds): "
        f"{tuning.cv_auc_mean:.3f} +/- {tuning.cv_auc_std:.3f}\n"
        f"Checkpoint 1 (evaluation model, training split): {tuning.checkpoint_path}"
    )

    try:
        ev = run_evaluation(config, qc, tuning)
    except QCError as exc:
        log.error("%s", exc)
        return 2

    tr, te = ev.metrics["train"], ev.metrics["test"]
    print(
        "\nMetrics (threshold 0.5)        train     test\n"
        + "\n".join(
            f"{label:<28}{tr[key]:>8.3f}{te[key]:>10.3f}"
            for key, label in [
                ("auroc", "AUROC"), ("accuracy", "Accuracy"), ("precision", "Precision"),
                ("recall", "Sensitivity (recall)"), ("specificity", "Specificity"),
                ("f1", "F1"), ("balanced_accuracy", "Balanced accuracy"),
            ]
        )
        + f"\nCV AUROC: {tuning.cv_auc_mean:.3f} +/- {tuning.cv_auc_std:.3f}"
        + f"\nOutputs: {ev.output_dir}"
        + f"\nFinal model (100% of samples): {ev.final_model_path}"
    )

    code = _interpret(config, qc, tuning)
    return code if code else _validate_if_configured(config)


if __name__ == "__main__":
    sys.exit(main())
