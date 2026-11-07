# Omics binary classification pipeline

Trains and evaluates a binary classifier (group 0 = control, group 1 = target group) and saves a
final model ready for external validation. External validation is in scope: it runs only when
`validation_input_data` is set (section 5).

The workflow standardizes the features (`StandardScaler`), imputes missing values (`KNNImputer`,
only when the data has missing values) and trains a classifier of choice: elastic net (logistic
regression) or XGBoost. It runs locally in Docker.

Out of scope: workflow orchestrators, batch correction, and use of metadata.

## Repository contents

1. `/data`: input CSV files. Contains synthetic example data to test the code. It is not real data
   and must not be used in any research.
2. `/pipeline`: source code of the pipeline.
3. `/tests`: test suite.
4. `/results`: all tables, models and plots produced by a run are written here. The results in the
   repository come from synthetic data and must not be interpreted as valid biological results.
5. `params.yaml`: run configuration.
6. `Dockerfile` and `requirements.lock`: Docker image and pinned dependencies.

## 1. Input data

The `data/` folder must contain one CSV file with:
- `sample_id` (name set by `id_col`): unique sample identifier.
- `group` (name set by `label_col`): values must be exactly 0 and 1.
- All other columns: used as features by the models. They must contain numbers (already in log2);
  empty cells (missing values) are allowed. If any of these columns contains text, the run stops
  with an error, so do not put metadata columns in this file.

CSV files with metadata can be kept in `data/`, but they are not used.

**Example data:** `data/example_test_dataset.csv` and `data/example_validation_dataset.csv` are
synthetic. They exist only to show the input format and to exercise the pipeline. Their sample IDs
(`sample_001` onward) are placeholders. Do not use their results for any scientific conclusion.

## 2. Configuration (`params.yaml`)

All settings are defined in `params.yaml`, and every parameter is documented in that file.

- Required: `input`, `label_col`, `group0_name` and `group1_name`. Any other missing key uses
  its default.
- Unknown keys are errors, so typos fail early.
- Relative paths are resolved from the folder that contains `params.yaml`.
- `min_filled_percent` (0 to 100): a feature is kept only if at least that percentage of samples
  have a value (e.g. `80` keeps features filled in at least 80% of samples). Features with no
  values are always removed. Recommended: 50 or higher. The filter uses all samples and never
  looks at the labels.
- The decision threshold is fixed at 0.5.

## 3. Pipeline stages and outputs

Each stage writes its checkpoint to disk before the next one starts.

| Folder | Contents |
|---|---|
| `01_qc/` | Data summary, feature filter table, removed features, train and test ID lists |
| `02_tuning/` | `cv_results.csv` (every combination and fold), `best_params.json`, `evaluation_model.joblib`, `transformed_samples.csv` (samples after scaling and imputation) |
| `03_final_model/` | Metrics, confusion matrices, predictions, ROC and confusion-matrix figures, `evaluation_model.joblib` (copy), `final_model.joblib` (fit on 100% of samples) |
| `04_interpretation/` | Elastic net: coefficient tables, importance table, lollipop plots. XGBoost: SHAP tables and plots |
| `05_validation/` | Only if `validation_input_data` is set: metrics, confusion matrix, predictions, ROC and confusion-matrix figures, validation data QC, transformed validation samples and feature interpretation (`interpretation/`) |
| root | `params_used.yaml` (configuration with defaults filled in), `run_info.json` (seed, versions, date, input hash) |

- **Evaluation model** (`evaluation_model.joblib`): trained on the training split only. Its test
  metrics are the real results of the run. Use it to recompute metrics or draw new figures without
  rerunning the search.
- **Final model** (`final_model.joblib`): trained on 100% of the samples with the selected
  hyperparameters. It has no independent test set and is meant for external validation.

Each saved model is a bundle with the pipeline, threshold, feature names, removed features, group
names, seed, hyperparameters and library versions.

## 4. External validation (optional)

Set in `params.yaml`:

- `validation_input_data: null` (default): the run stops after saving the final model.
- `validation_input_data: data/validation_data.csv`: the final model is applied to this file,
  which must be inside `data/`.

The validation file has the same format, `id_col`, `label_col` and group names as the training
input. Rules:

- A feature used by the model that is missing from the file is an error (the error lists them).
  Extra features are ignored with a warning.
- `min_filled_percent` is applied to the validation samples. If a feature used by the model does
  not pass, the run stops with an error.
- Nothing is refit on the validation data (model, scaler, imputer or threshold).
- Validation samples must not be in the training data. This is not checked automatically, so
  check the sample IDs before running.

## 5. Running with Docker

All commands run from the project folder. The `docker run` commands mount three volumes:
`data/` (read-only input), `results/` (where outputs are written) and `params.yaml` (read-only
configuration).

**Step 1. Build the image.** Creates the `omics-pipeline` image with Python 3.11 and the pinned
dependencies. Repeat only when the code or dependencies change.

```bash
docker build -t omics-pipeline:latest .
```

**Step 2. Check the configuration and data (recommended before every run).** Validates
`params.yaml` and the training input, prints the data summary (samples per group, missing values,
features removed by the filter) and the size of the hyperparameter search, then exits. It writes
nothing and trains nothing. The validation file is not checked in this mode.

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml --validate-only
```

**Step 3. Full run.** Runs every stage in order: QC and train/test split, hyperparameter search
with cross-validation on the training split, test evaluation and final model, interpretation, and
external validation if it is set. Outputs go to `results/` (section 3).

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml
```

**Step 4 (optional). Rerun only interpretation and validation.** Loads the models already saved in
`results/` and reruns only the interpretation (and the validation, if set), with no retraining.
Useful after changing the validation file or interpretation settings.

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml --only-interpret
```

**Step 5 (optional). Run the test suite.** Runs `pytest` inside the image. `data/` must be mounted
because the tests use `data/example_test_dataset.csv`.

```bash
docker run --rm -v "$PWD/data:/app/data:ro" --entrypoint python omics-pipeline:latest -m pytest -q
```

## 6. Development

Dependencies are pinned in `requirements.lock`. The image uses Python 3.11. The test suite
(`tests/`) covers configuration and data validation, the feature filter, the split, the pipeline
(scaler and imputer fitted on the training data only), the checkpoints, the elastic net importance
rules, the SHAP stage, the external validation stage and the `--validate-only` output.
