# Omics binary classification pipeline

Trains and evaluates a binary classifier (group 0 = control, group 1 = target group) from a
log2 feature matrix, and saves a final model ready for external validation.
Models: elastic net (logistic regression) or XGBoost. Runs locally in Docker.

Out of scope: workflow orchestrators, batch correction, and use of metadata.
External validation is in scope: it runs only when `validation_input_data` is set (section 5).

## 1. Input data

One CSV file with:
- `sample_id` (name set by `id_col`): unique sample identifier.
- `group` (name set by `label_col`): values must be exactly 0 and 1.
- Remaining columns: numeric features, already in log2. Missing values (NaN) are allowed.

Any other non-numeric column is an error. Metadata columns must not be in the file.

**Example data:** the files `data/example_test_dataset.csv` and `data/example_validation_dataset.csv`
are synthetic. They do not come from real patients or real experiments. They exist only to show the
input format and to exercise the pipeline. Their sample IDs (`sample_001` onward) are placeholders.
Do not use their results for any scientific conclusion.

## 2. Configuration (`params.yaml`)

Every parameter is documented in the example file. Only `input`, `label_col`,
`group0_name` and `group1_name` are required. Any other key that is missing uses its default.
Unknown keys are errors, so typos fail early. Relative paths are resolved from the folder
that contains `params.yaml`.

Feature filter, `min_filled_percent`: a percentage from 0 to 100. A feature is kept only if
at least that percentage of its samples have a value. For example, `80` keeps features filled
in at least 80% of samples, and `70` keeps features filled in at least 70% of samples. Features
with no values at all are always removed. Recommended: 50 or higher. The filter uses all
samples and never looks at the labels.

The decision threshold is fixed at 0.5.

## 3. Checking the data without training

`python -m pipeline --config params.yaml --validate-only` validates `params.yaml` and the training
input, prints the data summary (samples per group, missing values, features removed by the filter)
and the hyperparameter search size, and exits. It writes nothing and trains nothing. The
validation file (if set) is checked only when the validation stage runs, not by this mode.

## 4. Pipeline stages and outputs

Each stage writes its checkpoint to disk before the next one starts.

| Folder | Contents |
|---|---|
| `01_qc/` | Data summary, feature filter table, removed features, train and test ID lists |
| `02_tuning/` | `cv_results.csv` (every combination and fold), `best_params.json`, `evaluation_model.joblib` (checkpoint 1), `transformed_samples.csv` (samples after scaling and imputation) |
| `03_final_model/` | Metrics, confusion matrices, predictions, ROC and confusion-matrix figures, `evaluation_model.joblib` (copy), `final_model.joblib` (fit on 100% of samples) |
| `04_interpretation/` | Elastic net: coefficient tables, importance table, lollipop plots. XGBoost: SHAP tables and plots |
| `05_validation/` | Only if `validation_input_data` is set: metrics, confusion matrix, predictions, ROC and confusion-matrix figures, QC of the validation data (`validation_data_qc.txt`, `validation_features_filter.csv`), samples after scaling and imputation (`transformed_validation_samples.csv`), and the feature interpretation of the final model on the validation data (`interpretation/`) |
| root | `params_used.yaml` (configuration with defaults filled in), `run_info.json` (seed, versions, date, input hash) |

**Evaluation model** (`evaluation_model.joblib`): trained on the training split only. Use it to
recompute test metrics or draw new figures and tables without rerunning the search. Its test
metrics are the real results of the run.

**Final model** (`final_model.joblib`): trained on 100% of the samples with the selected
hyperparameters. It has no independent test set. It is meant for external validation.

Each saved model is a bundle with the pipeline, threshold (0.5), feature names (original and
safe), removed features, group names, seed, hyperparameters, library versions and a note.

## 5. External validation (optional)

The pipeline asks nothing at run time. Whether validation runs is set in `params.yaml`:

- `validation_input_data: null` (default): the pipeline stops after saving the final model, ready
  for later validation.
- `validation_input_data: data/validation_data.csv`: after the final model, the pipeline applies
  it to the validation data. The file must be inside `data/`.

The validation file has the same format as the training input: `sample_id`, `group` (0 and 1),
and the features. It uses the same `id_col`, `label_col` and group names. Rules:

- A feature the model uses that is missing from the file is an error. The error lists the
  missing features.
- A feature the model does not use is ignored, and the run prints a warning.
- The missing-value filter is applied to the validation data with the same `min_filled_percent`,
  computed on the validation samples. If a feature the model uses does not pass the filter, the
  run stops with an error listing it: the model cannot be applied without that feature.
  Features the model does not use are not affected.
- Validation samples must not be in the training data. The pipeline does not check this
  automatically. Check the sample IDs before running validation.
- The model is not refit. No scaler, imputer or threshold is fit on the validation data.
  The threshold stays at 0.5.

## 6. Running with Docker

Build the image (run from the project folder):

```
docker build -t omics-pipeline:latest .
```

Mount `data/` (read-only) and `results/` as volumes. Always validate first:

```
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml --validate-only
```

Full run (search, evaluation, final model, interpretation):

```
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml
```

Reuse saved checkpoints and rerun only the interpretation (and the validation, if it is set),
with no retraining:

```
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/results:/app/results" \
  -v "$PWD/params.yaml:/app/params.yaml:ro" \
  omics-pipeline:latest --config params.yaml --only-interpret
```

Run the test suite inside the image (the `data/` folder must be mounted, because the tests use
`data/example_test_dataset.csv`):

```
docker run --rm -v "$PWD/data:/app/data:ro" --entrypoint python omics-pipeline:latest -m pytest -q
```

### Memory and parallel jobs

`n_jobs: -1` uses every CPU core. Inside a container this can exceed the memory available to
Docker. If the run is killed or slow, set `n_jobs` to a small number (for example 2). The
SHAP interaction heatmap is skipped when its estimated memory or run time is too large, and a
warning is printed.

## 7. Known limitations

- With a small test set, metrics are unstable. Treat test results as a check of the pipeline,
  not as performance estimates.
- With the fixed threshold of 0.5, a model trained on unbalanced data may predict no positive
  cases. This is a property of the threshold, not a bug.
- The validation file must not share samples with the training file. The pipeline does not check
  this automatically. Check the sample IDs before running validation.

## 8. Development

Dependencies are pinned in `requirements.lock`. The image uses Python 3.11.
The test suite (`tests/`) covers configuration and data validation, the feature filter, the
split, the pipeline (scaler and imputer fitted on the training data only), the checkpoints,
the elastic net importance rules, the SHAP stage, the external validation stage (including
the missing-feature and null-filter errors), and the `--validate-only` output.
