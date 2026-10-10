# Failure localization: Kaggle Offline

This study continues the experimental-integrity fixes on `Quantum_research`.
It prepares nine fresh runs and does not contain benchmark results. No full
training was performed in Codex: the local environment has no CUDA or Kaggle
input assets. Unit tests use temporary synthetic fixtures, never as results.

## Fixed experiment

| Variant | Base task | Incremental representation regularization |
|---|---|---|
| A0 | Original RSIAT | Original RSIAT |
| A1 | Current PQK | Original RSIAT |
| A2 | Current PQK | Current PQK |

Seeds: **1993, 2015, 2026**. Default dataset: **CIFAR224 (CIFAR-100 resized to
224)**, using `exps/adapter_cifar224.json` unchanged for learning rates,
loss weights, optimizer, batch size, task sizes and epoch budgets. The current
PQK settings, frozen incremental metric, `old_proj`, SSCA and CA remain intact.
The runner only changes the two quantum switches plus operational output,
offline and logging settings. All variants use the corrected Đợt 1 optimizer.

ImageNet-A/R is optional via `--dataset imageneta` or `imagenetr`, using that
dataset's current `exps/adapter_<dataset>.json`. Missing quantum options inherit
the current CIFAR PQK configuration; existing dataset training settings are
retained. A study always uses one dataset and one shared configuration.
`exps/failure_localization_cifar224.json` describes the matrix; it is not a
`main.py` training configuration. Nine complete JSON configs are materialized
after preflight under the new study's `configs/` directory.

Each seed fixes a class permutation shared by its three variants. Private
sampler/worker generators from Đợt 1 retain paired input streams. Each run starts
from the uploaded pretrained backbone and a fresh seed. No checkpoints are
shared between variants. CPU tests check identical A1/A2 base model+metric state;
actual GPU base state hashes are exported for checking after the experiment.

## Upload once; Internet OFF

1. Build the source bundle locally:

   ```powershell
   python -B scripts/package_failure_localization.py
   ```

   Upload `artifacts/kaggle_failure_localization/source.zip` as a private Kaggle
   input dataset. The ZIP contains source, configs, tests, notebook and docs;
   it excludes historical logs/checkpoints and datasets. Existing ZIPs are never
   replaced; use `--output <new-file.zip>` for a newer bundle.

2. Upload the real dataset manually. Supported layouts:

   ```text
   /kaggle/input/<any-upload-name>/.../cifar-100-python/{train,test,meta}
   /kaggle/input/<any-upload-name>/.../cifar-100-python.tar.gz
   /kaggle/input/<any-upload-name>/.../<dataset>/{train,test}/<class>/*.jpg
   ```

   CIFAR uses torchvision's official Python batch format, with integrity checks
   and `download=False`. Supply the directory above `cifar-100-python` as
   `DATA_ROOT` / `--data-root`; its directory itself is also accepted.
   The official uploaded archive can be expanded safely into a fresh study
   directory under `/kaggle/working/`, without downloading or changing input files.

   ImageNet-A/R must already have the intended `train/` and `test/` partitions
   with 200 identical class names and mappings. The original flat ImageNet-A/R
   release alone is not sufficient: this runner does not invent a random split.
   Both splits must have examples in every class, and training needs at least
   two images per class for covariance computation.

3. Upload the **same existing pretrained ViT-B/16 checkpoint** for every run.
   Standard timm/HF PyTorch `.pth/.pt/.bin` state dictionaries, `.safetensors`,
   or Google's local ViT `.npz` are supported. Wrapped `state_dict` / `model`
   dictionaries and `module.` / `model.` prefixes are handled. Classifier heads
   are ignored because the original adapter backbone has `num_classes=0`.
   Backbone shape/key coverage is checked strictly; only new adapter parameters
   may be missing. Trained adapter checkpoints and partial backbones are rejected.
   Shape checks cannot prove IN21K provenance: upload the correct checkpoint for
   `pretrained_vit_b16_224_in21k_adapter` and retain its source/provenance yourself.

4. Keep Kaggle's existing CUDA-compatible Torch/torchvision pair. If small
   dependencies are missing, manually upload Linux wheels compatible with the
   Kaggle Python version, including transitive dependencies. Notebook installation
   uses `pip --no-index --find-links ... -r requirements-kaggle-offline.txt`.
   There is no online fallback. Do not install the local machine's
   `requirements.txt` or replace the CUDA pair. NPZ loading uses local timm;
   safetensors weights require the local `safetensors` dependency.

## Run the notebook

Import `notebooks/QKSR_Failure_Localization_Kaggle_Offline.ipynb` into Kaggle.
Set **Internet OFF** and **one GPU ON**. Attach the uploaded inputs. Run cells in
order. Leave `DATA_ROOT` and `PRETRAINED_PATH` as `None` for recursive discovery;
set full paths when more than one candidate exists. No upload directory names
are hardcoded. The notebook verifies the source ZIP's file hashes and extracts
it into a fresh working directory. If Kaggle has already unpacked the uploaded
ZIP, its directory is also discovered and copied to a new working directory.
Use `SOURCE_PATH` in the first cell for an ambiguous ZIP/directory selection.

Preflight imports the real dependencies, loads both dataset splits with no
downloads, verifies class coverage and train/test mappings, checks evaluation
transforms, loads the real pretrained backbone on CPU and checks its trainable
adapter mask. It fingerprints dataset content, weights, source, nine configs,
class permutations, task sizes and package versions. The training command
rechecks these before starting. IP connections are blocked in-process as an
additional guard; local Unix IPC for DataLoader workers remains available.

Set `SELECT_VARIANTS` and `SELECT_SEEDS` in the run cell. Default: all nine runs.
If Kaggle session limits are too short, run complete experiments in separate
sessions, e.g. all three variants for one seed. Keep assets, source and runtime
versions identical and save every session's results ZIP. No intermediate task
checkpoint exists, so an interrupted run must start fresh. A run directory that
already exists is rejected, including an interrupted one. Within one session
you may run remaining, unstarted selections in the same prepared study.

The same CLI can be used from a notebook after extracting the source:

```bash
python scripts/run_failure_localization.py --prepare-only --dataset cifar224 \
  --input-root /kaggle/input --output-root /kaggle/working
# If paths are ambiguous, add --data-root ... --pretrained-path ...
# For an uploaded archive: --cifar-archive /kaggle/input/.../cifar-100-python.tar.gz

python scripts/run_failure_localization.py --run-study /kaggle/working/<new-study-dir>
# Optional complete-run selection: --variants A0 A1 A2 --seeds 1993

python scripts/collect_failure_localization.py \
  --study /kaggle/working/<new-study-dir> \
  --output /kaggle/working/<new-results-name>.zip
```

Always use the dedicated runner, not `main.py`, for CSV/JSON exports and console
policy. `--prepare-only` needs real assets but permits a CPU-only environment;
`--run-study` refuses CPU benchmark training. No code edits or hyperparameter
tuning should occur between preflight and execution.

## Outputs and console

```text
/kaggle/working/failure_localization_<dataset>_<unique-id>/
  study_manifest.json       # expected nine runs; preparation is not training
  source_config.json
  preflight.json            # content hashes, class orders, budgets, runtime
  configs/A0_1993.json ...  # nine complete configs
  runs/A0_1993/
    run_manifest.json       # exact runtime config and class order
    tasks.csv               # flushed after each task
    tasks.jsonl             # task matrices and detailed read-only diagnostics
    train.log               # file-only learner log
    runtime_output.log      # captured prints/warnings, no progress bars
    run_summary.json        # only created after every task and final save succeed
    final_task.pkl          # exactly one model checkpoint per completed run
    FAILED.json             # only when a started run fails
```

One console line per task shows variant/seed/task, completed training and CA
epochs, pre/post total/old/new accuracy, post-CA forgetting and task time. Base
task has no CA, so pre/post coincide and old accuracy/forgetting are `n/a`.
No per-batch losses, kernel histograms, gradient diagnostics or progress bars
are printed. New diagnostics remain available in JSON. No old output is repaired
or overwritten. Checkpoints are saved only after the final task; full covariance
is retained, so allow disk space for nine large checkpoints.

## Metric definitions and localization limits

All accuracies are percentages, forgetting is in percentage points. With tasks
`t=0..T-1`, `a_t` is post-CA total accuracy over seen classes, and `a[t,i]` is
accuracy on the class range of task `i` evaluated after task `t`.

- `average_incremental_accuracy = mean(a_t)`, including the base task.
- `final_accuracy = a_(T-1)`.
- `F_t = mean_i<t(max_(s=i..t-1) a[s,i] - a[t,i])`; base forgetting is undefined.
- `average_forgetting = mean(F_1..F_(T-1))`. `final_forgetting = F_(T-1)` is also
  exported to distinguish the two common reporting conventions. Gains can yield
  negative forgetting; values are not clamped.
- `training_seconds` sums timed adapter-training and CA stages, including their
  calibration and epoch evaluation. `total_task_seconds` also includes data
  preparation, drift/statistics and pre/post probes. `run_wall_seconds` includes
  initialization and final checkpoint I/O. CUDA is synchronized around timers.

Compare A1 versus A0 for the base-PQK intervention (including downstream effects),
and A2 versus A1 for incremental PQK conditional on a shared base. A1/A2 base
state hashes are exported to audit this control. Pre/post-CA deltas expose the
observed effect of classifier alignment in each task. All variants keep SSCA
enabled: this matrix cannot establish SSCA's independent causal effect or a
quantum-specific advantage. Those would need additional experiments, outside
this request. No aggregate comparison report is generated here.

## Files to send back

Send the results ZIP created by the collection cell, or every session's ZIP if
you split the nine runs. It includes task CSV/JSON, run summaries, class orders,
configs, preflight provenance and logs. `collection_manifest.json` explicitly
marks missing/incomplete runs. You do **not** need to send large `.pkl` files for
the comparison report; retain/download them separately. These exports contain
the values needed for AIA, Final Accuracy, Average Forgetting, timing, paired
pre/post-CA inspection and mean ± sample std across the three seeds.

## Local verification

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/run_failure_localization.py --help
git diff --check
```

Tests cover the fixed matrix/budget, local-only loaders, incompatible backbones,
safe archive handling, network guard, exact metric formulas, real adapter weight
mapping/freezing, short logging, final-only checkpointing and A1/A2 base parity
through a temporary three-task synthetic pipeline. Notebook code cells compile
with empty outputs. These checks do not certify a GPU/benchmark run.

Verified on 2026-10-10: **60 tests, 58 passed, 2 CUDA skips**, in 75.539 seconds
(`python -B -m unittest discover -s tests -v`). There are 21 new failure-localization
tests alongside the 39 Đợt 1/quantum tests. Notebook bootstrap was executed against
both a temporary source ZIP and its unpacked directory; every bundled source hash
was verified. The CLI help and `git diff --check` also passed.

Local runtime: Python 3.12.5, PyTorch 2.12.1+cpu, NumPy 2.3.3. AST comparison
against the Đợt 1 commit confirms the architecture classes, losses, optimizer
builder, transforms, displacement/covariance methods and CA algorithm are unchanged.
All nine historical log files retain their original SHA-256; no existing checkpoint
files were present. Real Kaggle asset preflight and GPU training remain pending.
