# Q-BRIDGE experiment runbook

## 1. Choose a locked source config

Use `exps/qbridge/sources/cifar224.json` or
`exps/qbridge/sources/imageneta.json`. Set machine-specific paths in a runtime
copy. Keep dataset split, pretrained tensor, seeds, class order, and optimizer
budget identical across every branch.

## 2. Train the paired task-0 base

```bash
python main.py \
  --config exps/qbridge/sources/cifar224.json \
  --overlay exps/qbridge/stages/train_base.json
```

Verify that `task_0.pkl` contains the model, class means, full covariance,
class order, run metadata, and quantum-kernel state. Record its SHA-256. Every
incremental branch for that seed must use this exact checkpoint.

## 3. Prepare base-only artifacts

Use `scripts/qbridge_prepare.py cache` to extract deterministic base-training
features. Fit the R6 calibrator with `calibrate`; fit R7 and R8 controls with
`rbf` and `mlp`. Each artifact is bound to the base checkpoint hash. Do not use
test data or incremental-task data for fitting or model selection.

## 4. Generate resolved configs

```bash
python scripts/generate_qbridge_configs.py \
  --source exps/qbridge/sources/cifar224.json \
  --base <task_0.pkl> \
  --artifact-root <artifact-directory> \
  --output-dir outputs/configs/cifar224 \
  --seeds 1993 2015 2026
```

The tracked files in `exps/qbridge/overlays/` define the locked method
variants. The generated files under `outputs/` contain local paths and should
not be committed.

## 5. Run and summarize

Run each generated JSON with `python main.py --config <resolved.json>`. Use a
fresh `output_root` for every protocol revision. Summarize completed runs with:

```bash
python scripts/summarize_qbridge.py \
  --root <output-root>/logs \
  --output <output-root>/qbridge_summary.json
```

Only compare runs with the same base hash and protocol hash. Report AIA, final
top-1, old/new accuracy, forgetting, BWT, run time, and memory cost. Unit tests
and smoke runs are implementation checks rather than benchmark evidence.

## Offline Kaggle

`QBRIDGE_Kaggle.ipynb` implements the same stages and packages logs, resolved
configs, report JSON, and artifact metadata. `scripts/package_qbridge_kaggle_repo.py`
creates the source ZIP when a fresh upload is required.
