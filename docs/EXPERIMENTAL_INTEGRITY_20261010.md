# Experimental integrity before QKSR ablations

Branch: `Quantum_research`. Base commit: `baa54ff`.
Scope: experimental-integrity fixes only; no full dataset training, downloads,
architecture changes, loss changes, hyperparameter edits, or covariance
compensation changes. All test checkpoints live in temporary directories.

## Confirmed implementation bugs and fixes

| Finding | Evidence before the fix | Result after the fix |
|---|---|---|
| Drift rows were not paired observations | Two separate iterations of a shuffled, randomly augmented training loader, followed by row-wise subtraction | Reuse one deterministic evaluation view of the training subset; retain source sample IDs, align explicitly by ID, reject duplicates/missing IDs/label changes |
| Validation leaked into memory and CA inputs | `_compute_class_mean()` fetched full class training data, bypassing the task's split | Require the explicit training subset during tuning; compute means/covariances from its deterministic view; CA samples from those statistics |
| Test metrics were exposed during tuning | CA and `eval_task()` selected `test_loader` regardless of the validation split | Build validation over all seen classes; route train-stage, CA-stage and final metrics to it; do not fetch the test dataset in tuning runs |
| Checkpoints omitted continuation randomness | Python/NumPy/Torch and sampler generators were not saved/restored; rebuilding heads/AE consumes RNG | Save all global RNGs and private loader generators; restore after rebuilding model state; validate hyperparameters and recorded split manifests |
| Incremental AdamW differed between RSIAT and QKSR | Baseline used only network parameters, while QKSR used groups containing `old_ae` | Both use the same non-metric groups, including `old_ae` at configured LR/decay; exclude frozen parameters and reject missing/duplicate active parameters |

These are correctness/protocol findings. Their impact on benchmark accuracy,
forgetting and quantum-specific benefit has NOT been measured in this work.

## Data identity and partitions

`DummyDataset.sample_ids` identify positions in the source train/test arrays.
IDs survive train/validation splitting and `get_eval_view()`. ID namespaces
are specific to a data source, not shared between train and test.

The drift loader uses `shuffle=False`, deterministic test transforms and
`num_workers=0`. It is reused before and after adapter training; inference is
in eval mode with gradients disabled. Probes preserve global RNG and do not
advance the training sampler/worker streams.

When tuning, held-out validation images may be evaluated but never enter
adapter training, quantum calibration, drift estimation, class statistics or
synthetic-feature CA training. Validation for earlier tasks is reconstructed
by the evaluator from the same deterministic partitions; it is not an old
image replay source for the optimizer. Final locked runs with `val_ratio=0`
continue to evaluate the test split.

Checkpoints record train/validation IDs and data/label SHA-256 fingerprints.
Array-backed images are hashed by content. Path-backed datasets are hashed
by paths and labels, NOT by the bytes of every image file; external dataset
integrity checks are still needed to detect files changed in place under the
same filenames. Historical partition fingerprints are validated when a new
task starts after loading a checkpoint.

## Resume contract

Supported resume point: a completed task, after evaluation and `after_task()`.

- Python `random`, NumPy global RNG, Torch CPU RNG, and all visible CUDA RNGs.
- Independent `train_sampler`, `train_workers`, evaluation and statistics
  generators, keyed by stable SHA-256-derived seeds. Worker initialization
  seeds Python/NumPy from the PyTorch worker seed.
- Model, old AE, quantum state/calibration, class statistics, task/class order,
  accuracy curves, split manifests and diagnostics.
- Optimizer/scheduler recreation at each task remains unchanged. They need
  not be serialized for a task-boundary continuation.
- Persistent workers are replaced when the next task's dataset/loader is
  constructed. Their live per-epoch state is NOT serialized. This does not
  support mid-epoch or mid-task resume.

New-format metadata rejects changed optimizer, rates, losses, schedules,
worker count/persistence, validation ratio and quantum settings, including
fields removed from the resume config. Checkpoints claiming the new format
but missing required integrity state fail closed. CUDA RNG restoration requires
the same visible device count.

Legacy checkpoints can be loaded without rewriting them. They warn and set
`resume_reproducible=False`; this provenance survives later saves. Missing
historical split manifests cannot be retroactively certified. Diagonal-only
covariance checkpoints are also explicitly approximate.

Exact equality is demonstrated below on CPU synthetic runs in a matching
runtime. Restored RNG alone is not a guarantee across different software,
hardware or nondeterministic CUDA kernels. Target-GPU verification remains
required before a long ablation study.

## Optimizer compatibility

Existing configured learning rates/decays remain unchanged. Base SGD keeps its
existing hard-coded network LR of `0.01`. Base AdamW keeps network `init_lr`
and the existing `0.01`-based quantum projector/metric rates. Incremental
SGD/AdamW use network `init_lr` and AE `ae_init_lr` / `ae_weight_decay`.

Adding the omitted baseline AE group corrects a bug but changes historical
AdamW baseline trajectories. Do not pool the old behavior with corrected
ablations. Current supplied SGD configs retain their optimizer rates.

The optimizer checks structural coverage of parameters with `requires_grad`;
this does not assert every parameter has a nonzero gradient in every batch.
Frozen QKSR parameters remain outside the optimizer while input gradients
continue to flow.

## Read-only diagnostics

`task_diagnostics` is saved in new checkpoints; summaries are also logged.
Each incremental task records:

- Paired drift sample IDs and the training statistics sample count.
- Per-old-class stored-mean update L2 norm.
- Per-old-class stored-covariance update Frobenius norm and trace before/after.
- Pre/post CA old/new/total and per-task accuracy, tagged `validation` or `test`.
- Whether classifier alignment actually ran.

Covariance-update norm is zero under the unchanged retention policy. It is
NOT a measurement that old-image feature covariance stayed constant. No old
training images are accessed as an oracle. Confirming the true distribution
error remains a separate research experiment.

Diagnostics preserve global RNG and model weights. Extra evaluation does not
advance training sampler streams. Full old-covariance snapshots add temporary
CPU memory proportional to stored covariance size and pre/post CA evaluation
adds inference time; account for this overhead when profiling later.

## Validation and reproducible command

```powershell
python -B -m unittest discover -s tests -v
git diff --check
```

The integrity suite uses six-dimensional toy features and 48 synthetic images,
the production task/data/loss/CA/checkpoint paths, and an appropriately sized
instance of the existing AE. No pretrained ViT or real dataset training is run.

Regression coverage includes deterministic views and source IDs; invalid
pairing; exact train-only means/covariances; no test access while tuning;
unchanged old covariance; Python/NumPy/Torch RNG roundtrip; sampler continuation;
checkpoint reconstruction ordering; metadata/split mismatches; malformed and
legacy checkpoints; diagonal approximation; optimizer groups and actual AE
updates; frozen/trainable metric handling; and read-only diagnostics.

Integration compares uninterrupted three-task runs with save/load at task 0:
RSIAT and QKSR, both validation and final-test modes. It compares weights,
class statistics, diagnostics, partitions and RNG/generator states exactly.
An additional QKSR case uses one persistent worker with augmentation consuming
Python, NumPy and Torch RNG. CUDA checks skip on a CPU-only machine.

Result on 2026-10-10: **39 tests, 37 passed, 2 CUDA skips**, in 71.289 seconds.
Runtime: Python 3.12.5, PyTorch 2.12.1+cpu, NumPy 2.3.3. There are 22 new
integrity tests and 17 existing quantum/protocol tests. `git diff --check`
passed. AST comparison against the base commit confirms `_inc_loss`,
`_compute_rt_loss`, `RS_Loss`, `displacement` and `displacement_cov` are unchanged.
Architecture and experiment configuration files were not edited.

SHA-256 comparison before/after confirms all nine existing files under `logs/`
are unchanged; no existing `ckpt/` files were present. Tests create checkpoints
only in temporary directories. Two notebook deletions and two tracked bytecode
deletions appeared outside this work and are intentionally excluded from the
integrity commit.

## Unchanged hypotheses / remaining limits

- Covariance compensation quality, Gaussian replay fidelity, projector bias,
  `old_proj` versus `current`, and quantum advantage remain untested hypotheses.
- No changes to the residual sigmoid projector, representation losses, gamma
  calibration policy, covariance transport algorithm or experiment JSONs.
- Historical logs/checkpoints are not repaired and remain historical evidence.
  Use a fresh `output_root` for new runs; existing output-retention behavior in
  `trainer.py` has not been changed.
- CUDA and full ViT/dataset equality have not been exercised locally. Multiple
  seeds are still needed for scientific conclusions after the protocol fixes.
