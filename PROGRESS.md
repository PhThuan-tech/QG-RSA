# KeepLoRA Replacement Progress

## Completed

- Added a KeepLoRA-only ViT path for RSIAT. The existing RSIAT AdaptFormer
  branch is disabled for this path; pretrained ViT loading, the transformer
  block layout, and Q/K/V/O projections remain RSIAT's implementation.
- Added one KeepLoRA module for every selected RSIAT attention projection:
  q_proj, k_proj, v_proj, and proj.
- Reused KeepLoRA's task algorithm: PTM principal-weight subspaces,
  projected first-step gradients, SVD initialization, frozen A/trainable B,
  subtract-before-training, merge-after-training, and cumulative dominant
  task-feature subspaces.
- Added a small controller which maps the KeepLoRA lifecycle to RSIAT's split
  Q/K/V/O weights. It does not synthesize CLIP's packed in_proj_weight.
- Kept RSIAT's cosine classifier, base representation-steering loss, residual
  autoencoder alignment, prototype/classifier stages, task order, evaluation,
  and task-boundary checkpoint flow.
- Added the keeplora_cifar224_smoke.json configuration and documented how to run it.

## Verification completed

- python -m py_compile passes for every changed Python source.
- python -m json.tool exps/keeplora_cifar224_smoke.json passes.
- git diff --check reports no newly introduced whitespace errors.

## Not yet verified

- End-to-end training has not run in this Codex environment because its
  available Python interpreter does not include torch. No dependencies or
  CUDA stack were changed automatically.
- The next execution should run the one-task smoke configuration, then resume
  it for a second task to verify the saved principal and feature bases load
  correctly before launching a full RSIAT-versus-KeepLoRA benchmark.

## Review Update

- Rechecked the original KeepLoRA helpers and trainer. CustomCLIP, Peft_ViT,
  Peft_Text, prompt processing, and MTIL classifiers are intentionally not
  imported because they replace RSIAT's ViT/PTM, classifier, or CIL protocol.
- The retained components are KeepLoRA's low-rank block, the principal
  subspace update algorithm, projected-gradient initialization, and
  subtract/merge lifecycle.
- The default PTM weight threshold is now 0.85 and task-feature threshold is
  0.99, matching the KeepLoRA configuration. Batch limits set to zero process
  a complete task; the smoke configuration deliberately keeps small limits.
- Full-task feature accumulation uses a fixed-size second-moment matrix rather
  than retaining all ViT token activations. Its left singular vectors recover
  the same dominant input directions used by the principal-subspace update.
- Added keeplora_cifar224.json, which mirrors adapter_cifar224.json in all
  RSIAT settings and changes only the PEFT mechanism and its KeepLoRA settings.
- Gradient SVD now uses torch.svd_lowrank with the original KeepLoRA sketch
  size and iteration count rather than a full SVD.
- Added matched KeepLoRA configurations for CUB, ImageNet-A, ImageNet-R,
  OmniBenchmark, and VTAB. Each preserves the matching RSIAT configuration's
  dataset protocol, optimizer, epochs, classifier settings, and losses.
- Fixed feature accumulation so it runs only in RSIAT's explicit end-of-task
  pass. This avoids collecting feature statistics during normal optimization.
- Made Q/K/V/O target order deterministic, validated KeepLoRA configuration
  values at model construction, and checkpointed all KeepLoRA settings for
  compatible task-boundary resume validation.
- Restored the original KeepLoRA threshold-selection convention for PTM and
  feature principal subspaces while preserving guards for zero-energy inputs.
- Matched KeepLoRA's training mode for the gradient-estimation pass; the
  explicit feature-collection pass remains evaluation mode as in the original.
- Matched KeepLoRA's `accum_loader` lifecycle: gradient initialization and
  end-of-task feature accumulation now use a separate shuffled, drop-last
  loader, while RSIAT's main train/test loaders are unchanged.
- Replaced direct SVD of the feature second-moment matrix with a compact
  factor that preserves both the original feature matrix's singular vectors
  and singular-value energy. The KeepLoRA threshold therefore selects the
  same subspace dimension without storing all token features.
- Audit note: RSIAT's semantic mean-shift compensation is active. Its
  `displacement_cov` helper is dead code in the original repository (it has
  no caller and references an undefined `cov_computation`), so no covariance
  shift compensation is executed by either baseline or replacement.
- Corrected feature-subspace state so `PrinSubspace` receives the combined
  PTM-weight and previous-task basis, and evaluates its energy threshold on
  the original task feature factor. This now follows the original KeepLoRA
  update order instead of applying the threshold only to a pre-projected
  residual.
- Added log-only RSIAT semantic-shift diagnostics: mean per-sample feature
  drift, norm of mean drift, old-prototype drift, and proposed/applied SSC
  shift statistics. These metrics read existing arrays and do not alter any
  RSIAT training, compensation, classifier, or KeepLoRA behavior.
