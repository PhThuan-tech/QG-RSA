# QKSR implementation notes

For the 2026-10-03 source audit, opt-in improvements and Kaggle ablations, see
[QKSR_LIMITATIONS_AND_IMPROVEMENTS.md](QKSR_LIMITATIONS_AND_IMPROVEMENTS.md).
Legacy loss defaults remain unchanged; a better ImageNet accuracy is not yet
established by the local tests.

For independent Kaggle experiments on ImageNet-A, ImageNet-R or CIFAR224, use
[`QKSR_Kaggle_Ablations.ipynb`](../QKSR_Kaggle_Ablations.ipynb). It provides
one run cell per `A`, `B`, and `G0`-`G8` configuration, with separate smoke and
full prefixes/checkpoints.

The implementation follows `QKSR_Spec_for_RSIAT_v3.1.md` and includes the
review errata agreed after v3.1:

- gamma calibration accepts either self-distances or rectangular
  prototype–feature cross-distances; only self-distance calibration removes a
  diagonal;
- calibration uses a deterministic evaluation-transform view of a fixed subset
  and a private `torch.Generator`, so it does not move the training RNG stream;
- module initialization seeds and restores the CPU generator without touching
  CUDA RNG;
- when QKSR incremental training is enabled with AdamW, `old_ae` is included in
  the optimizer. Opt-in signed/reset projectors or relation retention also
  include it for the non-quantum AdamW control; unmodified RSIAT defaults keep
  the original optimizer path;
- the RBF numerical floor is implemented as a convex mixture with a constant
  kernel, avoiding a hard element-wise floor that could break PSD.

## Experimental retention profile

`utils/qksr_profiles.py` contains shared options for QKSR and a non-quantum
control. `QKSR_PROFILE='retention'` in the Kaggle notebook selects paired
training-only drift features, a signed identity-initialized projector reset at
each task, detached repulsion anchors, cosine relation retention, guarded
low-rank mean/covariance transport, and diagonal covariance shrinkage.
`G0` through `G6` configs isolate these mechanisms; `G7` substitutes the matched
RBF metric and `G8` applies the shared changes to RSIAT. These settings are
untuned experimental candidates, not established ImageNet improvements.

Validation-enabled runs now evaluate all seen held-out classes rather than
test data. Stage probes report pre/post classifier alignment. Full covariance
must be retained for transport and exact resume. Partial backbone or task
checkpoint loads fail clearly instead of silently retaining random weights.

## Local verification

No package installation or local environment creation is needed. Run:

```bash
python -B -m unittest discover -s tests -v
```

The CUDA RNG test is skipped on a CPU machine and must be exercised once in the
target Colab GPU runtime.

## Pilot sequence

Run the paired baseline and QKSR configs first, then generate the minimum
ablation configs:

```bash
python main.py --config=exps/adapter_cub_baseline.json
python main.py --config=exps/adapter_cub.json
python scripts/generate_qksr_ablation_configs.py
```

Full dataset training was not performed during local implementation because it
requires the prepared datasets, pretrained weights, and target GPU runtime.

## Package source for Kaggle

Create a checked, minimal upload archive with:

```powershell
py -3.12 -X utf8 -B scripts\package_kaggle_repo.py
```

The command parses all selected Python/JSON/notebooks, runs the complete test
suite, packages only the source whitelist, and reopens the ZIP to verify its
member list, CRC, sizes and SHA-256 hashes. The output is
`artifacts/QG-RSA_Kaggle.zip`; datasets, model weights, logs, checkpoints,
caches and analysis outputs are excluded. The embedded
`_KAGGLE_PACKAGE_MANIFEST.json` records every member hash.
