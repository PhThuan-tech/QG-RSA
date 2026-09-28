# QKSR implementation notes

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
  the optimizer.  With both QKSR flags disabled, the original optimizer path is
  preserved;
- the RBF numerical floor is implemented as a convex mixture with a constant
  kernel, avoiding a hard element-wise floor that could break PSD.

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

