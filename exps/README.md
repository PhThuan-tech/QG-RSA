# Experiment configuration layout

## RSIAT baseline

Files in `rsiat/` are complete, directly runnable classical RSIAT configs. They
explicitly disable the quantum kernel and Q-BRIDGE:

```bash
python main.py --config exps/rsiat/cifar224.json
python main.py --config exps/rsiat/imageneta.json
```

Change runtime-only values such as `device`, `seed`, `data_root`,
`pretrained_path`, and `output_root` in a copy outside this directory when
running a machine-specific experiment.

## Q-BRIDGE

`qbridge/sources/` contains the full dataset and optimizer settings shared by
the paired base and every incremental branch. These source configs enable PQK
on task 0 and freeze it for incremental tasks.

Train only the paired base task by adding the stage overlay:

```bash
python main.py \
  --config exps/qbridge/sources/cifar224.json \
  --overlay exps/qbridge/stages/train_base.json
```

The resulting checkpoint is normally:

```text
ckpt/qbridge_base/<dataset>/<init_cls>_<increment>/seed_<seed>/task_0.pkl
```

`qbridge/overlays/R0.json` through `R10.json` are locked method/control
overlays. They contain checkpoint and artifact placeholders and are not
standalone configs. Create resolved configs in an ignored runtime directory:

```bash
python scripts/generate_qbridge_configs.py \
  --source exps/qbridge/sources/cifar224.json \
  --base <task_0.pkl> \
  --artifact-root <base-only-artifact-directory> \
  --output-dir outputs/configs/cifar224 \
  --seeds 1993 2015 2026
```

R5 is the raw-PQK full pilot. R6 is the full calibrated contribution. R6, R7,
and R8 require artifacts produced from base-training data by
`scripts/qbridge_prepare.py`.

## Smoke config

`smoke/rsiat_cifar224.json` runs one short task with one epoch. It checks the
pipeline and checkpoint path; it is not benchmark evidence.
