# RSIAT baseline and Q-BRIDGE

This repository contains two active experiment paths:

1. **RSIAT baseline** reproduces the classical ViT-adapter, representation
   steering, SSCA, and classifier-alignment pipeline.
2. **Q-BRIDGE** starts every branch from the same QKSR task-0 checkpoint and
   studies risk-weighted cross-task margins and old-anchor guards.

The QKSR documents and quantum-kernel tests remain as technical background
because Q-BRIDGE uses a frozen PQK trained on the base task. Standalone QKSR
experiments are no longer part of the active configuration layout.

## Installation

The original environment used Ubuntu 20.04, Python 3.10, and CUDA 11.8. Install
the packages in `requirements.txt`. For an offline notebook runtime, prepare the
wheels listed in `requirements-colab.txt`.

Expected local datasets include `cifar-100-python`, `imagenet-a`, `imagenet-r`,
`cub`, `omnibenchmark`, and `vtab`. Dataset paths may also be supplied through
the `data_root` config key.

## Experiment configurations

Configuration roles and commands are documented in [`exps/README.md`](exps/README.md).
The active layout is:

```text
exps/
├── rsiat/                 # directly runnable classical baselines
├── smoke/                 # short pipeline checks
└── qbridge/
    ├── sources/           # dataset/training settings shared by all branches
    ├── stages/            # stage overlays, including task-0 base training
    └── overlays/          # R0-R10 contribution/control overlays
```

Run a baseline directly:

```bash
python main.py --config exps/rsiat/cifar224.json
python main.py --config exps/rsiat/imageneta.json
```

Train a local Q-BRIDGE task-0 base:

```bash
python main.py \
  --config exps/qbridge/sources/cifar224.json \
  --overlay exps/qbridge/stages/train_base.json
```

R0-R10 require the exact base checkpoint and, for R6/R7/R8, fitted base-only
artifacts. Generate complete runtime configs with
`scripts/generate_qbridge_configs.py`; do not run the overlay JSON files alone.
See [`docs/QBRIDGE_RUNBOOK.md`](docs/QBRIDGE_RUNBOOK.md) for the staged workflow.

For Kaggle with Internet disabled, use `QBRIDGE_Kaggle.ipynb`. The notebook
accepts a repository ZIP, dataset input, and pretrained ViT tensor.

## Checkpoint and resume

Classical RSIAT saves task-boundary checkpoints under:

```text
ckpt/<prefix>/<dataset>/<init_cls>_<increment>/seed_<seed>/task_<N>.pkl
```

Set `"resume": true` to continue from the latest completed task, or provide a
specific `resume_path`. This restores the network and class statistics but does
not restore optimizer, scheduler, DataLoader, or full RNG state. Q-BRIDGE
branches intentionally restart from their paired task-0 base and do not support
incremental resume.

## Validation

```bash
python -B -m unittest discover -s tests -q
python scripts/build_qbridge_kaggle_notebook.py
python scripts/package_qbridge_kaggle_repo.py
```

Unit tests establish source behavior only. Accuracy and contribution claims
require completed paired dataset runs with the same base checkpoint, data
protocol, class order, seeds, and training budget.

## Citation

```text
@inproceedings{zhao2026representation,
  title={Representation-Steered Incremental Adapter-Tuning for Class-Incremental Learning with Pre-Trained Models},
  author={Zhao, Jiarui and Huang, Libo and Li, Xiangqi and An, Zhulin and Yang, Chuanguang and Wang, Yu and Diao, Boyu and Xu, Yongjun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18010--18020},
  year={2026}
}
```

The upstream RSIAT implementation is based on PILOT and SSIAT.
