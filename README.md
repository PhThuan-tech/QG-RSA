# QG-RSA: RSIAT with quantum components and statistics–classifier fusion

This repository extends RSIAT (classical ViT-adapter, representation steering,
SSCA and classifier alignment) and evaluates several quantum components under
a measurement protocol of paired replicates, in-run multi-head evaluation and
pre-registered rules.

| Path | Content |
|---|---|
| `exps/` | Every experiment: specification, overlays, results and analysis. Start with [`exps/README.md`](exps/README.md), which also fixes the seed convention (1993, 2015, 2026). |
| `paper/` | LNCS manuscript; `paper/make_figures.py` regenerates the figures from `exps/`. |
| `QGRSA_EXPERIMENTS.ipynb` | Offline Kaggle notebook for all current rounds (built by `scripts/build_experiments_notebook.py`). |
| `utils/qsd.py`, `scripts/summarize_qsd*.py` | QSD-CIL heads and statistics–classifier fusion (current direction). |
| `utils/ca_heads.py`, `utils/quantum_kernel.py`, `utils/qbridge*.py` | Earlier quantum directions (QRF-CA, QKSR, Q-BRIDGE), kept for reproducibility. |

Current result: a training-free fusion of the CA classifier with a generative
score from the stored class statistics beats RSIAT on every held-out run
(`exps/qsd/results/a2/README.md`). The quantum version of that score (a
pretty-good measurement) does not beat its classical ridge counterpart.

## Installation

The original environment used Ubuntu 20.04, Python 3.10, and CUDA 11.8. Install
the packages in `requirements.txt`. For an offline notebook runtime, prepare the
wheels listed in `requirements-colab.txt`.

Expected local datasets include `cifar-100-python`, `imagenet-a`, `imagenet-r`,
`cub`, `omnibenchmark`, and `vtab`. Dataset paths may also be supplied through
the `data_root` config key.

## Experiment configurations

Configuration roles and commands are documented in [`exps/README.md`](exps/README.md).

```text
exps/
├── rsiat/                 # directly runnable classical baselines (6 datasets)
├── smoke/                 # short pipeline checks
├── <experiment>/          # diagnostics, measurement, upper, qksr, qrfca, qsd:
│   ├── README*.md         #   pre-registered specification
│   ├── <variant>.json     #   overlays used by QGRSA_EXPERIMENTS.ipynb
│   └── results/           #   logs, summaries, analysis
├── qbridge/               # earlier Q-BRIDGE configs (inactive)
└── archive/               # legacy single runs and old configs
```

Run a baseline directly:

```bash
python main.py --config exps/rsiat/cifar224.json
python main.py --config exps/rsiat/imageneta.json
```

### Earlier direction: Q-BRIDGE (inactive)

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
python scripts/build_experiments_notebook.py QGRSA_EXPERIMENTS.ipynb
python scripts/package_qbridge_kaggle_repo.py   # dist/QG-RSA_QBRIDGE_Kaggle.zip for Kaggle
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
