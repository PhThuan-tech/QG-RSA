#!/usr/bin/env bash
set -euo pipefail

# Directly runnable classical RSIAT baselines.
python main.py --config exps/rsiat/cifar224.json
python main.py --config exps/rsiat/cub.json
python main.py --config exps/rsiat/imageneta.json
python main.py --config exps/rsiat/imagenetr.json
python main.py --config exps/rsiat/omnibenchmark.json
python main.py --config exps/rsiat/vtab.json

# Q-BRIDGE is staged. Train the paired task-0 base first; see exps/README.md.
# python main.py --config exps/qbridge/sources/cifar224.json \
#   --overlay exps/qbridge/stages/train_base.json
