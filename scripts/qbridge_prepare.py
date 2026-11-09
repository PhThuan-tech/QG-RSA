"""Offline CLI: extract base-train cache, fit R6 calibration, fit R7/R8 selectors."""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from utils.qbridge import kernel_from_config, sha256_file, trusted_load, write_json
from utils.qbridge_calibration import fit_calibrator, fit_classical_selector, metric_config_from_args
from utils.qbridge_experiment import preflight_base


def extract_cache(args, checkpoint_path, output, device):
    from data.data_manager import DataManager
    from models.RSIAT_adapter import Learner
    from torch.utils.data import DataLoader
    from trainer import _set_random
    args = copy.deepcopy(args)
    args["qbridge"] = {"enabled": False, "base_checkpoint_path": str(checkpoint_path)}
    preflight_base(args)
    _set_random(args["seed"])
    args["device"] = [torch.device(device)]
    dm = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"],
                     data_root=args.get("data_root"), offline=True)
    learner = Learner(args)
    learner.class_order = list(dm._class_order)
    learner.load_checkpoint(checkpoint_path)
    learner._network.eval()
    dataset = dm.get_dataset(np.arange(args["init_cls"]), source="train", mode="test")
    sample_ids = np.concatenate([np.flatnonzero(dm._train_targets == c) for c in range(args["init_cls"])]).tolist()
    loader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=False,
                        num_workers=int(args.get("num_workers", 0)))
    features, labels, scores = [], [], []
    with torch.no_grad():
        for _, x, y in loader:
            z = learner._network.extract_vector(x.to(device))
            features.append(z.cpu())
            labels.append(y)
            scores.append(learner._network.fc(z)["logits"].cpu())
    cache = {"kind": "base_feature_cache", "source": "base_train_only",
             "base_checkpoint_hash": sha256_file(checkpoint_path), "dataset": args["dataset"],
             "seed": args["seed"], "class_order": learner.class_order,
             "sample_ids": sample_ids, "transforms": "legacy deterministic evaluation transforms",
             "features": torch.cat(features), "labels": torch.cat(labels), "scores": torch.cat(scores)}
    save_artifact(output, cache)


def save_artifact(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)
    report = {k: v for k, v in artifact.items() if k not in ("state_dict", "features", "labels", "scores")}
    report["artifact_sha256"] = sha256_file(path)
    write_json(path.with_suffix(".json"), report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "calibrate", "rbf", "mlp"))
    parser.add_argument("--config", required=True, help="Full original base experiment config")
    parser.add_argument("--base", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mlp-kind", choices=("mlp_small", "mlp_cap"), default="mlp_small")
    parser.add_argument("--proxy-margin", type=float, default=.1)
    cli = parser.parse_args()
    if cli.steps < 1 or cli.epochs < 1 or cli.batch_size < 2 or (cli.lr is not None and cli.lr <= 0):
        parser.error("Positive fit steps/epochs/lr and batch-size>=2 are required")
    args = json.loads(Path(cli.config).read_text(encoding="utf-8"))
    args["seed"] = cli.seed
    if cli.data_root:
        args["data_root"] = cli.data_root
    args["qbridge"] = {"enabled": False, "base_checkpoint_path": cli.base}
    preflight_base(args)
    if cli.command == "cache":
        extract_cache(args, cli.base, cli.output, cli.device)
    else:
        if not cli.cache:
            parser.error("--cache is required for artifact fitting")
        cache = trusted_load(cli.cache)
        checkpoint = trusted_load(cli.base)
        config = metric_config_from_args(args, args.get("q_kernel_type", "pqk"))
        pqk = kernel_from_config(config).to(cli.device)
        pqk.load_state_dict(checkpoint["quantum_kernel_state_dict"], strict=True)
        base_hash = sha256_file(cli.base)
        if cli.command == "calibrate":
            artifact = fit_calibrator(cache, pqk, base_hash, seed=cli.seed, proxy_margin=cli.proxy_margin,
                                      steps=cli.steps, lr=cli.lr or .03, device=cli.device)
        else:
            config["kernel_type"] = "rbf_proj" if cli.command == "rbf" else cli.mlp_kind
            artifact = fit_classical_selector(cache, pqk, config, base_hash, args, seed=cli.seed,
                                              epochs=cli.epochs, batch_size=cli.batch_size,
                                              lr=cli.lr or .01, device=cli.device)
        artifact["feature_cache_sha256"] = sha256_file(cli.cache)
        save_artifact(cli.output, artifact)
        if artifact.get("status") != "fitted":
            raise SystemExit("Artifact gate failed; see report JSON. R6 must not run with this artifact.")
    print(str(Path(cli.output).resolve()))


if __name__ == "__main__":
    main()
