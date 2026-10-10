"""Prepare and run the fixed nine-run study using only manually uploaded assets."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Set before importing timm/Hugging Face, including in spawned subprocesses.
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})

from utils.failure_localization import SEEDS, VARIANTS, TaskRecorder, build_configs, validate_configs, write_json, task_line
from utils.offline_assets import resolve_data_root, resolve_weights, extract_cifar_archive, sha256_file, no_network


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def source_fingerprints(source):
    paths = [REPO_ROOT / "main.py", REPO_ROOT / "trainer.py", Path(source)]
    for folder in ("data", "models", "network", "utils", "scripts"):
        paths.extend((REPO_ROOT / folder).rglob("*.py"))
    return {str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p): sha256_file(p)
            for p in sorted(set(paths))}


def dataset_digest(manager):
    """Content hash of both splits, independent of a mounted directory name."""
    import numpy as np
    digest = hashlib.sha256()
    for images, labels in ((manager._train_data, manager._train_targets),
                           (manager._test_data, manager._test_targets)):
        digest.update(np.asarray(labels, dtype=np.int64).tobytes())
        if manager.use_path:
            for image in images:
                digest.update(bytes.fromhex(sha256_file(image)))
        else:
            digest.update(np.ascontiguousarray(images).tobytes())
    return digest.hexdigest()


def preflight(configs, require_gpu=False):
    """Load real dataset and local backbone on CPU before spending GPU budget."""
    import numpy as np
    import torch
    from data.data_manager import DataManager
    from utils.inc_net import get_convnet
    from utils.experimental_integrity import preserve_rng_state

    common = validate_configs(configs)
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("Training requires a Kaggle GPU. No CPU benchmark or fake results will be produced.")
    if len(common["device"]) != 1 or str(common["device"][0]) != "0":
        raise ValueError("This controlled runner uses one GPU, device 0, for every run.")
    if common["model_name"] != "adapter" or common["convnet_type"] not in {
        "pretrained_vit_b16_224_in21k_adapter", "pretrained_vit_b16_224_adapter",
    }:
        raise ValueError("Unsupported backbone for the offline study.")
    with no_network(), preserve_rng_state(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        manager = DataManager(common["dataset"], common["shuffle"], SEEDS[0],
                              common["init_cls"], common["increment"],
                              data_root=common["data_root"], offline=True)
        expected = 100 if common["dataset"] == "cifar224" else 200
        if manager.get_total_classnum() != expected:
            raise ValueError("Expected {} classes; check the uploaded dataset splits.".format(expected))
        train_counts = np.bincount(manager._train_targets, minlength=expected)
        test_counts = np.bincount(manager._test_targets, minlength=expected)
        if (train_counts < 2).any() or (test_counts == 0).any():
            raise ValueError("Each class needs >=2 training images and >=1 test image.")
        for source in ("train", "test"):
            dataset = manager.get_dataset([0], source=source, mode="test")
            _, image, _ = dataset[0]
            if tuple(image.shape) != (3, 224, 224) or not torch.isfinite(image).all():
                raise ValueError("Unexpected offline image transform output.")
        fingerprint = dataset_digest(manager)
        backbone = get_convnet(common, pretrained=True)
        if not any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("No adapter parameters remain trainable after weight loading.")
        del backbone
        gc.collect()
    versions = {}
    for name in ("torch", "torchvision", "timm", "numpy", "scipy", "easydict", "tqdm"):
        versions[name] = importlib.metadata.version(name)
    return {
        "python": sys.version, "versions": versions, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_root": common["data_root"], "pretrained_path": common["pretrained_path"],
        "weights_sha256": sha256_file(common["pretrained_path"]), "dataset_sha256": fingerprint,
        "class_orders": {str(seed): (np.random.RandomState(seed).permutation(expected).tolist()
                                    if common["shuffle"] else list(manager._class_order)) for seed in SEEDS},
        "task_sizes": list(manager._increments),
        "training_images": len(manager._train_targets), "test_images": len(manager._test_targets),
        "config_sha256": {c["prefix"] + "_" + str(c["seed"][0]): config_digest(c) for c in configs},
        "metric_policy": {
            "aia": "mean post-CA total accuracy over every task including base",
            "average_forgetting": "mean F_t over incremental tasks; F_t=mean prior-best minus current old-task accuracy",
            "training_seconds": "adapter-stage plus CA-stage wall time; includes their calibration and epoch evaluation",
            "task_seconds": "full incremental_train + final evaluation + after_task; excludes checkpoint I/O",
        },
    }


def prepare(args):
    source = Path(args.source or REPO_ROOT / "exps" / ("adapter_" + args.dataset + ".json")).resolve()
    base = json.loads(source.read_text(encoding="utf-8"))
    if base["dataset"] != args.dataset:
        raise ValueError("Source config dataset does not match --dataset.")
    defaults = json.loads((REPO_ROOT / "exps/adapter_cifar224.json").read_text(encoding="utf-8"))
    configs = build_configs(base, defaults)
    study_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    study = Path(args.output_root).resolve() / ("failure_localization_" + args.dataset + "_" + study_id)
    # Weight discovery happens before extraction or creating output directories.
    weights = resolve_weights(args.input_root, args.pretrained_path)
    if args.cifar_archive:
        if args.dataset != "cifar224" or args.data_root:
            raise ValueError("--cifar-archive is only for CIFAR224 without --data-root.")
        data_root = extract_cifar_archive(args.cifar_archive, study / "assets" / "cifar100")
    else:
        data_root = resolve_data_root(args.dataset, args.input_root, args.data_root)
    for config in configs:
        config.update({"data_root": str(data_root), "pretrained_path": str(weights), "output_root": str(study)})
    result = preflight(configs)
    result["source_sha256"] = source_fingerprints(source)
    study.mkdir(parents=True, exist_ok=args.cifar_archive is not None)
    (study / "configs").mkdir()
    write_json(study / "source_config.json", base)
    write_json(study / "preflight.json", result)
    entries = []
    for config in configs:
        name = config["prefix"] + "_" + str(config["seed"][0])
        file = "configs/" + name + ".json"
        write_json(study / file, config)
        entries.append({"run_id": name, "config": file})
    write_json(study / "study_manifest.json", {"schema_version": 1, "study_id": study_id,
               "source_config": str(source), "runs": entries,
               "status": "prepared_not_trained", "seeds": list(SEEDS), "variants": list(VARIANTS)})
    print("Offline preflight passed; 9 configs prepared: {}".format(study))
    return study


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measured_stages(model, timings):
    originals = {}
    def wrap(original, role):
        def measured(*args, **kwargs):
            synchronize()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                synchronize()
                timings[role] = timings.get(role, 0) + time.perf_counter() - start
        return measured
    for name, role in (("_train", "adapter"), ("_stage2_compact_classifier", "ca")):
        originals[name] = (name in model.__dict__, getattr(model, name))
        setattr(model, name, wrap(getattr(model, name), role))
    try:
        yield
    finally:
        for name, (was_instance, original) in originals.items():
            if was_instance:
                setattr(model, name, original)
            else:
                delattr(model, name)


def state_digest(model):
    import torch
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        tensor = value.detach().cpu().contiguous()
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def close_workers(model):
    for name in ("train_loader", "val_loader", "test_loader"):
        iterator = getattr(getattr(model, name, None), "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def execute_run(config, directory, expected_order, expected_tasks):
    """Production learner/task loop; no intermediate checkpoints or task resume."""
    import torch
    from data.data_manager import DataManager
    from trainer import _set_random, _set_device
    from utils.model_factory import get_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    args = copy.deepcopy(config)
    args["seed"] = args["seed"][0]
    console = sys.stdout
    root = logging.getLogger()
    old_handlers, old_level = root.handlers[:], root.level
    handler = logging.FileHandler(directory / "train.log", mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.handlers, root.level = [handler], logging.INFO
    model = None
    start = time.perf_counter()
    try:
        with no_network(), (directory / "runtime_output.log").open("x", encoding="utf-8") as output, redirect_stdout(output), redirect_stderr(output):
            _set_random(args["seed"])
            _set_device(args)
            manager = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"],
                                  data_root=args["data_root"], offline=True)
            if list(manager._class_order) != list(expected_order) or list(manager._increments) != list(expected_tasks):
                raise ValueError("Runtime class order/task sizes differ from the preflight plan.")
            model = get_model(args["model_name"], args)
            model.class_order = list(manager._class_order)
            write_json(directory / "run_manifest.json", {"config": config, "config_sha256": config_digest(config),
                       "class_order": model.class_order, "task_sizes": list(manager._increments), "status": "started"})
            recorder = TaskRecorder(directory, args["prefix"], args["seed"])
            base_hash = None
            for task in range(manager.nb_tasks):
                synchronize()
                task_start = time.perf_counter()
                timings = {}
                with measured_stages(model, timings):
                    model.incremental_train(manager)
                result = model.eval_task()
                model.after_task()
                synchronize()
                task_seconds = time.perf_counter() - task_start
                diagnostic = model.task_diagnostics[-1] if task > 0 else None
                row = recorder.add(task, args, result, diagnostic, timings, task_seconds)
                model.cnn_curve = {"top1": [r["post_ca_total"] for r in recorder.rows], "top5": []}
                model.task_accuracy_matrix = recorder.post_matrix
                if task == 0:
                    base_hash = {"network": state_digest(model._network),
                                 "quantum": state_digest(model.quantum_kernel) if model.quantum_kernel is not None else None}
                print(task_line(row), file=console, flush=True)
            checkpoint_start = time.perf_counter()
            model.save_checkpoint(str(directory / "final_task.pkl"))
            summary = recorder.summary(manager.nb_tasks, time.perf_counter() - start)
            summary.update({"base_state_sha256": base_hash, "checkpoint_seconds": time.perf_counter() - checkpoint_start,
                            "final_checkpoint": "final_task.pkl", "status": "complete"})
            write_json(directory / "run_summary.json", summary)
        return summary
    except Exception as error:
        logging.exception("Failure localization run did not complete.")
        write_json(directory / "FAILED.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if model is not None:
            close_workers(model)
        handler.close()
        root.handlers, root.level = old_handlers, old_level
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_study(study, variants, seeds):
    if not variants or not seeds or not set(variants).issubset(VARIANTS) or not set(seeds).issubset(SEEDS):
        raise ValueError("Select at least one valid variant and seed.")
    if not variants or not seeds or not set(variants).issubset(VARIANTS) or not set(seeds).issubset(SEEDS):
        raise ValueError("Select at least one valid variant and seed.")
    study = Path(study).resolve()
    plan = json.loads((study / "study_manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((study / "preflight.json").read_text(encoding="utf-8"))
    configs = [json.loads((study / item["config"]).read_text(encoding="utf-8")) for item in plan["runs"]]
    if saved["source_sha256"] != source_fingerprints(plan["source_config"]):
        raise ValueError("Source code changed after preflight; prepare a new study.")
    live = preflight(configs, require_gpu=True)
    for field in ("config_sha256", "weights_sha256", "dataset_sha256", "class_orders", "task_sizes", "versions"):
        if live[field] != saved[field]:
            raise ValueError("{} changed after preflight; prepare a new study.".format(field))
    selected = [(entry, config) for entry, config in zip(plan["runs"], configs)
                if config["prefix"] in variants and config["seed"][0] in seeds]
    for entry, _ in selected:
        if (study / "runs" / entry["run_id"]).exists():
            raise FileExistsError("Run already exists; no overwrite or task resume: {}".format(entry["run_id"]))
    for entry, config in selected:
        execute_run(config, study / "runs" / entry["run_id"],
                    saved["class_orders"][str(config["seed"][0])], saved["task_sizes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run-study", type=Path)
    parser.add_argument("--dataset", choices=("cifar224", "imageneta", "imagenetr"), default="cifar224")
    parser.add_argument("--source")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--data-root")
    parser.add_argument("--cifar-archive")
    parser.add_argument("--pretrained-path")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare(args)
        else:
            run_study(args.run_study, args.variants, args.seeds)
    except Exception as error:
        print("Offline study stopped: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
