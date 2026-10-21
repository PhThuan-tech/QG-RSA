import sys
import logging
import copy
import torch
import glob
import re
import numpy as np
from utils import model_factory
from data.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import random
import hashlib
import json


def _resolve_seed(value, run_seed, name):
    """Resolve an independent per-run seed without accepting implicit randomness."""
    if value is None or value == "{seed}":
        return int(run_seed)
    if isinstance(value, bool):
        raise TypeError("{} must be an integer seed.".format(name))
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError("{} must be an integer or '{{seed}}'.".format(name))


def _prepare_seeded_args(args, run_seed):
    prepared = copy.deepcopy(args)
    prepared["seed"] = int(run_seed)
    for name in ("class_order_seed", "train_seed", "partition_seed"):
        prepared[name] = _resolve_seed(prepared.get(name), run_seed, name)
    return prepared


def _sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def RSIAT_train(args):
    raw_seeds = copy.deepcopy(args["seed"])
    seed_list = [raw_seeds] if isinstance(raw_seeds, (int, np.integer)) else list(raw_seeds)
    if not seed_list:
        raise ValueError("seed must contain at least one run seed")
    device = copy.deepcopy(args["device"])

    sum_seed = 0.0
    for seed in seed_list:
        run_args = _prepare_seeded_args(args, seed)
        run_args["device"] = copy.deepcopy(device)
        # Every seed owns a full copy. Q-BRIDGE resolves artifact paths and
        # appends runtime state without contaminating the next seed.
        sum_seed += _train(run_args)
    avg_seed = sum_seed / len(seed_list)
    print('Average Seed Accuracy (CNN):', avg_seed)
    logging.info("Average Seed Accuracy (CNN): {}".format(avg_seed))


def find_latest_checkpoint(checkpoint_dir):
    """Return the highest-numbered completed task checkpoint, if one exists."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "task_*.pkl"))
    if not checkpoint_files:
        return None

    def task_number(filepath):
        match = re.search(r"task_(\d+)\.pkl$", os.path.basename(filepath))
        return int(match.group(1)) if match else -1

    return max(checkpoint_files, key=task_number)

def _train(args):
    try:
        return _train_impl(args)
    except Exception as exc:
        if args.get("_qbridge_report_dir") and args.get("_qbridge_base_hash"):
            from utils.qbridge import write_json
            write_json(os.path.join(args["_qbridge_report_dir"], "failure.json"),
                       {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                        "base_checkpoint_sha256": args["_qbridge_base_hash"],
                        "note": "No incremental checkpoints; restart from paired base after resolving failure"})
        raise


def _train_impl(args):

    algorithm = str(args.get("algorithm", "rsiat")).lower()
    if algorithm != "rsiat":
        raise ValueError("Unsupported algorithm {!r}.".format(algorithm))

    qb_experiment = bool(args.get("qbridge_experiment") or args.get("qbridge", {}).get("enabled"))
    base_path = None
    if qb_experiment:
        from utils.qbridge_experiment import preflight_base
        base_path = preflight_base(args)

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    output_root = args.get("output_root", "")
    logs_name = os.path.join(
        output_root, "logs", args["model_name"], args["dataset"],
        str(init_cls), str(args["increment"]),
    )
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(
            args["prefix"], args["seed"], args["convnet_type"],
        ),
    )
    args["_run_log_prefix"] = logfilename
    if qb_experiment:
        report_dir = logfilename + "_qbridge"
        if os.path.isdir(report_dir) and os.listdir(report_dir):
            raise ValueError("Q-BRIDGE output already exists; select a fresh output_root to avoid mixing runs")
        os.makedirs(report_dir, exist_ok=True)
        args["_qbridge_report_dir"] = report_dir
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if qb_experiment:
        class BridgeConsoleFilter(logging.Filter):
            def filter(self, record):
                return record.levelno >= logging.WARNING or record.getMessage().startswith("QBRIDGE")
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.addFilter(BridgeConsoleFilter())

    _set_random(args["train_seed"])
    _set_device(args)
    print_args(args)
    local_data = args.get("data_root") is not None
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["class_order_seed"],
        args["init_cls"],
        args["increment"],
        **({"data_root": args.get("data_root"),
            "offline": bool(args.get("offline", qb_experiment))}
           if qb_experiment or local_data else {}),
        class_order=args.get("_resolved_class_order"),
    )
    # Dataset discovery must not perturb model/training initialization.
    _set_random(args["train_seed"])
    args["_class_order_sha256"] = _sha256_json(data_manager._class_order)
    args["_task_schedule"] = list(data_manager._increments)
    model = model_factory.get_model(args["model_name"], args, algorithm=algorithm)
    model.class_order = list(data_manager._class_order)

    if qb_experiment:
        from utils.qbridge_experiment import run_manifest
        model.load_checkpoint(base_path)
        model.initialize_qbridge_branch()
        # Shared starting RNG stream for every variant, including legacy R0.
        _set_random(args["train_seed"])
        run_manifest(args, "initialized", class_order=model.class_order,
                     legacy_optimizer_policy="QKSR parameter groups; no literal old-row freeze claim")

    checkpoint_dir = os.path.join(
        output_root,
        "ckpt",
        str(args["prefix"]),
        str(args["dataset"]),
        "{}_{}".format(args["init_cls"], args["increment"]),
    )
    if args.get("isolate_runs", True):
        checkpoint_dir = os.path.join(checkpoint_dir, "seed_{}".format(args["seed"]))
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_task = 1 if qb_experiment else 0
    if args.get("resume", False):
        resume_path = args.get("resume_path") or find_latest_checkpoint(checkpoint_dir)
        if resume_path and os.path.isfile(resume_path):
            completed_task = model.load_checkpoint(resume_path)
            start_task = completed_task + 1
            logging.info("Resuming experiment from task %d", start_task)
        else:
            logging.info("Resume enabled but no checkpoint was found. Starting from task 0.")

    print()
    cnn_curve = getattr(model, "cnn_curve", {"top1": [], "top5": []})
    nme_curve = getattr(model, "nme_curve", {"top1": [], "top5": []})
    task_accuracy_matrix = getattr(model, "task_accuracy_matrix", [])
    if qb_experiment and (len(cnn_curve["top1"]) != 1 or len(task_accuracy_matrix) != 1):
        # Older base artifacts may lack curves: evaluate their fixed state once.
        from torch.utils.data import DataLoader
        from utils.qbridge_experiment import preserve_rng
        with preserve_rng():
            model.test_loader = DataLoader(data_manager.get_dataset(
                np.arange(args["init_cls"]), source="test", mode="test"),
                batch_size=args["batch_size"], shuffle=False, num_workers=0)
            saved_known = model._known_classes
            model._known_classes = 0
            base_metrics = model.eval_task()
            model._known_classes = saved_known
        cnn_curve = {"top1": [base_metrics["top1"]], "top5": [base_metrics["top5"]]}
        task_accuracy_matrix = [base_metrics["task_accuracies"]]
    if start_task >= data_manager.nb_tasks:
        if cnn_curve["top1"]:
            return sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        logging.info("All tasks are already complete, but no saved accuracy curve was found.")
        return 0.0

    end_task = data_manager.nb_tasks
    max_tasks_per_run = args.get("max_tasks_per_run")
    if max_tasks_per_run is not None:
        max_tasks_per_run = max(1, int(max_tasks_per_run))
        end_task = min(end_task, start_task + max_tasks_per_run)
        logging.info("This run will process tasks [%d, %d).", start_task, end_task)

    for task in range(start_task, end_task):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        model.incremental_train(data_manager)
        cnn_accy = model.eval_task()
        if args.get("save_predictions", False):
            from utils.stats_diagnostics import save_predictions
            save_predictions(args.get("_qbridge_report_dir") or logfilename + "_stats", task,
                             *model._last_eval, model._known_classes)
        model.after_task()
     
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        task_accuracy_matrix.append(cnn_accy.get("task_accuracies", []))
        model.cnn_curve = cnn_curve
        model.nme_curve = nme_curve
        model.task_accuracy_matrix = task_accuracy_matrix

        if len(task_accuracy_matrix) > 1:
            current_row = task_accuracy_matrix[-1]
            old_task_count = len(current_row) - 1
            backward_transfer = np.mean([
                current_row[index] - task_accuracy_matrix[index][index]
                for index in range(old_task_count)
            ])
            forgetting = np.mean([
                max(row[index] for row in task_accuracy_matrix[index:-1])
                - current_row[index]
                for index in range(old_task_count)
            ])
            logging.info(
                "Continual metrics after task %d: BWT=%.4f, forgetting=%.4f, task_acc=%s",
                task, backward_transfer, forgetting, current_row,
            )

        if args.get("save_checkpoints", True) and (not qb_experiment or task == data_manager.nb_tasks - 1):
            checkpoint_path = os.path.join(checkpoint_dir, "task_{}.pkl".format(task))
            model.save_checkpoint(checkpoint_path)
            if not qb_experiment and args.get("keep_last_checkpoint", True) and task > 0:
                previous_checkpoint = os.path.join(
                    checkpoint_dir, "task_{}.pkl".format(task - 1)
                )
                if os.path.isfile(previous_checkpoint):
                    os.remove(previous_checkpoint)
                    logging.info("Removed superseded checkpoint: %s", previous_checkpoint)


        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        if not qb_experiment:
            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
        logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        if qb_experiment:
            from utils.qbridge_experiment import run_manifest
            from utils.qbridge import write_json
            task_report = {"task": task, "accuracy": cnn_accy,
                           "aia": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                           "forgetting": float(forgetting) if len(task_accuracy_matrix) > 1 else None,
                           "bwt": float(backward_transfer) if len(task_accuracy_matrix) > 1 else None,
                           "task_accuracy_matrix": task_accuracy_matrix}
            write_json(os.path.join(args["_qbridge_report_dir"], "task_%d.json" % task), task_report)
            run_manifest(args, "completed" if task == data_manager.nb_tasks - 1 else "partial",
                         last_completed_task=task, cnn_curve=cnn_curve,
                         class_order=model.class_order,
                         task_accuracy_matrix=task_accuracy_matrix,
                         **{"final_aia": task_report["aia"]})
            logging.info("QBRIDGE task=%d AIA=%.3f old=%.2f new=%.2f status=completed",
                         task, task_report["aia"], cnn_accy["grouped"]["old"], cnn_accy["grouped"]["new"])
    if end_task < data_manager.nb_tasks:
        logging.info(
            ("QBRIDGE stopped after %d task(s) at task %d; partial pilot, restart from paired base for a full run."
             if qb_experiment else "Stopped after %d task(s); resume the same seed to continue at task %d."),
            end_task - start_task, end_task,
        )
    return sum(cnn_curve["top1"])/len(cnn_curve["top1"])
       
def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
