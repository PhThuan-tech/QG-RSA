import copy
import csv
import glob
import hashlib
import json
import logging
import os
import random
import re
import sys
import time

import numpy as np
import torch

from data.data_manager import DataManager
from utils import model_factory
from utils.toolkit import count_parameters


def _write_json_atomic(path, payload):
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_atomic(path, fieldnames, rows):
    if not path:
        return
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_arm_outputs(args, payload):
    records = payload["task_records"]
    task_rows = []
    class_rows = []
    for record in records:
        for stage in ("pre_ca", "post_ca"):
            evaluation = record.get(stage)
            if evaluation is None:
                continue
            metrics = evaluation["metrics"]
            task_rows.append({
                "arm": record["arm"],
                "mode": record["mode"],
                "task": record["task"],
                "stage": stage,
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "old_top1": metrics["grouped"]["old"],
                "new_top1": metrics["grouped"]["new"],
                "ca_gain_top1": record.get("ca_gain_top1"),
                "runtime_seconds": record.get("runtime_seconds", 0.0),
            })
            for class_id, accuracy in evaluation["per_class_top1"].items():
                class_rows.append({
                    "arm": record["arm"],
                    "mode": record["mode"],
                    "task": record["task"],
                    "stage": stage,
                    "class_id": class_id,
                    "top1": accuracy,
                })
    _write_csv_atomic(
        args.get("task_metrics_output"),
        ["arm", "mode", "task", "stage", "top1", "top5",
         "old_top1", "new_top1", "ca_gain_top1", "runtime_seconds"],
        task_rows,
    )
    _write_csv_atomic(
        args.get("per_class_output"),
        ["arm", "mode", "task", "stage", "class_id", "top1"],
        class_rows,
    )

    transport_rows = []
    for record in records:
        transport = record.get("transport")
        if not transport or record["task"] == 0:
            continue
        before = transport.get("paired_before_stage1") or {}
        after1 = transport.get("paired_after_stage1") or {}
        after2 = transport.get("paired_after_stage2") or {}
        A = transport.get("A_final") or {}
        D = transport.get("D_final") or {}
        stage1 = transport.get("stage1_epochs") or []
        ca_stabilization = transport.get("ca_covariance_stabilization") or {}
        ca_classes = ca_stabilization.get("classes") or []
        transport_rows.append({
            "arm": record["arm"], "mode": record["mode"],
            "task": record["task"],
            "A_initial_sha256": (transport.get("initial_hashes", {}).get("A") or {}).get("sha256"),
            "D_initial_sha256": (transport.get("initial_hashes", {}).get("D") or {}).get("sha256"),
            "A_mse_before_stage1": before.get("a_mse"),
            "A_mse_after_stage1": after1.get("a_mse"),
            "A_mse_after_stage2": after2.get("a_mse"),
            "D_mse_before_stage1": before.get("d_mse"),
            "D_mse_after_stage1": after1.get("d_mse"),
            "cycle_new_after_stage1": after1.get("cycle_new_mse"),
            "cycle_old_after_stage1": after1.get("cycle_old_mse"),
            "A_weight_fro": A.get("weight_frobenius_norm"),
            "A_bias_l2": A.get("bias_l2_norm"),
            "A_singular_min": A.get("singular_min"),
            "A_singular_median": A.get("singular_median"),
            "A_singular_max": A.get("singular_max"),
            "A_condition_number": A.get("condition_number"),
            "D_weight_fro": D.get("weight_frobenius_norm"),
            "ca_pd_fallback_count": ca_stabilization.get("fallback_count", 0),
            "ca_pd_max_relative_jitter_used": max(
                (item["relative_jitter"] for item in ca_classes), default=0.0),
            "final_epoch_L_align": stage1[-1]["align"] if stage1 else None,
            "final_epoch_L_orth": stage1[-1]["orth"] if stage1 else None,
            "final_epoch_L_cls": stage1[-1]["classification"] if stage1 else None,
            "final_epoch_L_A": stage1[-1]["loss_a"] if stage1 else None,
            "final_epoch_L_D": stage1[-1]["loss_d"] if stage1 else None,
            "final_epoch_cycle_new": stage1[-1]["cycle_new"] if stage1 else None,
            "final_epoch_cycle_old": stage1[-1]["cycle_old"] if stage1 else None,
        })
    _write_csv_atomic(
        args.get("transport_metrics_output"),
        [
            "arm", "mode", "task", "A_initial_sha256", "D_initial_sha256",
            "A_mse_before_stage1", "A_mse_after_stage1", "A_mse_after_stage2",
            "D_mse_before_stage1", "D_mse_after_stage1",
            "cycle_new_after_stage1", "cycle_old_after_stage1",
            "A_weight_fro", "A_bias_l2", "A_singular_min",
            "A_singular_median", "A_singular_max", "A_condition_number",
            "D_weight_fro", "ca_pd_fallback_count",
            "ca_pd_max_relative_jitter_used", "final_epoch_L_align", "final_epoch_L_orth",
            "final_epoch_L_cls", "final_epoch_L_A", "final_epoch_L_D",
            "final_epoch_cycle_new", "final_epoch_cycle_old",
        ],
        transport_rows,
    )
    _write_json_atomic(args.get("metrics_output"), payload)


def RSIAT_train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    sum_seed = 0.0
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        sum_seed += _train(args)
    avg_seed = sum_seed / len(seed_list)
    print("Average Seed Accuracy (CNN):", avg_seed)
    logging.info("Average Seed Accuracy (CNN): %s", avg_seed)


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "task_*.pkl"))
    if not checkpoint_files:
        return None

    def task_number(filepath):
        match = re.search(r"task_(\d+)\.pkl$", os.path.basename(filepath))
        return int(match.group(1)) if match else -1
    return max(checkpoint_files, key=task_number)


def _common_task0_record(args):
    report_path = args.get("common_task0_report")
    if not report_path:
        return None
    with open(report_path) as handle:
        report = json.load(handle)
    if report.get("status") != "PASS":
        raise RuntimeError("Common task0 compatibility report did not pass")
    evaluated = report["task0_metrics_reproduced"]
    return {
        "arm": args["arm"],
        "mode": args["bicyc_mode"],
        "task": 0,
        "pre_ca": copy.deepcopy(evaluated),
        "post_ca": copy.deepcopy(evaluated),
        "ca_gain_top1": 0.0,
        "transport": None,
        "runtime_seconds": 0.0,
        "checkpoint": copy.deepcopy(report["destination"]),
        "common_task0": True,
    }


def _train(args):
    run_started = time.time()
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    output_root = args.get("output_root", "")
    logs_name = os.path.join(
        output_root, "logs", args["model_name"], args["dataset"],
        str(init_cls), str(args["increment"]))
    os.makedirs(logs_name, exist_ok=True)
    logfilename = os.path.join(
        logs_name,
        "{}_{}_{}".format(args["prefix"], args["seed"], args["convnet_type"]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        force=True,
        handlers=[logging.FileHandler(logfilename + ".log"),
                  logging.StreamHandler(sys.stdout)],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"])
    model = model_factory.get_model(args["model_name"], args)
    model.class_order = list(data_manager._class_order)

    checkpoint_dir = os.path.join(
        output_root, "ckpt", str(args["prefix"]), str(args["dataset"]),
        "{}_{}".format(args["init_cls"], args["increment"]))
    if args.get("isolate_runs", True):
        checkpoint_dir = os.path.join(
            checkpoint_dir, "seed_{}".format(args["seed"]))
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_task = 0
    if args.get("resume", False):
        resume_path = args.get("resume_path") or find_latest_checkpoint(checkpoint_dir)
        if resume_path and os.path.isfile(resume_path):
            completed_task = model.load_checkpoint(resume_path)
            start_task = completed_task + 1
            logging.info("Resuming experiment from task %d", start_task)
        else:
            raise RuntimeError("Resume requested but no checkpoint was found")

    cnn_curve = getattr(model, "cnn_curve", {"top1": [], "top5": []})
    nme_curve = getattr(model, "nme_curve", {"top1": [], "top5": []})
    if not getattr(model, "experiment_records", None):
        common = _common_task0_record(args)
        model.experiment_records = [] if common is None else [common]

    if start_task >= data_manager.nb_tasks:
        return sum(cnn_curve["top1"]) / len(cnn_curve["top1"])

    end_task = data_manager.nb_tasks
    max_tasks_per_run = args.get("max_tasks_per_run")
    if max_tasks_per_run is not None:
        end_task = min(end_task, start_task + max(1, int(max_tasks_per_run)))
    stop_after_task = args.get("stop_after_task")
    if stop_after_task is not None:
        end_task = min(end_task, int(stop_after_task) + 1)
    logging.info("This run will process tasks [%d, %d).", start_task, end_task)

    checkpoints = []
    for task in range(start_task, end_task):
        logging.info("All params: %s", count_parameters(model._network))
        logging.info("Trainable params: %s", count_parameters(model._network, True))
        task_started = time.time()
        model.incremental_train(data_manager)
        post_ca = model.eval_task_detailed()
        pre_ca = copy.deepcopy(model.pre_ca_metrics[str(model._cur_task)])
        transport = copy.deepcopy(model._current_transport_record)
        record = {
            "arm": args["arm"],
            "mode": args["bicyc_mode"],
            "task": int(model._cur_task),
            "pre_ca": pre_ca,
            "post_ca": copy.deepcopy(post_ca),
            "ca_gain_top1": (
                post_ca["metrics"]["top1"] - pre_ca["metrics"]["top1"]),
            "transport": transport,
            "runtime_seconds": time.time() - task_started,
            "checkpoint": None,
            "common_task0": False,
        }
        model.experiment_records.append(record)
        model.after_task()
        cnn_accy = post_ca["metrics"]
        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        model.cnn_curve = cnn_curve
        model.nme_curve = nme_curve

        if args.get("save_checkpoints", True):
            checkpoint_path = os.path.join(
                checkpoint_dir, "task_{}.pkl".format(task))
            model.save_checkpoint(checkpoint_path)
            checkpoint_info = {
                "task": task,
                "path": os.path.abspath(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
            }
            checkpoints.append(checkpoint_info)
            record["checkpoint"] = checkpoint_info
            if args.get("keep_last_checkpoint", True) and task > 0:
                previous = os.path.join(
                    checkpoint_dir, "task_{}.pkl".format(task - 1))
                if os.path.isfile(previous):
                    os.remove(previous)

        payload = {
            "status": "RUNNING" if task + 1 < end_task else "PASS",
            "implementation_label": (
                "[IMPLEMENTATION ADAPTATION] RSIAT + BiCyc-style "
                "bidirectional/cycle transport"
            ),
            "arm": args["arm"],
            "mode": args["bicyc_mode"],
            "seed": args["seed"],
            "batch_size": args["batch_size"],
            "start_task": start_task,
            "end_task_exclusive": end_task,
            "task_records": model.experiment_records,
            "cnn_curve": cnn_curve,
            "checkpoint_dir": checkpoint_dir,
            "checkpoints": checkpoints,
            "runtime_seconds": time.time() - run_started,
        }
        _write_arm_outputs(args, payload)
        logging.info("TRACK_B_TASK %s", json.dumps(record, sort_keys=True))
        logging.info("CNN top1 curve: %s", cnn_curve["top1"])
        logging.info("CNN top5 curve: %s", cnn_curve["top5"])

    if end_task < data_manager.nb_tasks:
        logging.info("Stopped at task %d by max_tasks_per_run.", end_task - 1)
    return sum(cnn_curve["top1"]) / len(cnn_curve["top1"])


def _set_device(args):
    gpus = []
    for device in args["device"]:
        gpus.append(torch.device("cpu") if device == -1
                    else torch.device("cuda:{}".format(device)))
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
        logging.info("%s: %s", key, value)
