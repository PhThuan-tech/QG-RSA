"""Base-only artifact fitting. Validation is evaluation-only, never used for tuning."""
import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from utils.qbridge import MonotoneCalibrator, stable_hash


def stratified_split(labels, seed, fractions=(0.6, 0.2, 0.2)):
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("P/F/V fractions must be positive and sum to one")
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(np.asarray(labels) == label))
        if len(indices) < 3:
            raise ValueError("Every base class requires at least three samples for P/F/V")
        n_p = min(max(1, int(len(indices) * fractions[0])), len(indices)-2)
        n_f = min(max(1, int(len(indices) * fractions[1])), len(indices)-n_p-1)
        for output, subset in zip(parts, (indices[:n_p], indices[n_p:n_p+n_f], indices[n_p+n_f:])):
            output.extend(map(int, subset))
    return {key: values for key, values in zip(("P", "F", "V"), parts)}


def binary_metrics(scores, targets, bins=10):
    p = np.asarray(torch.as_tensor(scores).detach().cpu(), dtype=np.float64).reshape(-1)
    y = np.asarray(torch.as_tensor(targets).detach().cpu(), dtype=np.float64).reshape(-1)
    if not len(p) or len(p) != len(y) or not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid binary diagnostic inputs")
    p = np.clip(p, 0, 1)
    clipped = np.clip(p, 1e-12, 1-1e-12)
    ece = 0.
    bucket = np.minimum((p * bins).astype(int), bins-1)
    for k in range(bins):
        chosen = bucket == k
        if chosen.any():
            ece += chosen.mean() * abs(p[chosen].mean() - y[chosen].mean())
    result = dict(nll=float(-(y*np.log(clipped)+(1-y)*np.log1p(-clipped)).mean()),
                  brier=float(np.square(p-y).mean()), ece=float(ece),
                  positive_rate=float(y.mean()), count=len(y),
                  quantiles=np.quantile(p, [0, .25, .5, .75, 1]).tolist(),
                  auroc=None, auprc=None, spearman=None)
    if 0 < y.sum() < len(y):
        order = np.argsort(-p, kind="stable")
        ordered_p, ordered_y = p[order], y[order]
        ends = np.r_[np.flatnonzero(np.diff(ordered_p)), len(p)-1]
        tp = np.cumsum(ordered_y)[ends]
        fp = ends+1-tp
        tpr = np.r_[0, tp/y.sum()]
        fpr = np.r_[0, fp/(len(y)-y.sum())]
        result["auroc"] = float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2))
        result["auprc"] = float(np.sum(np.diff(tpr)*(tp/(ends+1))))
        def ranks(a):
            _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
            return (np.cumsum(counts) - (counts-1)/2)[inv]
        if np.ptp(p) > 0:
            result["spearman"] = float(np.corrcoef(ranks(p), ranks(y))[0, 1])
    return result


def check_feature_cache(cache, base_hash):
    if cache.get("kind") != "base_feature_cache" or cache.get("source") != "base_train_only":
        raise ValueError("Only an explicitly base-train feature cache is accepted")
    if cache.get("base_checkpoint_hash") != base_hash:
        raise ValueError("Feature cache/base checkpoint mismatch")
    x, y, scores = cache["features"], cache["labels"], cache["scores"]
    if len(x) != len(y) or len(x) != len(scores) or not torch.isfinite(x).all() or not torch.isfinite(scores).all():
        raise ValueError("Invalid base feature cache")
    if len(cache["sample_ids"]) != len(x) or len(set(cache["sample_ids"])) != len(x):
        raise ValueError("Sample IDs must be unique and aligned")
    if sorted(y.unique().tolist()) != list(range(scores.shape[1])):
        raise ValueError("Cache labels must map to all base-head rows")


def fit_calibrator(cache, metric, base_hash, seed=1993, proxy_margin=0.1,
                   fractions=(.6, .2, .2), steps=500, lr=0.03, device="cpu"):
    check_feature_cache(cache, base_hash)
    split = stratified_split(cache["labels"].numpy(), seed, fractions)
    x, labels, scores = cache["features"], cache["labels"], cache["scores"]
    fit_prototypes = torch.stack([x[split["P"]][labels[split["P"]] == c].mean(0)
                                  for c in range(scores.shape[1])]).to(device)
    metric.set_inc_mode("frozen")
    metric.eval()
    pairs = {}
    with torch.no_grad():
        proto_repr = metric.encode(fit_prototypes)
        for part in ("F", "V"):
            selected = split[part]
            risks = torch.cat([metric.kernel(metric.encode(x[batch].to(device)), proto_repr).cpu()
                               for batch in [selected[i:i+128] for i in range(0, len(selected), 128)]])
            logits = scores[selected]
            correct = labels[selected]
            mask = torch.arange(scores.shape[1])[None, :] != correct[:, None]
            targets = (proxy_margin + logits - logits.gather(1, correct[:, None]) > 0).double()
            pairs[part] = risks[mask].double(), targets[mask]
    artifact = {"kind": "calibrator", "source": "base_train_only", "base_checkpoint_hash": base_hash,
                "split_seed": seed, "proxy_margin": proxy_margin, "fractions": list(fractions),
                "split_ids": {part: [cache["sample_ids"][i] for i in indices] for part, indices in split.items()},
                "representation_scope": "in-sample encoder, calibrator-held-out",
                "fit_protocol": {"loss": "unweighted_bce", "steps": steps, "lr": lr,
                                 "optimizer": "Adam", "evaluation_used_for_selection": False},
                "gate_passed": False, "status": "gate_failed"}
    artifact["fit_id"] = stable_hash({k: v for k, v in artifact.items() if k not in ("status", "gate_passed")})
    if any(y.unique().numel() != 2 for _, y in pairs.values()):
        artifact["failure"] = "Both target classes are required in F and V; R6 unavailable"
        return artifact
    cal = MonotoneCalibrator()
    optimizer = torch.optim.Adam(cal.parameters(), lr=lr)
    p, y = pairs["F"]
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(cal.logits(p), y)
        if not torch.isfinite(loss):
            artifact["failure"] = "Nonfinite calibration loss"
            return artifact
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        raw, target = pairs["V"]
        calibrated = cal(raw)
        artifact["parameters"] = {"u": float(cal.u), "b": float(cal.b)}
        artifact["evaluation"] = {
            "raw": binary_metrics(raw, target), "calibrated": binary_metrics(calibrated, target),
            "constant_fit_prevalence": binary_metrics(torch.full_like(raw, y.mean()), target)}
        invariant = all(artifact["evaluation"]["raw"][name] == artifact["evaluation"]["calibrated"][name]
                        for name in ("auroc", "auprc", "spearman"))
        artifact["ranking_invariant"] = invariant
        finite = all(math.isfinite(v) for v in artifact["parameters"].values())
        noncollapsed = float(calibrated.max()-calibrated.min()) > 1e-10
        artifact["gate_passed"] = finite and noncollapsed and invariant
        artifact["status"] = "fitted" if artifact["gate_passed"] else "gate_failed"
        if not artifact["gate_passed"]:
            artifact["failure"] = "Nonfinite parameters, collapsed scores or finite-precision ranking loss"
    return artifact


def metric_config_from_args(args, kind="pqk"):
    return dict(input_dim=768, num_qubits=int(args.get("q_num_qubits", 8)),
                num_layers=int(args.get("q_num_layers", 2)), kernel_type=kind,
                kernel_order=int(args.get("q_kernel_order", 1)),
                order2_weight=float(args.get("q_order2_weight", 1)),
                reupload=bool(args.get("q_reupload", False)),
                gamma_mode=args.get("q_gamma_mode", "bounded_learned"),
                dtype=args.get("q_dtype", "float32"), init_seed=int(args.get("q_init_seed", 1234)))


def fit_classical_selector(cache, pqk, kernel_config, base_hash, args, seed=1993,
                           epochs=10, batch_size=64, lr=0.01, device="cpu"):
    from utils.qbridge import kernel_from_config
    check_feature_cache(cache, base_hash)
    selector = kernel_from_config(kernel_config).to(device)
    selector.projector.load_state_dict(pqk.projector.state_dict())
    generator = torch.Generator().manual_seed(seed)
    features, labels = cache["features"], cache["labels"]
    indices = torch.randperm(len(features), generator=generator)[:min(512, len(features))]
    selector.calibrate_gamma(features[indices].to(device))
    kind = "rbf_proj" if kernel_config["kernel_type"] == "rbf_proj" else "mlp"
    start = time.perf_counter()
    losses = []
    if kind == "mlp":
        # Same label-aware positive/hinge-negative PQK steering, on fixed cached
        # features only. No joint encoder/CosFace training: explicitly conditional.
        from models.RSIAT_adapter import RS_Loss
        objective = RS_Loss(args["alpha"], args["rs_margin"])
        optimizer = torch.optim.SGD(selector.param_groups(lr, args.get("weight_decay", 0),
                                      args.get("q_metric_lr_mult", .1)), momentum=.9)
        for _ in range(epochs):
            order = torch.randperm(len(features), generator=generator)
            epoch_loss = []
            for offset in range(0, len(order), batch_size):
                batch = order[offset:offset+batch_size]
                loss = objective(features[batch].to(device), labels[batch].to(device),
                                 quantum_kernel_module=selector,
                                 margin_override=args.get("rs_margin_q", .5),
                                 margin_mode=args.get("rs_margin_q_mode", "fixed"),
                                 margin_quantile=args.get("rs_margin_quantile", .9))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss.append(float(loss.detach()))
            losses.append(sum(epoch_loss)/len(epoch_loss))
    selector.set_inc_mode("frozen")
    manifest = {"kind": kind, "source": "base_train_only", "base_checkpoint_hash": base_hash,
                "status": "fitted", "kernel_config": kernel_config,
                "protocol": {"seed": seed, "epochs": epochs if kind == "mlp" else 0,
                             "batch_size": batch_size, "lr": lr, "optimizer": "SGD(momentum=0.9)",
                             "objective": "PQK positive plus alpha hinge-negative steering; fixed base features",
                             "objective_args": {k: args.get(k) for k in ("alpha", "rs_margin_q", "rs_margin_q_mode",
                                                                      "rs_margin_quantile", "weight_decay", "q_metric_lr_mult")},
                             "gamma_fit": "inverse median off-diagonal squared distances, deterministic base-train subset",
                             "gamma_fit_sample_ids": [cache["sample_ids"][i] for i in indices.tolist()],
                             "projection_initialization": "exact learned base PQK projector",
                             "quantum_advantage_claim_allowed": False,
                             "scope": "conditional selector control; not matched joint-from-scratch base training"},
                "training_seconds": time.perf_counter()-start, "training_losses": losses,
                "parameter_counts": selector.parameter_counts()}
    manifest["fit_id"] = stable_hash({"base": base_hash, "protocol": manifest["protocol"], "config": kernel_config})
    manifest["state_dict"] = {k: v.cpu() for k, v in selector.state_dict().items()}
    return manifest
