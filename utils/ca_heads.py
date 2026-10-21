"""Alternative classifier-alignment heads, evaluated side by side (QRF-CA).

RSIAT's classifier alignment (CA) refits a linear cosine head on Gaussian
pseudo-features N(s_c * mu_c, Sigma_c). The heads here are fitted on the same
class-conditional model and differ only in the classifier:

  linear           the trained RSIAT head (its predictions are passed in)
  qda              per-class Mahalanobis with shrunk covariances, no training
  rff_ridge        RBF random Fourier features + closed-form ridge
  rp_ridge         ReLU random projection + closed-form ridge
  pqk_ridge        projected-quantum-kernel random features + closed-form ridge
  pqk_noent_ridge  the same circuits without CNOTs (role of entanglement)

Evaluation only: heads never feed back into training. In RSIAT the next task
trains the adapter on new-class logits only, and class statistics do not depend
on the head, so all heads can be compared inside one run. Every random draw uses
private generators inside ``preserve_rng``; the trained model and the training
random stream are unchanged.
"""
import json
import logging
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

HEADS = ("linear", "qda", "rff_ridge", "rp_ridge", "pqk_ridge", "pqk_noent_ridge")
RIDGE_HEADS = ("rff_ridge", "rp_ridge", "pqk_ridge", "pqk_noent_ridge")
# Pre-registered settings (exps/qrfca/README.md). Same M for every feature map.
DEFAULTS = {
    "ca_head_samples": 512,       # pseudo-features per class
    "ca_head_dim": 8192,          # feature-map dimension M
    "ca_head_qubits": 8,
    "ca_head_layers": 2,
    "ca_head_reupload": True,
    "ca_head_order": 2,           # 1: one-qubit RDMs; 2: adds neighbouring two-qubit RDMs
    "ca_head_seed": 20261021,
    "ca_head_holdout": 0.2,       # pseudo-features held out to choose the ridge strength
    "ca_head_gate_samples": 1000, # pseudo-features for kernel alignment / geometric difference
}
RIDGE_GRID = tuple(10.0 ** k for k in range(-3, 4))  # times trace(G) / M
MAP_SEEDS = {"rff": 0, "rp": 1, "pqk": 2, "pqk_noent": 2}
QDA_SHRINK = (1.0, 1.0)  # diagonal and off-diagonal shrinkage weights (FeCAM-style)
CHUNK = 2048


def head_settings(args):
    """Validated QRF-CA settings, or None when no extra head is requested."""
    heads = args.get("ca_heads")
    if not heads:
        return None
    if isinstance(heads, str) or not all(isinstance(h, str) for h in heads):
        raise ValueError("ca_heads must be a list of head names.")
    unknown = sorted(set(heads) - set(HEADS))
    if unknown:
        raise ValueError("Unknown ca_heads {}; choose from {}.".format(unknown, list(HEADS)))
    cfg = {key: args.get(key, default) for key, default in DEFAULTS.items()}
    for key in ("ca_head_samples", "ca_head_dim", "ca_head_qubits", "ca_head_layers",
                "ca_head_order", "ca_head_seed", "ca_head_gate_samples"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int):
            raise ValueError("{} must be an integer.".format(key))
    if cfg["ca_head_samples"] < 10 or cfg["ca_head_dim"] < 2 or cfg["ca_head_gate_samples"] < 0:
        raise ValueError("ca_head_samples >= 10, ca_head_dim >= 2 and ca_head_gate_samples >= 0 are required.")
    if not 2 <= cfg["ca_head_qubits"] <= 12 or not 1 <= cfg["ca_head_layers"] <= 3:
        raise ValueError("ca_head_qubits must be in [2, 12] and ca_head_layers in [1, 3].")
    if cfg["ca_head_order"] not in (1, 2):
        raise ValueError("ca_head_order must be 1 or 2.")
    if not 0.0 < float(cfg["ca_head_holdout"]) < 0.5:
        raise ValueError("ca_head_holdout must be in (0, 0.5).")
    cfg["ca_head_reupload"] = bool(cfg["ca_head_reupload"])
    cfg["heads"] = ["linear"] + [h for h in HEADS if h in heads and h != "linear"]
    return cfg


# ----------------------------------------------------------------------------- pseudo-features
def ca_mean_scale(task_sizes, cur_task):
    """CA's per-class mean scaling 0.9 + 0.1 * (t_c + 1) / (cur_task + 1) (models/base.py)."""
    return np.concatenate([np.full(size, 0.9 + 0.1 * (t + 1) / (cur_task + 1))
                           for t, size in enumerate(task_sizes)])


def sample_pseudo(means, covs, per_class, generator, device):
    """Draw ``per_class`` samples from every N(mean_c, cov_c) with a private generator."""
    xs = []
    for mean, cov in zip(means, covs):
        chol = torch.linalg.cholesky(cov.to(device=device, dtype=torch.float64))
        eps = torch.randn(per_class, mean.shape[0], generator=generator, device=device, dtype=torch.float64)
        xs.append((mean.to(device=device, dtype=torch.float64) + eps @ chol.T).float())
    labels = torch.arange(len(means), device=device).repeat_interleave(per_class)
    return torch.cat(xs), labels


def holdout_mask(num_classes, per_class, fraction, device):
    """Last ``fraction`` of every class's samples (samples are i.i.d., so this is a random split)."""
    keep = per_class - max(1, int(round(per_class * fraction)))
    return (torch.arange(per_class, device=device) >= keep).repeat(num_classes)


# ----------------------------------------------------------------------------- feature maps
def _median_distance(x, generator, limit=2000):
    idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:limit]
    d = torch.cdist(x[idx], x[idx])
    return d[torch.triu(torch.ones_like(d, dtype=torch.bool), diagonal=1)].median().clamp_min(1e-6)


class FeatureMap:
    """Frozen random feature map. Weights come from the seed; scales from the current pseudo-data."""

    def __init__(self, kind, input_dim, dim, seed, device, qubits=8, layers=2, reupload=True, order=2):
        self.kind, self.dim, self.device = kind, dim, device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        if kind in ("rff", "rp"):
            self.weight = torch.randn(input_dim, dim, generator=gen).to(device)
            if kind == "rff":
                self.phase = (2 * math.pi * torch.rand(dim, generator=gen)).to(device)
        elif kind in ("pqk", "pqk_noent"):
            from utils.quantum_kernel import LocalRDMExtractor, QuantumFeatureEncoder
            self.qubits, self.layers, self.order = qubits, layers, order
            self.per_circuit = 2 * qubits + (10 * qubits if order == 2 else 0)
            self.circuits = math.ceil(dim / self.per_circuit)
            self.weight = torch.randn(input_dim, self.circuits * qubits, generator=gen).to(device)
            # Frozen random variational angles, one set per circuit.
            self.angles = (2 * math.pi * torch.rand(self.circuits, layers, qubits, generator=gen)).to(device)
            self.encoder = QuantumFeatureEncoder(qubits, layers, torch.float32, entangle=kind == "pqk",
                                                 reupload=reupload, frozen_random=True).to(device)
            self.extractor = LocalRDMExtractor(qubits, order)
        else:
            raise ValueError("Unknown feature map " + kind)
        self.scale = None

    def fit_scale(self, x, generator):
        x = F.layer_norm(x, x.shape[-1:])
        if self.kind == "rff":
            self.scale = _median_distance(x, generator)       # RBF bandwidth: median heuristic
        elif self.kind.startswith("pqk"):
            idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:4096]
            self.scale = (x[idx] @ self.weight).std(0).clamp_min(1e-6)  # per (circuit, qubit) projection
        return self

    def __call__(self, x):
        x = F.layer_norm(x, x.shape[-1:])
        if self.kind == "rp":
            return F.relu(x @ self.weight)
        if self.kind == "rff":
            return math.sqrt(2.0 / self.dim) * torch.cos(x @ self.weight / self.scale + self.phase)
        theta = math.pi * torch.tanh(x @ self.weight / self.scale)
        return pqk_features(theta.reshape(-1, self.circuits, self.qubits), self.angles, self.encoder,
                            self.extractor, self.order)[:, :self.dim]


def pqk_features(theta, var_angles, encoder, extractor, order):
    """Bloch features of local RDMs for a batch of circuits.

    theta: (B, P, q) data angles; var_angles: (P, L, q). Real amplitudes, so a
    one-qubit RDM has two free entries (<X>, <Z>) and a two-qubit RDM ten.
    """
    batch, circuits, qubits = theta.shape
    flat = theta.reshape(batch * circuits, qubits)
    state = encoder._zero_state(flat.shape[0], flat.device, flat.dtype)
    if not encoder.reupload:
        state = encoder._apply_encoding(state, flat)
    for layer in range(var_angles.shape[1]):
        if encoder.reupload:
            state = encoder._apply_encoding(state, flat)
        layer_angles = var_angles[:, layer, :].repeat(batch, 1)  # row b*P + p -> circuit p
        for q in range(qubits):
            state = encoder._apply_ry(state, layer_angles[:, q], q)
        state = encoder._apply_entangling(state)
    rdm = extractor(state)
    one = rdm["one"]
    feats = [one[..., 0, 0] - one[..., 1, 1], 2 * one[..., 0, 1]]  # <Z>, <X>
    feats = torch.stack(feats, dim=-1).reshape(flat.shape[0], -1)
    if order == 2:
        iu = torch.triu_indices(4, 4, device=theta.device)
        two = rdm["two"][..., iu[0], iu[1]].reshape(flat.shape[0], -1)
        feats = torch.cat([feats, two], dim=1)
    return feats.reshape(batch, -1)


def apply_map(fmap, x):
    return torch.cat([fmap(x[i:i + CHUNK]) for i in range(0, x.shape[0], CHUNK)])


# ----------------------------------------------------------------------------- heads
def fit_ridge(fmap, x, y, num_classes, hold):
    """Closed-form ridge W = (G + lam I)^-1 Phi^T Y; lam chosen on held-out pseudo-features."""
    dim = fmap.dim
    gram = torch.zeros(dim, dim, dtype=torch.float64, device=x.device)
    proj = torch.zeros(dim, num_classes, dtype=torch.float64, device=x.device)
    train_idx = torch.nonzero(~hold).squeeze(1)
    for i in range(0, train_idx.numel(), CHUNK):
        idx = train_idx[i:i + CHUNK]
        phi = fmap(x[idx])
        gram += (phi.T @ phi).double()
        proj += (phi.T @ F.one_hot(y[idx], num_classes).float()).double()
    phi_hold, y_hold = apply_map(fmap, x[hold]), y[hold]
    base = float(torch.trace(gram)) / dim
    eye = torch.eye(dim, dtype=torch.float64, device=x.device)
    scores = []
    for factor in RIDGE_GRID:
        weight = torch.cholesky_solve(proj, torch.linalg.cholesky(gram + factor * base * eye)).float()
        scores.append(float(((phi_hold @ weight).argmax(1) == y_hold).float().mean()))
    best = max(range(len(RIDGE_GRID)), key=lambda k: (scores[k], k))  # ties -> stronger ridge
    gram += (phi_hold.T @ phi_hold).double()
    proj += (phi_hold.T @ F.one_hot(y_hold, num_classes).float()).double()
    lam = RIDGE_GRID[best] * float(torch.trace(gram)) / dim
    weight = torch.cholesky_solve(proj, torch.linalg.cholesky(gram + lam * eye)).float()
    return weight, {"ridge_factor": RIDGE_GRID[best], "holdout_accuracy": scores}


def shrink_cov(cov, gamma1=QDA_SHRINK[0], gamma2=QDA_SHRINK[1]):
    d = cov.shape[-1]
    eye = torch.eye(d, dtype=cov.dtype, device=cov.device)
    v1 = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1)[..., None, None]
    v2 = (cov.sum((-2, -1)) - torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1))[..., None, None] / (d * (d - 1))
    return cov + gamma1 * v1 * eye + gamma2 * v2 * (1 - eye)


def qda_predict(means, covs, z):
    """argmax_c  -1/2 (z-mu_c)^T S_c^-1 (z-mu_c) - 1/2 log det S_c with shrunk S_c."""
    z = z.double()
    scores = []
    for mean, cov in zip(means, covs):
        chol = torch.linalg.cholesky(shrink_cov(cov.to(z.device, torch.float64)))
        sol = torch.linalg.solve_triangular(chol, (z - mean.to(z.device, torch.float64)).T, upper=False)
        scores.append(-0.5 * sol.square().sum(0) - torch.log(torch.diagonal(chol)).sum())
    return torch.stack(scores, 1).argmax(1)


# ----------------------------------------------------------------------------- gates
def _centered_kernel(phi):
    k = (phi @ phi.T).double()
    k = k * (k.shape[0] / torch.trace(k).clamp_min(1e-12))  # tr(K) = n (Huang et al. 2021)
    n = k.shape[0]
    h = torch.eye(n, dtype=k.dtype, device=k.device) - 1.0 / n
    return k, h @ k @ h


def kernel_gates(maps, x, y, num_classes):
    """Centered kernel-target alignment per map and geometric differences g(K_c || K_pqk)."""
    out, kernels = {"alignment": {}, "geometric_difference": {}}, {}
    target = F.one_hot(y, num_classes).double()
    yy = target @ target.T
    for name, fmap in maps.items():
        k, kc = _centered_kernel(apply_map(fmap, x))
        kernels[name] = k
        out["alignment"][name] = float((kc * yy).sum() / (kc.norm() * yy.norm()).clamp_min(1e-12))
    if "pqk" in kernels:
        n = x.shape[0]
        evals, evecs = torch.linalg.eigh(kernels["pqk"])
        sqrt_q = evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.T
        for name, kc in kernels.items():
            if name == "pqk":
                continue
            # tr(K) = n, so the mean eigenvalue is 1 and 1e-3 is a small relative ridge.
            inv = torch.linalg.inv(kc + 1e-3 * torch.eye(n, dtype=kc.dtype, device=kc.device))
            g = torch.linalg.matrix_norm(sqrt_q @ inv @ sqrt_q, ord=2)
            out["geometric_difference"][name] = float(g.sqrt())
    return out


# ----------------------------------------------------------------------------- driver
def _accuracy(pred, target, task_sizes, known):
    pred, target = np.asarray(pred), np.asarray(target)
    bounds = np.cumsum([0] + list(task_sizes))
    per_task = [float(np.around((pred[(target >= lo) & (target < hi)] == target[(target >= lo) & (target < hi)]).mean() * 100, 2))
                for lo, hi in zip(bounds[:-1], bounds[1:])]
    old, new = target < known, target >= known
    # Same expression as utils.toolkit.accuracy, so the linear head reproduces the CNN curve exactly.
    return {"top1": float(np.around((pred == target).sum() * 100 / len(target), decimals=2)),
            "task_accuracies": per_task,
            "old": float(np.around((pred[old] == target[old]).mean() * 100, 2)) if old.any() else None,
            "new": float(np.around((pred[new] == target[new]).mean() * 100, 2)) if new.any() else None}


def evaluate_heads(learner, task, linear_pred, report_dir):
    """Fit and evaluate every configured head for the current task; log curves."""
    from utils.qbridge_experiment import preserve_rng
    cfg = head_settings(learner.args)
    if cfg is None:
        return None
    start = time.perf_counter()
    with preserve_rng():
        modes = [(module, module.training) for module in learner._network.modules()]
        with torch.no_grad():
            result = _evaluate(learner, cfg, task, linear_pred)
        for module, training in modes:
            module.training = training
    result["seconds"] = time.perf_counter() - start
    curves = getattr(learner, "_ca_head_curves", {})
    for head, metrics in result["heads"].items():
        curves.setdefault(head, []).append(metrics["top1"])
        logging.info("CAHEAD task=%d head=%s top1=%.2f old=%s new=%s", task, head, metrics["top1"],
                     metrics["old"], metrics["new"])
        logging.info("CAHEAD tasks head=%s: %s", head, json.dumps(metrics["task_accuracies"]))
        logging.info("CAHEAD curve head=%s: %s", head, json.dumps(curves[head]))
    learner._ca_head_curves = curves
    logging.info("CAHEAD task=%d gates=%s seconds=%.1f", task, json.dumps(result.get("gates")), result["seconds"])
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "cahead_task_%d.json" % task), "w", encoding="utf-8") as stream:
        json.dump({k: v for k, v in result.items() if k not in ("predictions", "target")}, stream, indent=2)
    np.savez_compressed(os.path.join(report_dir, "cahead_predictions_task_%d.npz" % task),
                        target=result["target"], **{h: p for h, p in result["predictions"].items()})
    return result


def _evaluate(learner, cfg, task, linear_pred):
    device = learner._device
    total, known = learner._total_classes, learner._known_classes
    task_sizes = list(learner.task_sizes)
    z_test, target = learner._extract_vectors(learner.test_loader)
    z_test = torch.as_tensor(z_test, dtype=torch.float32, device=device)
    if linear_pred is not None and len(linear_pred) != len(target):
        raise ValueError("linear predictions and test features differ in length.")

    scale = torch.as_tensor(ca_mean_scale(task_sizes, task), dtype=torch.float64)
    means = torch.as_tensor(np.asarray(learner._class_means[:total]), dtype=torch.float64) * scale[:, None]
    covs = torch.as_tensor(learner._class_covs[:total]).double()

    gen = torch.Generator(device=device).manual_seed(cfg["ca_head_seed"] + 1000 * task)
    per_class = cfg["ca_head_samples"]
    x, y = sample_pseudo(means, covs, per_class, gen, device)
    hold = holdout_mask(total, per_class, cfg["ca_head_holdout"], device)

    predictions, details, maps = {}, {}, {}
    if linear_pred is not None:
        predictions["linear"] = np.asarray(linear_pred)
    if "qda" in cfg["heads"]:
        predictions["qda"] = qda_predict(means, covs, z_test).cpu().numpy()
    for head in RIDGE_HEADS:
        if head not in cfg["heads"]:
            continue
        kind = head[:-len("_ridge")]
        # pqk and pqk_noent share projections and angles, so they differ only in the CNOTs.
        fmap = FeatureMap(kind, x.shape[1], cfg["ca_head_dim"], cfg["ca_head_seed"] + MAP_SEEDS[kind],
                          device, cfg["ca_head_qubits"], cfg["ca_head_layers"], cfg["ca_head_reupload"],
                          cfg["ca_head_order"]).fit_scale(x, gen)
        weight, details[head] = fit_ridge(fmap, x, y, total, hold)
        predictions[head] = torch.cat([(fmap(z_test[i:i + CHUNK]) @ weight).argmax(1)
                                       for i in range(0, z_test.shape[0], CHUNK)]).cpu().numpy()
        maps[kind] = fmap

    gates = None
    if cfg["ca_head_gate_samples"] and maps:
        idx = torch.randperm(x.shape[0], generator=gen, device=device)[:cfg["ca_head_gate_samples"]]
        gates = kernel_gates(maps, x[idx], y[idx], total)
    heads = {h: _accuracy(p, target, task_sizes, known) for h, p in predictions.items()}
    return {"task": task, "settings": {k: v for k, v in cfg.items()}, "heads": heads,
            "ridge": details, "gates": gates, "predictions": predictions, "target": np.asarray(target)}
