"""QSD-CIL stage A: class states, drift channels and pretty-good measurements.

Exemplar-free class-incremental classification as quantum state discrimination
(exps/qsd/README.md):

  state     rho_c = mean_i psi(z_i) psi(z_i)^T over the real features of class c,
            computed when the class is learned (stored as a low-rank factor)
  channel   rho -> A rho A^T / tr(.), A fitted in closed form on the current
            task's images seen by the previous and the current network
  measure   pretty-good measurement E_c = pi_c (rho_bar + dI)^-1/2 rho_c (rho_bar + dI)^-1/2

psi is an amplitude encoding of the full 768-d feature (centred, plus a constant
coordinate): the 10-qubit state padded with zeros. Padding does not change any
quantity, so the code works in 769 dimensions.

Round A2 (exps/qsd/README_A2.md) adds states built only from the mean and
covariance RSIAT already stores (exemplar-free, SSCA-updated means), an oracle
control, calibrated fusion with the CA head for every source, and saved scores.

SVF (exps/svf/README.md) adds a second view: a frozen copy of the network after task 0.
Its class statistics are computed when each class is learned and are exact forever (the
snapshot never changes), so ``snap_ridge`` is closed-form ridge on them, fused with the
plastic CA head like any other source.

Evaluation only, like utils/ca_heads.py: everything runs after eval_task with
private generators inside preserve_rng and never feeds back into training.
"""
import copy
import json
import logging
import math
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

HEADS = ("linear", "linear_rerun", "linear_noscale", "ridge_stats", "pgm_stale", "pgm_channel",
         "pgm_channel_hier", "pgm_fuse", "pgm_nonlinear",
         # Round A2: states from stored statistics (exemplar-free) and the oracle control.
         "pgm_stats_stale", "pgm_stats", "pgm_stats_hier", "pgm_stats_oracle",
         # SVF: ridge on exact statistics of the frozen task-0 snapshot view.
         "snap_ridge")
# Round A2 fusion: argmax(scale * cos + w * k_s * score_s), k_s = sigma_linear / sigma_s measured
# label-free on the current task's training images, so w is a relative weight for every source.
FUSE_SOURCES = ("ridge_stats", "pgm_channel", "pgm_channel_hier", "pgm_stats", "pgm_stats_hier", "snap_ridge")
FUSE_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
# SVF three-way fusion: linear + SCF ridge at its weight frozen in round A2 + w * snapshot view.
SCF_RIDGE_WEIGHT = 0.25
# Pre-registered settings (exps/qsd/README.md). Do not edit after seeing results.
DEFAULTS = {
    "qsd_delta": 0.1,          # whitening ridge, times 1/d (d = 2^qubits of the encoding)
    "qsd_channel_ridge": 0.1,  # channel ridge towards identity, times tr(Psi_old^T Psi_old)/D
    "qsd_ridge": 0.1,          # ridge_stats ridge, times tr(G)/768
    "qsd_nl_dim": 32,          # projection size of the two-copy (nonlinear) block: 5 qubits per copy
    "qsd_seed": 20261022,
    "qsd_batch": 128,
}
SWEEP = (0.01, 0.1, 1.0)  # sensitivity only, never used for selection
CA_SAMPLES = 256          # per class and epoch, as models/base.py::_stage2_compact_classifier


def qsd_settings(args):
    heads = args.get("qsd_heads")
    if not heads:
        return None
    if isinstance(heads, str) or not all(isinstance(h, str) for h in heads):
        raise ValueError("qsd_heads must be a list of head names.")
    unknown = sorted(set(heads) - set(HEADS))
    if unknown:
        raise ValueError("Unknown qsd_heads {}; choose from {}.".format(unknown, list(HEADS)))
    cfg = {key: args.get(key, default) for key, default in DEFAULTS.items()}
    for key in ("qsd_delta", "qsd_channel_ridge", "qsd_ridge"):
        if not (isinstance(cfg[key], (int, float)) and math.isfinite(cfg[key]) and cfg[key] > 0):
            raise ValueError("{} must be a positive number.".format(key))
    for key in ("qsd_nl_dim", "qsd_seed", "qsd_batch"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError("{} must be a positive integer.".format(key))
    for key in ("qsd_fuse", "qsd_save_scores"):
        cfg[key] = args.get(key, False)
        if not isinstance(cfg[key], bool):
            raise ValueError("{} must be true or false.".format(key))
    base = ["linear"] + [h for h in HEADS if h in heads and h != "linear"]
    cfg["fuse"] = [(s, w) for s in FUSE_SOURCES if s in base for w in FUSE_WEIGHTS] if cfg["qsd_fuse"] else []
    if cfg["qsd_fuse"] and not cfg["fuse"]:
        raise ValueError("qsd_fuse needs at least one of {} in qsd_heads.".format(list(FUSE_SOURCES)))
    cfg["fuse3"] = list(FUSE_WEIGHTS) if cfg["qsd_fuse"] and {"ridge_stats", "snap_ridge"} <= set(base) else []
    cfg["heads"] = base + [fuse_name(s, w) for s, w in cfg["fuse"]] + [fuse3_name(w) for w in cfg["fuse3"]]
    return cfg


def fuse_name(source, weight):
    return "fuse_{}_b{}".format(source, "{:g}".format(weight).replace(".", "p"))  # b0p25: log-safe \w name


def fuse3_name(weight):
    return "fuse3_snap_b{}".format("{:g}".format(weight).replace(".", "p"))


# ----------------------------------------------------------------------------- encoding
class Encoder:
    """Amplitude encodings, fixed after task 0 so every task uses the same map."""

    def __init__(self, center, scale, projection):
        self.center, self.scale, self.projection = center, scale, projection

    @classmethod
    def from_base(cls, z, nl_dim, generator):
        center = z.mean(0)
        scale = (z - center).square().sum(1).mean().sqrt().clamp_min(1e-8)  # sqrt tr Cov
        q, _ = torch.linalg.qr(torch.randn(z.shape[1], nl_dim, generator=generator, device=z.device,
                                           dtype=torch.float64))
        return cls(center, scale, q.to(z.dtype))

    def linear(self, z):
        """[ (z - m0) / s0 ; 1 ] normalised: the 10-qubit amplitude encoding (zero padding omitted)."""
        v = torch.cat([(z - self.center) / self.scale, torch.ones_like(z[:, :1])], 1)
        return v / v.norm(dim=1, keepdim=True)

    def nonlinear(self, z):
        """(|0>|psi_lin> + |1>|u (x) u>)/sqrt(2): a flag qubit adds a two-copy block, keeping the linear one."""
        u = ((z - self.center) / self.scale) @ self.projection
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        two = (u[:, :, None] * u[:, None, :]).reshape(z.shape[0], -1)
        return torch.cat([self.linear(z), two], 1) / math.sqrt(2.0)

    @staticmethod
    def nominal_dim(width):
        return 1 << math.ceil(math.log2(width))  # qubit register size: 769 -> 1024, 1793 -> 2048


class StateBank:
    """rho_c = L_c L_c^T with tr = 1, stored per class as a D x n_c factor."""

    def __init__(self):
        self.factors = {}

    def add(self, psi, labels):
        for c in torch.unique(labels).tolist():
            rows = psi[labels == c]
            self.factors[c] = rows.T / math.sqrt(rows.shape[0])

    def transport(self, channel):
        for c, factor in self.factors.items():
            moved = channel @ factor
            self.factors[c] = moved / moved.norm().clamp_min(1e-12)

    def stacked(self, num_classes):
        missing = [c for c in range(num_classes) if c not in self.factors]
        if missing:
            raise ValueError("No state for classes {}".format(missing[:5]))
        mats = [self.factors[c] for c in range(num_classes)]
        owner = torch.cat([torch.full((m.shape[1],), c, dtype=torch.long, device=m.device)
                           for c, m in enumerate(mats)])
        return torch.cat(mats, 1), owner


def fit_channel(psi_old, psi_new, rel):
    """A = argmin sum ||psi_new - A psi_old||^2 + lam ||A - I||^2 (closed form)."""
    n, dim = psi_old.shape
    lam = rel * n / dim  # tr(Psi_old^T Psi_old) = n for unit rows
    eye = torch.eye(dim, dtype=torch.float64, device=psi_old.device)
    cxx = psi_old.double().T @ psi_old.double() + lam * eye
    cyx = psi_new.double().T @ psi_old.double() + lam * eye
    channel = torch.linalg.solve(cxx, cyx.T).T
    return channel.to(psi_old.dtype), {"deviation": float((channel - eye).norm() / math.sqrt(dim)),
                                       "pairs": n, "ridge": lam}


# ----------------------------------------------------------------------------- measurement
def _inv_sqrt(rho, delta):
    evals, evecs = torch.linalg.eigh(rho.double())
    return (evecs * (evals.clamp_min(0) + delta).rsqrt()) @ evecs.T, evals


def class_scores(psi, factors, owner, num_classes, whitening, chunk=1024):
    """psi^T W rho_c W psi for every class (uniform priors drop out of the argmax)."""
    whitening = whitening.to(psi.dtype)
    out = []
    for i in range(0, psi.shape[0], chunk):
        proj = (psi[i:i + chunk] @ whitening) @ factors
        scores = torch.zeros(proj.shape[0], num_classes, dtype=proj.dtype, device=proj.device)
        out.append(scores.index_add_(1, owner, proj.square()))
    return torch.cat(out)


def pgm(psi, factors, owner, num_classes, delta):
    """Flat pretty-good measurement; returns normalised probabilities and diagnostics."""
    rho_bar = (factors * (1.0 / num_classes)) @ factors.T
    whitening, evals = _inv_sqrt(rho_bar, delta)
    scores = class_scores(psi, factors, owner, num_classes, whitening)
    probs = scores / scores.sum(1, keepdim=True).clamp_min(1e-30)
    spectrum = (evals.clamp_min(0) / evals.clamp_min(0).sum()).double()
    eff_rank = float(torch.exp(-(spectrum * torch.log(spectrum.clamp_min(1e-30))).sum()))
    return probs, {"effective_rank": eff_rank}


def _hierarchical(flat, task_of_class, within):
    """P(task | z) = sum_{c in T} P(c | z) (E_T = sum E_c), times P(c | z, T) from ``within(classes)``."""
    num_tasks = int(task_of_class.max()) + 1
    task_prob = torch.zeros(flat.shape[0], num_tasks, dtype=flat.dtype, device=flat.device)
    task_prob.index_add_(1, task_of_class, flat)
    probs = torch.zeros_like(flat)
    for t in range(num_tasks):
        classes = torch.nonzero(task_of_class == t).squeeze(1)
        probs[:, classes] = task_prob[:, t:t + 1] * within(classes)
    return probs


def pgm_hierarchical(psi, factors, owner, task_of_class, num_classes, delta):
    """Flat measurement for the task, then a PGM inside the task whitened by rho_T."""
    flat, _ = pgm(psi, factors, owner, num_classes, delta)

    def within(classes):
        cols = torch.isin(owner, classes)
        sub_owner = torch.searchsorted(classes, owner[cols])
        sub = factors[:, cols]
        rho_t = (sub * (1.0 / classes.numel())) @ sub.T
        whitening, _ = _inv_sqrt(rho_t, delta)
        scores = class_scores(psi, sub, sub_owner, classes.numel(), whitening)
        return scores / scores.sum(1, keepdim=True).clamp_min(1e-30)
    return _hierarchical(flat, task_of_class, within)


# ----------------------------------------------------------------------------- states from statistics (A2)
def stats_states(encoder, means, covs):
    """rho_c = E[v v^T] / tr, v = [(z - m0)/s0 ; 1], from one mean and covariance per class.

    Uses exactly what RSIAT stores (no per-sample features), so the states follow
    SSCA's mean updates. Equals the mixture of the encoded pure states psi(z)
    weighted by ||v||^2.
    """
    mu = (means.double() - encoder.center.double()) / encoder.scale.double()
    num, dim = mu.shape
    rhos = torch.empty(num, dim + 1, dim + 1, dtype=torch.float64, device=mu.device)
    rhos[:, :dim, :dim] = covs.double() / encoder.scale.double() ** 2 + mu[:, :, None] * mu[:, None, :]
    rhos[:, :dim, dim] = mu
    rhos[:, dim, :dim] = mu
    rhos[:, dim, dim] = 1.0
    return (rhos / rhos.diagonal(dim1=1, dim2=2).sum(1)[:, None, None]).float()


def dense_scores(psi, rhos, whitening, chunk=128):
    """psi^T W rho_c W psi for every class, with full (C, D, D) states."""
    v = psi @ whitening.to(psi.dtype)
    return torch.cat([torch.einsum("bi,cij,bj->bc", v[i:i + chunk], rhos, v[i:i + chunk])
                      for i in range(0, v.shape[0], chunk)])


def pgm_dense(psi, rhos, delta):
    whitening, _ = _inv_sqrt(rhos.mean(0), delta)
    scores = dense_scores(psi, rhos, whitening).clamp_min(0)
    return scores / scores.sum(1, keepdim=True).clamp_min(1e-30)


def pgm_dense_hierarchical(psi, rhos, task_of_class, delta):
    def within(classes):
        return pgm_dense(psi, rhos[classes], delta)
    return _hierarchical(pgm_dense(psi, rhos, delta), task_of_class, within)


def class_statistics(z, labels, num_classes):
    """Mean and covariance per class exactly as models/base.py::_compute_class_mean."""
    means = torch.zeros(num_classes, z.shape[1], dtype=torch.float64, device=z.device)
    covs = torch.zeros(num_classes, z.shape[1], z.shape[1], dtype=torch.float32, device=z.device)
    eye = torch.eye(z.shape[1], dtype=torch.float64, device=z.device)
    for c in range(num_classes):
        rows = z[labels == c].double()
        means[c] = rows.mean(0)
        covs[c] = (torch.cov(rows.T) + 1e-3 * eye).float()
    return means, covs


def fusion_scale(logits, scores):
    """k = sigma_linear / sigma_source: mean over samples of the spread across classes."""
    spread = lambda s: s.double().std(1).mean()
    return float(spread(logits) / spread(scores).clamp_min(1e-12))


def log_prob(probs):
    return torch.log(probs.double().clamp_min(1e-30))


def ridge_scores(means, covs, z, rel):
    """Closed-form ridge on stored statistics (DPCR-like): W = (sum_c S_c + lam I)^-1 [mu_c]."""
    means, covs = means.double(), covs.double()
    second = covs.sum(0) + means.T @ means
    lam = rel * float(torch.trace(second)) / second.shape[0]
    weight = torch.linalg.solve(second + lam * torch.eye(second.shape[0], dtype=torch.float64,
                                                          device=second.device), means.T)
    return z.double() @ weight


def ridge_stats(means, covs, z, rel):
    return ridge_scores(means, covs, z, rel).argmax(1)


def retrain_ca(learner, means, covs, scaled, seed, z_test):
    """Re-run RSIAT's classifier alignment from the pre-CA head with a private generator.

    Mirrors models/base.py::_stage2_compact_classifier (SGD, cosine schedule, 256
    samples per class and epoch, CE on scale * cosine); ``scaled`` toggles the
    0.9 + 0.1 (t+1)/(T+1) mean shrinkage. Both calls share eps and shuffles.
    """
    args, device = learner.args, z_test.device
    fc = copy.deepcopy(learner._network.fc).to(device)
    fc.load_state_dict(learner._network.fc.old_state_dict)
    for p in fc.parameters():
        p.requires_grad_(True)
    num_classes = means.shape[0]
    scale = torch.as_tensor(ca_mean_scale(learner.task_sizes, learner._cur_task), dtype=torch.float64,
                            device=device) if scaled else torch.ones(num_classes, dtype=torch.float64, device=device)
    centers = (means.double() * scale[:, None]).float()
    chols = torch.linalg.cholesky(covs.double()).float()
    optimizer = torch.optim.SGD(fc.parameters(), lr=args["init_lr"], momentum=0.9, weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args["ca_epochs"])
    gen = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(num_classes, device=device).repeat_interleave(CA_SAMPLES)
    with torch.enable_grad():
        for _ in range(args["ca_epochs"]):
            eps = torch.randn(num_classes, CA_SAMPLES, means.shape[1], generator=gen, device=device)
            data = (centers[:, None, :] + eps @ chols.transpose(1, 2)).reshape(-1, means.shape[1])
            perm = torch.randperm(data.shape[0], generator=gen, device=device)
            data, target = data[perm], labels[perm]
            for b in range(num_classes):
                part = slice(b * CA_SAMPLES, (b + 1) * CA_SAMPLES)
                logits = args["scale"] * fc(data[part])["logits"][:, :num_classes]
                loss = F.cross_entropy(logits, target[part])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
    with torch.no_grad():
        return fc(z_test)["logits"][:, :num_classes].argmax(1)


def ca_mean_scale(task_sizes, cur_task):
    return np.concatenate([np.full(size, 0.9 + 0.1 * (t + 1) / (cur_task + 1)) for t, size in enumerate(task_sizes)])


# ----------------------------------------------------------------------------- metrics
def head_metrics(pred, target, task_sizes, known):
    pred, target = np.asarray(pred), np.asarray(target)
    bounds = np.cumsum([0] + list(task_sizes))
    task_of = lambda labels: np.searchsorted(bounds, labels, side="right") - 1
    per_task = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (target >= lo) & (target < hi)
        per_task.append(float(np.around((pred[mask] == target[mask]).mean() * 100, 2)))
    wrong = pred != target
    tp, tt = task_of(pred), task_of(target)
    within, inter = (wrong & (tp == tt)).mean(), (wrong & (tp != tt)).mean()
    size = float(np.mean(task_sizes))
    total = float(bounds[-1])
    ratio = None
    if len(task_sizes) > 1 and within > 0:
        ratio = float((inter / (total - size)) / (within / (size - 1)))
    old, new = target < known, target >= known
    return {"top1": float(np.around((pred == target).sum() * 100 / len(target), decimals=2)),
            "task_accuracies": per_task,
            "old": float(np.around((pred[old] == target[old]).mean() * 100, 2)) if old.any() else None,
            "new": float(np.around((pred[new] == target[new]).mean() * 100, 2)) if new.any() else None,
            "task_prediction": float(np.around((tp == tt).mean() * 100, 2)),
            "pair_ratio": ratio}


# ----------------------------------------------------------------------------- driver
def _extract(network, loader, device):
    module = network.module if hasattr(network, "module") else network
    feats, labels = [], []
    for _, inputs, targets in loader:
        feats.append(module.extract_vector(inputs.to(device, non_blocking=True)).float())
        labels.append(torch.as_tensor(targets, device=device))
    return torch.cat(feats), torch.cat(labels)


def evaluate_qsd(learner, task, linear_pred, report_dir):
    from utils.qbridge_experiment import preserve_rng
    cfg = qsd_settings(learner.args)
    if cfg is None:
        return None
    start = time.perf_counter()
    with preserve_rng():
        modes = [(module, module.training) for module in learner._network.modules()]
        result = _evaluate(learner, cfg, task, linear_pred)
        for module, training in modes:
            module.training = training
    result["seconds"] = time.perf_counter() - start
    curves = getattr(learner, "_qsd_curves", {})
    for head, m in result["heads"].items():
        curves.setdefault(head, []).append(m["top1"])
        logging.info("QSD task=%d head=%s top1=%.2f old=%s new=%s task_pred=%.2f pair_ratio=%s", task, head,
                     m["top1"], m["old"], m["new"], m["task_prediction"], m["pair_ratio"])
        logging.info("QSD tasks head=%s: %s", head, json.dumps(m["task_accuracies"]))
        logging.info("QSD mech head=%s: %s", head, json.dumps({"task_prediction": m["task_prediction"],
                                                               "pair_ratio": m["pair_ratio"]}))
        logging.info("QSD curve head=%s: %s", head, json.dumps(curves[head]))
    learner._qsd_curves = curves
    logging.info("QSD task=%d diag=%s seconds=%.1f", task, json.dumps(result["diagnostics"]), result["seconds"])
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "qsd_task_%d.json" % task), "w", encoding="utf-8") as stream:
        json.dump({k: v for k, v in result.items() if k not in ("predictions", "target", "scores")}, stream,
                  indent=2)
    np.savez_compressed(os.path.join(report_dir, "qsd_predictions_task_%d.npz" % task),
                        target=result["target"], **result["predictions"])
    if result["scores"]:
        np.savez_compressed(os.path.join(report_dir, "qsd_scores_task_%d.npz" % task),
                            target=result["target"], **result["scores"])
    return result


def _evaluate(learner, cfg, task, linear_pred):
    device = learner._device
    known, total = learner._known_classes, learner._total_classes
    task_sizes = list(learner.task_sizes)
    heads = cfg["heads"]
    diagnostics = {}

    with torch.no_grad():
        z_test, target = learner._extract_vectors(learner.test_loader)
        z_test = torch.as_tensor(z_test, dtype=torch.float32, device=device)
        loader = _train_loader(learner, cfg, np.arange(known, total), task)
        z_new, y_new = _extract(learner._network, loader, device)
        z_old = None
        if task > 0:
            z_old, y_old = _extract(learner._old_network, loader, device)
            if not torch.equal(y_old, y_new):
                raise RuntimeError("QSD channel pairs visited samples in different orders.")

        state = getattr(learner, "_qsd_state", None)
        if state is None:
            if task != 0:
                raise RuntimeError("QSD must observe task 0 to fix its encoding.")
            encoder = Encoder.from_base(z_new, cfg["qsd_nl_dim"], torch.Generator(device=device).manual_seed(cfg["qsd_seed"]))
            state = {"encoder": encoder, "stale": StateBank(), "channel": StateBank(), "nonlinear": StateBank()}
            learner._qsd_state = state
        enc = state["encoder"]
        if z_old is not None:
            if "pgm_channel" in heads or "pgm_channel_hier" in heads or "pgm_fuse" in heads:
                a_lin, diagnostics["channel"] = fit_channel(enc.linear(z_old), enc.linear(z_new), cfg["qsd_channel_ridge"])
                state["channel"].transport(a_lin)
            if "pgm_nonlinear" in heads:
                a_nl, diagnostics["channel_nonlinear"] = fit_channel(enc.nonlinear(z_old), enc.nonlinear(z_new),
                                                                     cfg["qsd_channel_ridge"])
                state["nonlinear"].transport(a_nl)
        psi_new = enc.linear(z_new)
        state["stale"].add(psi_new, y_new)
        state["channel"].add(psi_new, y_new)
        if "pgm_nonlinear" in heads:
            state["nonlinear"].add(enc.nonlinear(z_new), y_new)

        for c in range(known, total):  # the mean each class had when it was learned (before any SSCA update)
            state.setdefault("learned_means", {})[c] = np.array(learner._class_means[c], dtype=np.float64)

        predictions = {"linear": np.asarray(linear_pred)}
        wanted = set(heads) | {source for source, _ in cfg["fuse"]}
        psi_test = enc.linear(z_test)
        d_lin = Encoder.nominal_dim(psi_test.shape[1])
        delta = cfg["qsd_delta"] / d_lin
        task_of_class = torch.as_tensor(np.repeat(np.arange(len(task_sizes)), task_sizes), device=device)
        # Fusion sources: (z, psi) -> class scores whose argmax is the source's own prediction.
        sources, test_scores = {}, {}
        if "pgm_stale" in heads:
            factors, owner = state["stale"].stacked(total)
            predictions["pgm_stale"] = pgm(psi_test, factors, owner, total, delta)[0].argmax(1).cpu().numpy()
        if {"pgm_channel", "pgm_channel_hier", "pgm_fuse"} & wanted:
            factors, owner = state["channel"].stacked(total)
            probs, diagnostics["pgm_channel"] = pgm(psi_test, factors, owner, total, delta)
            predictions["pgm_channel"] = probs.argmax(1).cpu().numpy()
            sources["pgm_channel"] = lambda z, psi, f=factors, o=owner: log_prob(pgm(psi, f, o, total, delta)[0])
            test_scores["pgm_channel"] = log_prob(probs)
            diagnostics["delta_sensitivity"] = {
                str(s): float(np.around((pgm(psi_test, factors, owner, total, s / d_lin)[0].argmax(1).cpu().numpy()
                                         == np.asarray(target)).mean() * 100, 2)) for s in SWEEP}
            if "pgm_channel_hier" in wanted:
                sources["pgm_channel_hier"] = lambda z, psi, f=factors, o=owner: log_prob(
                    pgm_hierarchical(psi, f, o, task_of_class, total, delta))
            if "pgm_fuse" in heads:
                logits = learner.args["scale"] * learner._network.fc(z_test)["logits"][:, :total]
                fused = F.log_softmax(logits.double(), 1) + torch.log(probs.double().clamp_min(1e-30))
                predictions["pgm_fuse"] = fused.argmax(1).cpu().numpy()
        if "pgm_nonlinear" in heads:
            psi_nl = enc.nonlinear(z_test)
            nl_factors, nl_owner = state["nonlinear"].stacked(total)
            probs_nl, diagnostics["pgm_nonlinear"] = pgm(psi_nl, nl_factors, nl_owner, total,
                                                         cfg["qsd_delta"] / Encoder.nominal_dim(psi_nl.shape[1]))
            predictions["pgm_nonlinear"] = probs_nl.argmax(1).cpu().numpy()

        means = torch.as_tensor(np.asarray(learner._class_means[:total]), dtype=torch.float32, device=device)
        covs = torch.as_tensor(learner._class_covs[:total]).to(device=device, dtype=torch.float32)
        if "ridge_stats" in wanted:
            sources["ridge_stats"] = lambda z, psi: ridge_scores(means, covs, z, cfg["qsd_ridge"])
            diagnostics["ridge_sensitivity"] = {
                str(s): float(np.around((ridge_stats(means, covs, z_test, s).cpu().numpy()
                                         == np.asarray(target)).mean() * 100, 2)) for s in SWEEP}
        if {"pgm_stats", "pgm_stats_hier"} & wanted:
            rhos = stats_states(enc, means, covs)
            sources["pgm_stats"] = lambda z, psi: log_prob(pgm_dense(psi, rhos, delta))
            if "pgm_stats_hier" in wanted:
                sources["pgm_stats_hier"] = lambda z, psi: log_prob(
                    pgm_dense_hierarchical(psi, rhos, task_of_class, delta))
        if "pgm_stats_stale" in heads:
            learned = torch.as_tensor(np.stack([state["learned_means"][c] for c in range(total)]),
                                      dtype=torch.float32, device=device)
            predictions["pgm_stats_stale"] = pgm_dense(psi_test, stats_states(enc, learned, covs),
                                                       delta).argmax(1).cpu().numpy()
            diagnostics["ssca_mean_shift"] = float(((means - learned).norm(dim=1) / learned.norm(dim=1)).mean())
        if "pgm_stats_oracle" in heads:  # control only: re-reads old-class images (not exemplar-free)
            z_all, y_all = z_new, y_new
            if known > 0:
                z_prev, y_prev = _extract(learner._network, _train_loader(learner, cfg, np.arange(0, known),
                                                                          10_000 + task), device)
                z_all, y_all = torch.cat([z_prev, z_new]), torch.cat([y_prev, y_new])
            o_means, o_covs = class_statistics(z_all, y_all, total)
            predictions["pgm_stats_oracle"] = pgm_dense(psi_test, stats_states(enc, o_means, o_covs),
                                                        delta).argmax(1).cpu().numpy()
            diagnostics["oracle_mean_error"] = float(((means.double() - o_means).norm(dim=1)
                                                      / o_means.norm(dim=1)).mean())
        train_view = {}  # sources scored on another view's features: their training-image scores
        if "snap_ridge" in wanted:
            if "snapshot" not in state:
                if task != 0:
                    raise RuntimeError("SVF must observe task 0 to take its snapshot.")
                snapshot = copy.deepcopy(learner._network).eval()
                snapshot.requires_grad_(False)
                state.update(snapshot=snapshot, snap_means=[], snap_covs=[])
            z0_new, y0_new = _extract(state["snapshot"], loader, device)
            if not torch.equal(y0_new, y_new):
                raise RuntimeError("SVF snapshot visited samples in a different order.")
            new_means, new_covs = class_statistics(z0_new, y0_new - known, total - known)
            state["snap_means"].append(new_means.float())  # appended once, never updated
            state["snap_covs"].append(new_covs)
            snap_means, snap_covs = torch.cat(state["snap_means"]), torch.cat(state["snap_covs"])
            z0_test, y0_test = _extract(state["snapshot"], learner.test_loader, device)
            if not np.array_equal(y0_test.cpu().numpy(), np.asarray(target)):
                raise RuntimeError("SVF snapshot test pass visited samples in a different order.")
            test_scores["snap_ridge"] = ridge_scores(snap_means, snap_covs, z0_test, cfg["qsd_ridge"])
            train_view["snap_ridge"] = ridge_scores(snap_means, snap_covs, z0_new, cfg["qsd_ridge"])
            predictions["snap_ridge"] = test_scores["snap_ridge"].argmax(1).cpu().numpy()
        for name, fn in sources.items():
            if name not in test_scores:
                test_scores[name] = fn(z_test, psi_test)
            if name in heads:
                predictions[name] = test_scores[name].argmax(1).cpu().numpy()

        logits = learner.args["scale"] * learner._network.fc(z_test)["logits"][:, :total]
        if cfg["fuse"]:
            train_logits = learner.args["scale"] * learner._network.fc(z_new)["logits"][:, :total]
            diagnostics["fuse_scale"] = {}
            for source in dict.fromkeys(s for s, _ in cfg["fuse"]):
                train_scores = train_view[source] if source in train_view else sources[source](z_new, psi_new)
                k = fusion_scale(train_logits, train_scores)
                diagnostics["fuse_scale"][source] = k
                for weight in FUSE_WEIGHTS:
                    fused = logits.double() + weight * k * test_scores[source].double()
                    predictions[fuse_name(source, weight)] = fused.argmax(1).cpu().numpy()
            scale = diagnostics["fuse_scale"]
            for weight in cfg["fuse3"]:
                fused = (logits.double() + SCF_RIDGE_WEIGHT * scale["ridge_stats"] * test_scores["ridge_stats"].double()
                         + weight * scale["snap_ridge"] * test_scores["snap_ridge"].double())
                predictions[fuse3_name(weight)] = fused.argmax(1).cpu().numpy()
        scores = {}
        if cfg["qsd_save_scores"]:  # float16 is enough to redo any fusion offline
            scores = {name: s.float().cpu().numpy().astype(np.float16)
                      for name, s in dict(test_scores, linear=logits).items()}

    retrain = [h for h in ("linear_rerun", "linear_noscale") if h in heads]
    if retrain:
        default_ca = (learner.args.get("ca_logit_norm") in (None, 0) and not learner.args.get("ca_task_margin"))
        for head in retrain:
            if task == 0 or not learner.args.get("ca", True) or not default_ca:
                predictions[head] = np.asarray(linear_pred)  # no CA at task 0 (or non-default CA): same head
            else:
                pred = retrain_ca(learner, means, covs, head == "linear_rerun", cfg["qsd_seed"] + 7 * task, z_test)
                predictions[head] = pred.cpu().numpy()

    target = np.asarray(target)
    metrics = {h: head_metrics(predictions[h], target, task_sizes, known) for h in heads if h in predictions}
    return {"task": task, "settings": {k: v for k, v in cfg.items()}, "heads": metrics,
            "diagnostics": diagnostics, "predictions": predictions, "target": target, "scores": scores}


def _train_loader(learner, cfg, classes, offset):
    from utils.stats_diagnostics import deterministic_loader
    dataset = learner.data_manager.get_dataset(classes, source="train", mode="test")
    return deterministic_loader(dataset, cfg["qsd_batch"], int(learner.args.get("stats_num_workers", 2)),
                                False, cfg["qsd_seed"] + offset)
