"""QKSR P0 probe: read-only measurements behind the QKSR weaknesses (reports/Khắc phục điểm yếu QKSR.md, P0).

Nothing here changes training. Gradients are taken with ``torch.autograd.grad`` (no ``.grad`` writes,
no optimizer state), feature passes run under ``torch.no_grad`` + ``preserve_rng`` in eval mode, and
the module touches no RNG. ``tests/test_qksr_probe.py`` checks that weights stay bit-identical.

Measured, per task:
  * gradients on the trainable adapter parameters, every ``qksr_probe_every`` steps:
    norms and pairwise cosines of CosFace (c), alignment KD (kd), separation (sep) and, in task 0,
    the base steering loss (base); share of sep that survives a PCGrad projection against c;
  * at the end of the task, on up to ``qksr_probe_per_class`` train images of every seen class
    (current network, eval transform):
    - W1: share of between- / within-class scatter of the LayerNormed features inside the row space
      of the 768->n projector;
    - angles: saturation of pi*tanh (|theta| > 0.9 pi) and spread per qubit;
    - selectivity: AUC that a different-class pair is farther than a same-class pair, for the PQK
      distance and for classical twins on the same inputs (Euclidean on theta, Euclidean and cosine
      on z); kernel means within / between class at the trained gamma;
    - alignment: centred kernel alignment of the PQK Gram with RBF Grams (median bandwidth);
    - the loss pairs (old prototypes through old_ae vs features of the new classes): kernel mean
      over all pairs and over the k nearest prototypes of each sample.
"""
import json
import logging
import math
import statistics

import numpy as np
import torch
import torch.nn.functional as F

DEFAULTS = {"qksr_probe": False, "qksr_probe_every": 10, "qksr_probe_per_class": 20,
            "qksr_probe_topk": 5, "qksr_probe_seed": 20261031}
PAIR_SAMPLE = 200000  # pairs drawn for each AUC


def probe_settings(args):
    cfg = {k: args.get(k, v) for k, v in DEFAULTS.items()}
    if not cfg["qksr_probe"]:
        return None
    for key in ("qksr_probe_every", "qksr_probe_per_class", "qksr_probe_topk"):
        if int(cfg[key]) < 1:
            raise ValueError("{} must be >= 1.".format(key))
    if args.get("resume", False):
        raise ValueError("qksr_probe does not support resume.")
    return cfg


def _flat_norm(grads):
    return math.sqrt(sum(float(g.double().square().sum()) for g in grads))


def _dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def _auc(within, between):
    """P(between > within) + 0.5 P(tie), by ranks."""
    values = torch.cat([within, between]).double()
    ranks = torch.empty_like(values)
    order = values.argsort()
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=torch.float64, device=values.device)
    # average ranks over ties
    uniq, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = torch.zeros_like(uniq).index_add_(0, inverse, ranks)
        ranks = (sums / counts.double())[inverse]
    n_w, n_b = within.numel(), between.numel()
    r_b = ranks[n_w:].sum()
    return float((r_b - n_b * (n_b + 1) / 2) / (n_w * n_b))


def _cka(k1, k2):
    n = k1.shape[0]
    h = torch.eye(n, dtype=torch.float64, device=k1.device) - 1.0 / n
    a, b = h @ k1.double() @ h, h @ k2.double() @ h
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


class QKSRProbe:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.step = 0
        self.records = []

    # ------------------------------------------------------------------ gradients
    @staticmethod
    def _params(learner):
        return [p for p in learner._network_module_ptr.convnet.parameters() if p.requires_grad]

    def on_step(self, learner, terms):
        """terms: name -> scalar loss (already weighted as in the total loss). Read-only."""
        self.step += 1
        if (self.step - 1) % int(self.cfg["qksr_probe_every"]):
            return
        params = self._params(learner)
        grads = {}
        for name, loss in terms.items():
            if not torch.is_tensor(loss) or not loss.requires_grad:
                continue
            got = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            grads[name] = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, got)]
        rec = {"norm_" + n: _flat_norm(g) for n, g in grads.items()}
        names = list(grads)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                na, nb = rec["norm_" + a], rec["norm_" + b]
                rec["cos_{}_{}".format(a, b)] = (_dot(grads[a], grads[b]) / (na * nb)) if na * nb > 0 else None
        if "c" in grads and "kd" in grads and "sep" in grads:
            reg = [x + y for x, y in zip(grads["kd"], grads["sep"])]
            nr = _flat_norm(reg)
            rec["norm_reg"] = nr
            rec["cos_c_reg"] = _dot(grads["c"], reg) / (rec["norm_c"] * nr) if rec["norm_c"] * nr > 0 else None
        if rec.get("cos_c_sep") is not None:
            cos = rec["cos_c_sep"]
            rec["pcgrad_keep_sep"] = math.sqrt(max(0.0, 1 - cos * cos)) if cos < 0 else 1.0
        self.records.append(rec)

    def end_epoch(self, task, epoch):
        if not self.records:
            return
        out = {"steps": len(self.records)}
        keys = sorted({k for r in self.records for k in r})
        for key in keys:
            vals = [r[key] for r in self.records if r.get(key) is not None]
            if not vals:
                continue
            out[key] = {"median": statistics.median(vals), "mean": statistics.mean(vals)}
            if key.startswith("cos_"):
                out[key]["frac_neg"] = sum(v < 0 for v in vals) / len(vals)
                out[key]["frac_below_m0.5"] = sum(v < -0.5 for v in vals) / len(vals)
        logging.info("QKSRPROBE grad task=%d epoch=%d %s", task, epoch + 1, json.dumps(out))
        self.records = []

    # ------------------------------------------------------------------ end of task
    def _sample(self, learner):
        from utils.stats_diagnostics import deterministic_loader
        total = learner._total_classes
        dataset = learner.data_manager.get_dataset(np.arange(0, total), source="train", mode="test")
        loader = deterministic_loader(dataset, 128, int(learner.args.get("stats_num_workers", 2)), False,
                                      int(self.cfg["qksr_probe_seed"]) + learner._cur_task)
        network = learner._network.module if hasattr(learner._network, "module") else learner._network
        cap, count = int(self.cfg["qksr_probe_per_class"]), {}
        feats, labels = [], []
        for _, inputs, targets in loader:
            targets = torch.as_tensor(targets)
            keep = []
            for i, y in enumerate(targets.tolist()):
                if count.get(y, 0) < cap:
                    count[y] = count.get(y, 0) + 1
                    keep.append(i)
            if keep:
                idx = torch.as_tensor(keep)
                feats.append(network.extract_vector(inputs[idx].to(self.device, non_blocking=True)).float())
                labels.append(targets[idx].to(self.device))
            if len(count) == total and all(v >= cap for v in count.values()):
                break
        return torch.cat(feats), torch.cat(labels)

    def _pair_aucs(self, dist, y, task_of):
        n = y.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(int(self.cfg["qksr_probe_seed"]))
        i = torch.randint(0, n, (PAIR_SAMPLE,), generator=gen).to(dist.device)
        j = torch.randint(0, n, (PAIR_SAMPLE,), generator=gen).to(dist.device)
        keep = i != j
        i, j = i[keep], j[keep]
        d = dist[i, j]
        same = y[i] == y[j]
        same_task = task_of[y[i]] == task_of[y[j]]
        out = {"auc_class": _auc(d[same], d[~same])}
        cross = ~same & ~same_task
        within_task = ~same & same_task
        if cross.any() and within_task.any():
            out["auc_cross_task_vs_same_class"] = _auc(d[same], d[cross])
            out["auc_within_task_vs_same_class"] = _auc(d[same], d[within_task])
        out["median_same"] = float(d[same].median())
        out["median_diff"] = float(d[~same].median())
        return out

    @staticmethod
    def _sqdist(a, b=None):
        b = a if b is None else b
        return (a.square().sum(1)[:, None] + b.square().sum(1)[None] - 2 * a @ b.T).clamp_min(0)

    def end_task(self, learner):
        from utils.qbridge_experiment import preserve_rng
        with preserve_rng(), torch.no_grad():
            modes = [(m, m.training) for m in learner._network.modules()]
            ae_mode = learner.old_ae.training if learner.old_ae is not None else None
            learner._network.eval()
            if learner.old_ae is not None:
                learner.old_ae.eval()
            try:
                report = self._end_task(learner)
            finally:
                for m, training in modes:
                    m.training = training
                if learner.old_ae is not None:
                    learner.old_ae.train(ae_mode)
        logging.info("QKSRPROBE end task=%d %s", learner._cur_task, json.dumps(report))
        return report

    def _end_task(self, learner):
        z, y = self._sample(learner)
        sizes = list(learner.task_sizes)
        task_of = torch.as_tensor(np.repeat(np.arange(len(sizes)), sizes), device=z.device)
        report = {"n": int(z.shape[0]), "classes": int(learner._total_classes)}
        twins = {"euclid_z": self._sqdist(z), "cosine_z": 1 - F.normalize(z, dim=1) @ F.normalize(z, dim=1).T}
        qk = learner.quantum_kernel
        if qk is not None:
            proj = qk.projector
            z_in = z.to(proj.linear.weight.dtype)
            x = z_in if getattr(proj, "skip_norm", False) else proj.norm(z_in)  # P2 refit bypasses the LayerNorm
            pre = proj.linear(x)
            theta = math.pi * torch.tanh(pre)
            # W1: share of class scatter inside the projector's row space (random n-dim subspace: n/D)
            w = proj.linear.weight.double()
            basis = torch.linalg.qr(w.T)[0]  # D x n
            xd = x.double()
            classes = torch.unique(y)
            means = torch.stack([xd[y == c].mean(0) for c in classes])
            between = means - xd.mean(0)
            within = xd - means[torch.searchsorted(classes, y)]
            share = lambda m: float((m @ basis).square().sum() / m.square().sum().clamp_min(1e-30))
            report["w1"] = {"qubits": int(w.shape[0]), "dim": int(w.shape[1]), "random_share": w.shape[0] / w.shape[1],
                            "between_share": share(between), "within_share": share(within)}
            report["angles"] = {"frac_saturated_0.9pi": float((theta.abs() > 0.9 * math.pi).float().mean()),
                                "frac_pre_abs_gt2": float((pre.abs() > 2).float().mean()),
                                "theta_std_per_qubit": [float(v) for v in theta.std(0)]}
            rep = qk.encode(z)
            pick = lambda r, s: {k: v[s] for k, v in r.items()} if isinstance(r, dict) else r[s]
            dist_q = torch.cat([qk.distance(pick(rep, slice(i, i + 256)), rep).double()  # chunked: N x N x 4n
                                for i in range(0, z.shape[0], 256)])
            twins["euclid_theta"] = self._sqdist(theta.double())
            gamma = float(qk.gamma)
            kq = torch.exp(-gamma * dist_q)
            same = y[:, None] == y[None]
            off = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
            report["pqk"] = dict(self._pair_aucs(dist_q, y, task_of), gamma=gamma,
                                 kernel_mean_same=float(kq[same & off].mean()), kernel_mean_diff=float(kq[~same].mean()),
                                 kernel_mean_all=float(kq[off].mean()))
            rbf = lambda d: torch.exp(-d / d[off].median().clamp_min(1e-30))
            report["cka_pqk_vs"] = {"rbf_theta": _cka(kq, rbf(twins["euclid_theta"])),
                                    "rbf_z": _cka(kq, rbf(twins["euclid_z"].double()))}
            known = learner._known_classes
            if learner._cur_task > 0 and learner.old_ae is not None and known > 0:
                protos = torch.from_numpy(learner._class_means[:known]).float().to(z.device)
                protos = learner.old_ae(protos)
                new = z[y >= known]
                kp = qk.kernel(qk.encode(protos), qk.encode(new)).double()  # K_old x n_new
                k = min(int(self.cfg["qksr_probe_topk"]), known)
                report["loss_pairs"] = {"kernel_mean_all": float(kp.mean()),
                                        "kernel_mean_topk": float(kp.topk(k, dim=0).values.mean()),
                                        "kernel_std_all": float(kp.std()), "topk": k}
        for name, dist in twins.items():
            report["twin_" + name] = self._pair_aucs(dist.double(), y, task_of)
        return report
