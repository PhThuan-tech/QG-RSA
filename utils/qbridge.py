"""Q-BRIDGE v1.1 losses and frozen selectors. No dataset or inference changes."""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULTS = dict(enabled=False, variant="custom", risk_mode="uniform", risk_gain=0.5,
                margin_x=0.1, margin_old=0.1, lambda_cross=0.1, lambda_guard=0.1,
                cross_normalization="mean_all_old_pairs", guard_refresh="each_step",
                detach_risk=True, detach_guard_anchors=True,
                old_head_grad_in_added_losses=False, kernel_source="base_frozen",
                prototype_source="previous_task_means", prototype_cache="per_task",
                legacy_incremental_separation="replace", preserve_original_ssca_ca=True)
MODES = {"uniform", "cosine", "raw_pqk", "calibrated_pqk", "rbf_proj", "mlp",
         "shuffled_raw_pqk", "constant_raw_pqk"}
VARIANTS = {"R1": "uniform", "R2": "uniform", "R3": "uniform", "R4": "cosine",
            "R5": "raw_pqk", "R6": "calibrated_pqk", "R7": "rbf_proj", "R8": "mlp",
            "R9": "shuffled_raw_pqk", "R10": "constant_raw_pqk"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def trusted_load(path):
    # Only load user-provided, trusted local torch artifacts.
    return torch.load(path, map_location="cpu", weights_only=False)


def config_from_args(args):
    supplied = args.get("qbridge", {})
    if not supplied.get("enabled", False):
        return None  # Disabled mode has no validation, artifact, RNG or loss side effects.
    cfg = {**copy.deepcopy(DEFAULTS), **copy.deepcopy(supplied)}
    for key in ("risk_gain", "lambda_cross", "lambda_guard", "margin_x", "margin_old"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) < 0:
            raise ValueError("qbridge.{} must be finite and nonnegative".format(key))
    for key in ("cross_normalization", "guard_refresh", "detach_risk", "detach_guard_anchors",
                "old_head_grad_in_added_losses", "kernel_source", "prototype_source",
                "prototype_cache", "legacy_incremental_separation", "preserve_original_ssca_ca"):
        if cfg[key] != DEFAULTS[key]:
            raise ValueError("Unsupported v1.1 policy qbridge.{}={!r}".format(key, cfg[key]))
    if args.get("use_quantum_kernel_inc", False):
        raise ValueError("Q-BRIDGE replaces separation: use_quantum_kernel_inc must be false")
    if not args.get("use_quantum_kernel_base", False):
        raise ValueError("Q-BRIDGE requires the unchanged QKSR base")
    if cfg["risk_mode"] not in MODES:
        raise ValueError("Unknown Q-BRIDGE risk_mode")
    variant = cfg["variant"]
    if variant != "custom":
        if variant not in VARIANTS or cfg["risk_mode"] != VARIANTS[variant]:
            raise ValueError("Variant/selector mismatch")
        if variant == "R1":
            if cfg.get("classification_mode") != "all_seen_cosface" or any(
                    cfg[k] != 0 for k in ("lambda_cross", "lambda_guard", "risk_gain")):
                raise ValueError("R1 must replace classification only and retain alignment")
        else:
            if cfg.get("classification_mode", "new_cosface") != "new_cosface":
                raise ValueError("Only R1 replaces classification")
            if cfg["lambda_cross"] <= 0 or (variant != "R2" and cfg["lambda_guard"] <= 0):
                raise ValueError("Locked variants require positive declared loss weights")
            if variant == "R2" and cfg["lambda_guard"] != 0:
                raise ValueError("R2 guard must be off")
            if variant in ("R2", "R3") and cfg["risk_gain"] != 0:
                raise ValueError("Uniform controls require risk_gain=0")
            if variant not in ("R2", "R3") and cfg["risk_gain"] <= 0:
                raise ValueError("Selector variants require risk_gain>0")
    if cfg["risk_mode"] == "calibrated_pqk":
        cal = cfg.get("calibration", {})
        if not cal.get("enabled") or not cal.get("strict_require_fitted", True):
            raise ValueError("Calibrated mode requires strict fitted calibration")
        if cal.get("recalibrate_per_task", False):
            raise ValueError("Incremental recalibration is forbidden")
        for key, expected in (("fit_source", "base_train_only"), ("fit_loss", "unweighted_bce"),
                              ("monotonic", "softplus_slope_sigmoid")):
            if cal.get(key, expected) != expected:
                raise ValueError("Unsupported calibration policy: " + key)
    elif cfg.get("calibration", {}).get("enabled", False):
        raise ValueError("Calibration enabled for a raw-selector variant")
    return cfg


def id_positions(ids, seen_ids, device):
    seen = list(map(int, seen_ids))
    if len(set(seen)) != len(seen):
        raise ValueError("Duplicate head class IDs")
    lookup = {label: pos for pos, label in enumerate(seen)}
    values = ids.detach().cpu().tolist() if torch.is_tensor(ids) else list(ids)
    try:
        positions = [lookup[int(label)] for label in values]
    except KeyError as exc:
        raise ValueError("Unknown global class ID: {}".format(exc)) from exc
    return torch.tensor(positions, device=device, dtype=torch.long)


def safe_cosine_scores(features, weights, seen_ids, old_ids):
    if weights.shape[0] != len(seen_ids):
        raise ValueError("Head rows and class IDs do not match")
    old_pos = id_positions(old_ids, seen_ids, weights.device)
    mask = torch.zeros(weights.shape[0], device=weights.device, dtype=torch.bool)
    mask[old_pos] = True
    safe = torch.where(mask[:, None], weights.detach(), weights)
    return F.normalize(features, dim=-1) @ F.normalize(safe, dim=-1).T


def margin_violations(scores, labels, seen_ids, old_ids, margin):
    correct = id_positions(labels, seen_ids, scores.device)
    old = id_positions(old_ids, seen_ids, scores.device)
    return (margin + scores[:, old] - scores.gather(1, correct[:, None])).relu()


def substitute_active_risk(risk, active, mode, generator):
    risk = risk.detach().clone()
    values = risk[active]
    if values.numel() == 0:
        return risk
    if mode == "constant_raw_pqk":
        risk[active] = values.mean()
    elif mode == "shuffled_raw_pqk" and values.numel() > 1:
        permutation = torch.randperm(values.numel(), generator=generator, device="cpu").to(values.device)
        risk[active] = values[permutation]
    return risk


def cross_margin_loss(violation, risk=None, gain=0.0):
    if violation.numel() == 0:
        return violation.sum() * 0
    if risk is None:
        return violation.square().mean()
    if risk.shape != violation.shape or not torch.isfinite(risk).all():
        raise ValueError("Invalid risk shape/values")
    if (risk < 0).any() or (risk > 1).any():
        raise ValueError("Risk must be in [0,1]")
    return ((1 + gain * risk.detach()) * violation.square()).mean()


def guard_loss(head, anchors, seen_ids, old_ids, new_ids, margin):
    if not old_ids or not new_ids:
        return anchors.detach().sum() * 0
    scores = head.qbridge_scores(anchors.detach(), seen_ids, old_ids)
    old = id_positions(old_ids, seen_ids, scores.device)
    new = id_positions(new_ids, seen_ids, scores.device)
    violations = (margin + scores[:, new].max(1).values
                  - scores.gather(1, old[:, None]).squeeze(1)).relu()
    return violations.square().mean()


class MonotoneCalibrator(nn.Module):
    def __init__(self, u=0.0, b=0.0):
        super().__init__()
        self.u = nn.Parameter(torch.tensor(float(u), dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(float(b), dtype=torch.float64))

    def logits(self, x):
        return (F.softplus(self.u) + 1e-6) * x.to(self.u.dtype) + self.b

    def forward(self, x):
        return torch.sigmoid(self.logits(x)).to(x.dtype)


def read_artifact(path, kind, base_hash):
    if not path or not Path(path).is_file():
        raise ValueError("Required fitted {} artifact not found: {}".format(kind, path))
    artifact = trusted_load(path)
    if artifact.get("kind") != kind or artifact.get("base_checkpoint_hash") != base_hash:
        raise ValueError("Incompatible {} artifact or base checkpoint hash".format(kind))
    if artifact.get("status") != "fitted" or artifact.get("source") != "base_train_only":
        raise ValueError("Artifact is not fitted on base training data")
    return artifact


def kernel_from_config(config):
    from utils.quantum_kernel import QuantumKernelModule
    return QuantumKernelModule(**config)


class BridgeRuntime:
    """Training-only owner of caches/artifacts, never attached to the inference network."""
    def __init__(self, cfg, pqk, base_hash, seed, device):
        self.cfg = cfg
        self.pqk = pqk
        self.base_hash = base_hash
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed) + 71031)
        self.calibrator = None
        self.selector = None
        self.artifact_hash = None
        self.fit_id = None
        self.encode_count = 0
        self.cache = None
        self.last = {}
        if pqk is None or not bool(pqk.gamma_initialized):
            raise ValueError("A calibrated, trained base PQK is required")
        pqk.set_inc_mode("frozen")
        pqk.eval()
        self.base_gamma = pqk.gamma.detach().clone()
        mode = cfg["risk_mode"]
        if mode == "calibrated_pqk":
            path = cfg.get("calibrator_artifact_path")
            artifact = read_artifact(path, "calibrator", base_hash)
            if not artifact.get("gate_passed", False):
                raise ValueError("Calibration gate failed; R6 is unavailable")
            for key, artifact_key in (("proxy_margin", "proxy_margin"), ("split_seed", "split_seed")):
                if key in cfg.get("calibration", {}) and cfg["calibration"][key] != artifact.get(artifact_key):
                    raise ValueError("Calibrator protocol mismatch: " + key)
            fraction_keys = ("prototype_fit_fraction", "calibrator_fit_fraction", "calibrator_eval_fraction")
            if any(key in cfg.get("calibration", {}) for key in fraction_keys):
                fractions = [cfg["calibration"].get(key, default)
                             for key, default in zip(fraction_keys, (.6, .2, .2))]
                if fractions != artifact.get("fractions"):
                    raise ValueError("Calibrator protocol mismatch: P/F/V fractions")
            self.calibrator = MonotoneCalibrator(**artifact["parameters"]).to(device)
            self.calibrator.requires_grad_(False).eval()
            if not all(torch.isfinite(p).all() for p in self.calibrator.parameters()):
                raise ValueError("Nonfinite calibrator")
            self.fit_id = artifact["fit_id"]
            self.artifact_hash = sha256_file(path)
        if mode in ("rbf_proj", "mlp"):
            path = cfg.get("selector_artifact_path")
            artifact = read_artifact(path, mode, base_hash)
            kernel_type = artifact["kernel_config"]["kernel_type"]
            if (mode == "rbf_proj" and kernel_type != "rbf_proj") or (
                    mode == "mlp" and kernel_type not in ("mlp_small", "mlp_cap")):
                raise ValueError("Classical artifact kind/architecture mismatch")
            self.selector = kernel_from_config(artifact["kernel_config"]).to(device)
            self.selector.load_state_dict(artifact["state_dict"], strict=True)
            self.selector.set_inc_mode("frozen")
            self.selector.eval()
            if not bool(self.selector.gamma_initialized) or not torch.isfinite(self.selector.gamma):
                raise ValueError("Invalid classical selector bandwidth")
            if mode == "rbf_proj":
                for key, value in pqk.projector.state_dict().items():
                    if not torch.equal(value.cpu(), self.selector.projector.state_dict()[key].cpu()):
                        raise ValueError("R7 must reuse the exact base PQK projector")
            self.artifact_hash = sha256_file(path)
            self.fit_id = artifact["fit_id"]

    @property
    def needs_risk(self):
        return (self.cfg["risk_mode"] != "uniform" and self.cfg["risk_gain"] > 0
                and self.cfg["lambda_cross"] > 0)

    @torch.no_grad()
    def start_task(self, means, old_ids, snapshot_id, means_id):
        if snapshot_id != means_id:
            raise ValueError("Old means/snapshot version mismatch")
        if means.shape[0] != len(old_ids) or not torch.isfinite(means).all():
            raise ValueError("Invalid old means")
        self.means = means.detach().clone().to(self.device)
        self.old_ids = list(old_ids)
        self.snapshot_id = snapshot_id
        self.cache = None
        self.encode_count = 0
        if not torch.equal(self.pqk.gamma.detach(), self.base_gamma):
            raise ValueError("Frozen PQK bandwidth changed")
        if self.needs_risk and len(old_ids):
            metric = self.selector or self.pqk
            if self.cfg["risk_mode"] == "cosine":
                self.cache = F.normalize(self.means, dim=-1)
            else:
                self.cache = metric.encode(self.means)
                if metric is self.pqk:
                    self.encode_count += len(old_ids)

    @torch.no_grad()
    def risk(self, teacher_features):
        if not self.needs_risk:
            return None
        if self.cfg["risk_mode"] == "cosine":
            result = (1 + F.normalize(teacher_features, dim=-1) @ self.cache.T) / 2
        else:
            metric = self.selector or self.pqk
            result = metric.kernel(metric.encode(teacher_features), self.cache)
            if metric is self.pqk:
                self.encode_count += len(teacher_features)
        if not torch.isfinite(result).all() or (result < -1e-6).any() or (result > 1+1e-6).any():
            raise ValueError("Invalid selector output")
        result = result.clamp(0, 1)
        if self.calibrator is not None:
            result = self.calibrator(result)
        return result.detach()

    def losses(self, z, teacher_features, labels, head, projector, seen_ids, new_ids):
        zero = z.sum() * 0
        cross, guard = zero, zero
        active_fraction, weight_mean = 0.0, 1.0
        active_risks = None
        if self.cfg["lambda_cross"] > 0 and self.old_ids:
            scores = head.qbridge_scores(z, seen_ids, self.old_ids)
            violations = margin_violations(scores, labels, seen_ids, self.old_ids, self.cfg["margin_x"])
            active = violations.detach() > 0
            risk = self.risk(teacher_features)
            if risk is not None:
                risk = substitute_active_risk(risk, active, self.cfg["risk_mode"], self.generator)
                active_risks = risk[active]
                if active_risks.numel():
                    weight_mean = float((1 + self.cfg["risk_gain"] * active_risks).mean())
            cross = cross_margin_loss(violations, risk, self.cfg["risk_gain"])
            active_fraction = float(active.float().mean())
        if self.cfg["lambda_guard"] > 0 and self.old_ids and new_ids:
            with torch.no_grad():
                anchors = projector(self.means)
            guard = guard_loss(head, anchors, seen_ids, self.old_ids, new_ids, self.cfg["margin_old"])
        self.last = dict(cross=float(cross.detach()), guard=float(guard.detach()),
                         active_fraction=active_fraction, active_weight_mean=weight_mean,
                         pqk_encoded_samples=self.encode_count)
        if active_risks is not None and active_risks.numel():
            self.last["active_risk_quantiles"] = torch.quantile(
                active_risks.float(), torch.tensor([0., .25, .5, .75, 1.], device=z.device)).tolist()
            self.last["active_risk_histogram_10bins_0to1"] = torch.histc(
                active_risks.float(), bins=10, min=0, max=1).long().tolist()
        # Observe a representative first-batch feature gradient without changing .grad/RNG.
        if getattr(self, "observe_gradient", False) and cross.requires_grad:
            grad = torch.autograd.grad(cross, z, retain_graph=True, allow_unused=True)[0]
            self.last["cross_feature_gradient_norm"] = 0.0 if grad is None else float(grad.norm())
            parameters = getattr(self, "adapter_parameters", ())
            if parameters:
                grads = torch.autograd.grad(cross, parameters, retain_graph=True, allow_unused=True)
                self.last["cross_adapter_gradient_norm"] = math.sqrt(sum(
                    float(g.detach().square().sum()) for g in grads if g is not None))
            if self.cfg["lambda_cross"] > 0 and self.old_ids:
                from utils.qbridge_calibration import binary_metrics
                if risk is not None:
                    self.last["risk_conflict_proxy"] = binary_metrics(risk.flatten(), active.flatten())
                    cosine = ((1 + F.normalize(teacher_features.detach(), dim=-1) @
                               F.normalize(self.means, dim=-1).T)/2).clamp(0, 1)
                    a, b = risk.flatten().double(), cosine.flatten().double()
                    if float(a.std(unbiased=False)) > 0 and float(b.std(unbiased=False)) > 0:
                        self.last["risk_cosine_pearson"] = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                    k = min(5, risk.shape[1])
                    chosen = risk.topk(k, dim=1).indices
                    captured = active.gather(1, chosen).sum().item()
                    total = active.sum().item()
                    self.last["top5_conflict_recall"] = captured/total if total else None
                    self.last["risk_diagnostic_scope"] = "first training batch of epoch; current margin proxy, not future risk"
            self.observe_gradient = False
        return self.cfg["lambda_cross"] * cross + self.cfg["lambda_guard"] * guard

    @torch.no_grad()
    def guard_metrics(self, head, projector, seen_ids, new_ids):
        if not self.old_ids or not new_ids:
            return {"violation_rate": 0., "loss": 0.}
        scores = head.qbridge_scores(projector(self.means), seen_ids, self.old_ids)
        old = id_positions(self.old_ids, seen_ids, scores.device)
        new = id_positions(new_ids, seen_ids, scores.device)
        v = (self.cfg["margin_old"] + scores[:, new].max(1).values
             - scores.gather(1, old[:, None]).squeeze(1)).relu()
        return {"violation_rate": float((v > 0).float().mean()), "loss": float(v.square().mean())}
