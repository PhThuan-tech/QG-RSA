"""Holevo-CIL and Holevo-POVM: training-time quantum-information objectives for EFCIL.

Pre-registered specs:
  exps/holevo/README.md         A: the adapter maximises the Holevo chi of every class seen
                                   (old classes as states built from the stored mean and covariance)
  exps/holevo_povm/README.md    A+B: plus a residual-POVM decoder loss (old classes: delta-regularised
                                   PGM, new classes measured inside the soft residual R = I - sum E_old)

Encoding (spec A 2.1): v(z) = [(z - m0)/s0 ; 1] (D = 769, not normalised); m0, s0 are fixed at the
end of task 0 from the stored statistics of the task-0 classes. Every state has trace one and is the
||v||^2-weighted mixture of the pure states v/||v||, so states of old classes come from the stored
mean and covariance (exemplar-free) and states of new classes from a per-class FIFO queue of the
current task's features (only the current batch carries gradient).

Entropies use a linear surrogate Tr[sg(f'(rho)) rho]: the value is exact, the gradient equals the
gradient of Tr f(rho), and no derivative goes through eigenvectors (degenerate spectra are safe).

Diagnostics that read old-class images (chi_oracle, I_orc) run under preserve_rng with a private
loader and never feed back into training. With holevo_mode / hp_mode "off" (the default) nothing
here runs and RSIAT is unchanged.
"""
import json
import logging
import math
import statistics

import numpy as np
import torch
from torch.nn import functional as F

# mode -> (encoder objective, decoder loss)
HOLEVO_MODES = {"vn": ("vn", None), "mcr2": ("mcr2", None), "spread": ("spread", None)}
HP_MODES = {"full": ("vn", "povm"), "povm": (None, "povm"), "twin": ("mcr2", "gauss"),
            "twin_povm": (None, "gauss"), "diag": ("vn", "povm")}
# Pre-registered settings (exps/holevo/README.md 3.1, exps/holevo_povm/README.md 3.2).
DEFAULTS = {
    "holevo_lambda": "auto",      # "auto": gradient-ratio calibration at task 1 (spec A 3.3); or a fixed number
    "holevo_grad_ratio": 0.5,     # r* of spec A
    "holevo_calib_steps": 20,
    "holevo_queue_cap": 256,      # per class; the queue holds min(cap, images of the class)
    "holevo_eps": 1e-8,
    "holevo_mcr2_eps2": 0.5,
    "holevo_oracle_diag": False,
    "holevo_log_every": 0,
    "hp_tau": 1.0,                # delta = tau / D; eta = tau * tr(mean old cov) / 768
    "hp_eps_omega": 1e-6,         # relative to tr(Omega_0) / D
    "hp_grad_ratio": 0.5,         # total r*, split equally over the active objectives
}
LAMBDA_RANGE = (1e-3, 1e3)
COV_RIDGE = 1e-3     # models/base.py::_compute_class_mean adds 1e-3 I to every stored covariance
ORACLE_SEED = 20261030
CHUNK = 256


def holevo_settings(args):
    a_mode, b_mode = args.get("holevo_mode", "off"), args.get("hp_mode", "off")
    if a_mode not in ("off",) + tuple(HOLEVO_MODES):
        raise ValueError("holevo_mode must be one of off, {}.".format(", ".join(HOLEVO_MODES)))
    if b_mode not in ("off",) + tuple(HP_MODES):
        raise ValueError("hp_mode must be one of off, {}.".format(", ".join(HP_MODES)))
    if a_mode != "off" and b_mode != "off":
        raise ValueError("Set holevo_mode or hp_mode, not both.")
    if a_mode == "off" and b_mode == "off":
        return None
    cfg = {key: args.get(key, default) for key, default in DEFAULTS.items()}
    lam = cfg["holevo_lambda"]
    if lam != "auto" and (isinstance(lam, bool) or not isinstance(lam, (int, float))
                          or not math.isfinite(lam) or lam <= 0):
        raise ValueError("holevo_lambda must be 'auto' or a positive number.")
    for key in ("holevo_grad_ratio", "holevo_eps", "holevo_mcr2_eps2", "hp_tau", "hp_eps_omega", "hp_grad_ratio"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not cfg[key] > 0:
            raise ValueError("{} must be a positive number.".format(key))
    for key, low in (("holevo_calib_steps", 1), ("holevo_queue_cap", 2), ("holevo_log_every", 0)):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < low:
            raise ValueError("{} must be an integer >= {}.".format(key, low))
    if not isinstance(cfg["holevo_oracle_diag"], bool):
        raise ValueError("holevo_oracle_diag must be true or false.")
    if args.get("resume"):
        raise ValueError("holevo_mode / hp_mode runs cannot resume: the queue and lambdas are not checkpointed.")
    family = "holevo" if a_mode != "off" else "hp"
    enc, dec = HOLEVO_MODES[a_mode] if family == "holevo" else HP_MODES[b_mode]
    terms = [name for name, kind in (("enc", enc), ("dec", dec)) if kind]
    ratio = cfg["holevo_grad_ratio"] if family == "holevo" else cfg["hp_grad_ratio"]
    cfg.update(family=family, mode=a_mode if family == "holevo" else b_mode, enc=enc, dec=dec,
               diag=b_mode == "diag", target={name: ratio / len(terms) for name in terms},
               objective=enc or "vn")  # the objective rule G reads (spec A 4.1)
    return cfg


# ----------------------------------------------------------------------------- spectral functions
def spectral_trace(mat, kind, eps=1e-8, rate=None):
    """Tr f(mat) of a symmetric PSD matrix: exact value, exact gradient through a linear surrogate.

    kind "vn": f(x) = -x log x (von Neumann entropy for trace-one mat); kind "rate": f(x) = log(1 + rate x)/2
    (MCR2 coding rate). Both have f(0) = 0, so Tr f(F F^T) = Tr f(F^T F) (use the smaller Gram matrix).
    """
    with torch.no_grad():
        evals, evecs = torch.linalg.eigh(mat.detach())
        lam = evals.clamp_min(0.0)
        if kind == "vn":
            value = -(lam * torch.log(torch.where(lam > 0, lam, torch.ones_like(lam)))).sum()
            deriv = -(torch.log(lam + eps) + 1.0)
        elif kind == "rate":
            value = 0.5 * torch.log1p(rate * lam).sum()
            deriv = 0.5 * rate / (1.0 + rate * lam)
        else:
            raise ValueError(kind)
        grad = (evecs * deriv) @ evecs.T
    if not mat.requires_grad:
        return value
    surrogate = (grad * mat).sum()
    return value + surrogate - surrogate.detach()


def inv_sqrt(mat):
    evals, evecs = torch.linalg.eigh(mat)
    return (evecs * evals.clamp_min(1e-300).rsqrt()) @ evecs.T


def _norm(grads):
    return math.sqrt(sum(float(g.detach().square().sum()) for g in grads if g is not None))


# ----------------------------------------------------------------------------- encoding and states
class StatsEncoder:
    """v(z) = [(z - m0)/s0 ; 1], fixed after task 0 from the stored task-0 statistics (spec A 2.1)."""

    def __init__(self, center, scale):
        self.center, self.scale = center.double(), float(scale)

    @classmethod
    def from_stats(cls, means, covs):
        means, covs = means.double(), covs.double()
        center = means.mean(0)
        spread = covs.diagonal(dim1=1, dim2=2).sum(1) + (means - center).square().sum(1)
        return cls(center, math.sqrt(float(spread.mean())))

    @property
    def dim(self):
        return self.center.shape[0] + 1

    def lift(self, z):
        z = z.double()
        return torch.cat([(z - self.center) / self.scale, torch.ones_like(z[:, :1])], 1)

    def factor(self, z):
        """D x n factor F with rho = F F^T = sum v v^T / sum ||v||^2."""
        v = self.lift(z)
        return v.T / v.square().sum().sqrt()

    def pure(self, z):
        v = self.lift(z)
        return v / v.norm(dim=1, keepdim=True)

    def stats_state(self, mean, cov):
        """rho from one mean and covariance, as utils/qsd.py::stats_states with this encoder."""
        mu = (mean.double() - self.center) / self.scale
        dim = mu.shape[0]
        rho = torch.empty(dim + 1, dim + 1, dtype=torch.float64, device=mu.device)
        rho[:dim, :dim] = cov.double() / self.scale ** 2 + mu[:, None] * mu[None, :]
        rho[:dim, dim] = mu
        rho[dim, :dim] = mu
        rho[dim, dim] = 1.0
        return rho / torch.trace(rho)


class FeatureQueue:
    """Per-class FIFO of the current task's features (spec A 2.2).

    Capacity per class = min(cap, images of the class in the initial pass). ``view`` replaces the
    oldest entries by the current batch (with gradient) without writing; ``commit`` writes the
    detached batch at the same positions.
    """

    def __init__(self, feats, labels, classes, cap, device):
        self.classes, self.buf, self.ptr = list(classes), {}, {}
        feats, labels = torch.as_tensor(feats), torch.as_tensor(labels)
        for c in self.classes:
            rows = feats[labels == c]
            if rows.shape[0] < 2:
                raise ValueError("Holevo queue: class {} has {} feature(s), need 2.".format(c, rows.shape[0]))
            self.buf[c] = rows[-cap:].to(device=device, dtype=torch.float32).clone()
            self.ptr[c] = 0

    def _slots(self, c, count):
        size = self.buf[c].shape[0]
        return (self.ptr[c] + torch.arange(min(count, size), device=self.buf[c].device)) % size

    def view(self, z, y):
        rows = {}
        for c in self.classes:
            q = self.buf[c]
            batch = z[y == c][-q.shape[0]:]
            if batch.shape[0] == 0:
                rows[c] = q
                continue
            keep = torch.ones(q.shape[0], dtype=torch.bool, device=q.device)
            keep[self._slots(c, batch.shape[0])] = False
            rows[c] = torch.cat([q[keep].to(batch.dtype), batch])
        return rows

    def commit(self, z, y):
        z = z.detach()
        for c in self.classes:
            batch = z[y == c][-self.buf[c].shape[0]:]
            if batch.shape[0]:
                self.buf[c][self._slots(c, batch.shape[0])] = batch.float()
                self.ptr[c] = (self.ptr[c] + batch.shape[0]) % self.buf[c].shape[0]


# ----------------------------------------------------------------------------- runtime
class HolevoRuntime:
    def __init__(self, cfg, warmup_epoch, device):
        self.cfg, self.warmup, self.device = cfg, int(warmup_epoch), device
        self.encoder, self.task, self.old, self.queue = None, None, None, None
        fixed = None if cfg["holevo_lambda"] == "auto" else float(cfg["holevo_lambda"])
        self.lambdas = {name: fixed for name in cfg["target"]}
        self.calib = {name: [] for name in cfg["target"]}
        self.calib_count, self.step = 0, 0
        self.epoch_stats = {}
        self.oracle_log = {}

    # ---------------------------------------------------------------- task boundaries
    def fix_encoder(self, learner):
        total = learner._total_classes
        means = torch.as_tensor(np.asarray(learner._class_means[:total]), device=self.device)
        covs = torch.as_tensor(learner._class_covs[:total]).to(self.device)
        self.encoder = StatsEncoder.from_stats(means, covs)
        logging.info("HOLEVO encoder classes=%d D=%d scale=%.6f", total, self.encoder.dim, self.encoder.scale)

    def start_task(self, learner, feats, labels):
        if self.encoder is None:
            raise RuntimeError("Holevo: the encoder is fixed at the end of task 0.")
        cfg, enc, dev = self.cfg, self.encoder, self.device
        known, total = learner._known_classes, learner._total_classes
        self.task = learner._cur_task
        means = torch.as_tensor(np.asarray(learner._class_means[:known]), dtype=torch.float64, device=dev)
        covs = torch.as_tensor(learner._class_covs[:known]).to(device=dev, dtype=torch.float64)
        rate = enc.dim / cfg["holevo_mcr2_eps2"]
        rho_bar = torch.zeros(enc.dim, enc.dim, dtype=torch.float64, device=dev)
        s_old = r_old = 0.0
        for c in range(known):  # stale states of the old classes: constants for the whole task
            rho = enc.stats_state(means[c], covs[c])
            rho_bar += rho
            s_old += float(spectral_trace(rho, "vn", cfg["holevo_eps"]))
            r_old += float(spectral_trace(rho, "rate", rate=rate))
        rho_bar /= known
        self.old = {"K": known, "C": total, "rho_bar": rho_bar, "S_sum": s_old, "R_sum": r_old, "rate": rate}
        if cfg["dec"] == "povm":
            delta = cfg["hp_tau"] / enc.dim
            evals, evecs = torch.linalg.eigh(rho_bar + delta * torch.eye(enc.dim, dtype=torch.float64, device=dev))
            r_diag = delta / evals
            self.old.update(delta=delta, R_half=(evecs * r_diag.sqrt()) @ evecs.T,
                            P_old=torch.eye(enc.dim, dtype=torch.float64, device=dev) - (evecs * r_diag) @ evecs.T,
                            residual_share=float((r_diag > 0.5).double().mean()))
        if cfg["dec"] == "gauss":
            eta = cfg["hp_tau"] * float(covs.diagonal(dim1=1, dim2=2).sum(1).mean()) / covs.shape[1]
            eye = torch.eye(covs.shape[1], dtype=torch.float64, device=dev)
            chol = torch.linalg.cholesky(covs + eta * eye)
            self.old.update(eta=eta, means=means, chol=chol,
                            logdet=2 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum(1))
        self.queue = FeatureQueue(feats, labels, range(known, total), cfg["holevo_queue_cap"], dev)
        logging.info("HOLEVO start task=%d old=%d new=%d %s", self.task, known, total - known, json.dumps(
            {"queue": {str(c): int(q.shape[0]) for c, q in self.queue.buf.items()},
             "residual_share": self.old.get("residual_share"), "eta": self.old.get("eta")}))
        if cfg["holevo_oracle_diag"]:
            self.oracle(learner, "start")

    def end_epoch(self, task, epoch):
        if task == 1 and epoch == 0 and not self.cfg["diag"] and self._pending():
            self._finalize_lambdas()
        stats = self.epoch_stats
        if stats:
            logging.info("HOLEVO epoch task=%d epoch=%d %s", task, epoch + 1, json.dumps(
                {k: statistics.mean(v) for k, v in stats.items() if v}))
        self.epoch_stats = {}

    def end_task(self, learner):
        if self.cfg["holevo_oracle_diag"]:
            self.oracle(learner, "end")
        logging.info("HOLEVO end task=%d lambdas=%s", self.task, json.dumps(self.lambdas))
        self.old, self.queue = None, None

    # ---------------------------------------------------------------- objectives
    def encoder_objective(self, factors, kind):
        """Normalised chi (or coding-rate reduction, or S(rho_bar)) of old + new classes: higher is better."""
        old, eps = self.old, self.cfg["holevo_eps"]
        known, total = old["K"], old["C"]
        stacked = torch.cat(list(factors.values()), 1)
        rho_bar = (known / total) * old["rho_bar"] + stacked @ stacked.T / total
        log_c = math.log(total)
        if kind in ("vn", "spread"):
            s_bar = spectral_trace(rho_bar, "vn", eps)
            if kind == "spread":
                return s_bar / log_c
            s_new = sum(spectral_trace(f.T @ f, "vn", eps) for f in factors.values())
            return (s_bar - (s_new + old["S_sum"]) / total) / log_c
        r_bar = spectral_trace(rho_bar, "rate", rate=old["rate"])
        r_new = sum(spectral_trace(f.T @ f, "rate", rate=old["rate"]) for f in factors.values())
        return (r_bar - (r_new + old["R_sum"]) / total) / log_c

    def povm(self, psi, factors):
        """Residual-POVM probabilities (spec B 2.2-2.4): p_new (n x new), p_old, p_null; rows sum to one."""
        old, cfg = self.old, self.cfg
        dim = psi.shape[1]
        tilted = [old["R_half"] @ f for f in factors.values()]
        owner = torch.cat([torch.full((f.shape[1],), j, dtype=torch.long, device=psi.device)
                           for j, f in enumerate(tilted)])
        stacked = torch.cat(tilted, 1)
        q = 1.0 / len(tilted)
        omega0 = q * (stacked @ stacked.T)
        eps = cfg["hp_eps_omega"] * float(torch.trace(omega0.detach())) / dim
        with torch.no_grad():  # detached whitening; completeness still exact (same Omega)
            whiten = inv_sqrt(omega0.detach() + eps * torch.eye(dim, dtype=torch.float64, device=psi.device))
        u = psi @ old["R_half"] @ whiten
        proj = (u @ stacked).square()
        p_new = q * torch.zeros(psi.shape[0], len(tilted), dtype=proj.dtype, device=psi.device).index_add(
            1, owner, proj)
        p_old = ((psi @ old["P_old"]) * psi).sum(1)
        p_null = eps * u.square().sum(1)
        return p_new, p_old, p_null

    def gauss(self, z, rows):
        """Classical twin of the decoder: log-likelihood logits of frozen old and queued new Gaussians."""
        old = self.old
        z = z.double()
        eye = torch.eye(z.shape[1], dtype=torch.float64, device=z.device)
        cols = []
        for start in range(0, z.shape[0], CHUNK):  # chunked: K x n x 768 in float64
            part = z[start:start + CHUNK]
            diff = (part[None] - old["means"][:, None]).transpose(1, 2)
            sol = torch.linalg.solve_triangular(old["chol"], diff, upper=False)
            cols.append(-0.5 * (sol.square().sum(1) + old["logdet"][:, None]).T)
        logits_old = torch.cat(cols)
        new = []
        for c in self.queue.classes:
            r = rows[c].double()
            mu = r.mean(0)
            cov = (r - mu).T @ (r - mu) / (r.shape[0] - 1) + (COV_RIDGE + old["eta"]) * eye
            chol = torch.linalg.cholesky(cov)
            sol = torch.linalg.solve_triangular(chol, (z - mu).T, upper=False)
            new.append(-0.5 * (sol.square().sum(0) + 2 * torch.log(chol.diagonal()).sum()))
        return torch.cat([logits_old, torch.stack(new, 1)], 1)

    def compute(self, z, y):
        """Objective terms (minimised) and logged quantities for one batch."""
        cfg, old = self.cfg, self.old
        rows = self.queue.view(z, y)
        factors = {c: self.encoder.factor(rows[c]) for c in self.queue.classes}
        terms, info = {}, {}
        if cfg["enc"]:
            obj = self.encoder_objective(factors, cfg["enc"])
            terms["enc"] = -obj
            info["objective"] = float(obj.detach())
        if cfg["dec"] == "povm":
            p_new, p_old, p_null = self.povm(self.encoder.pure(z), factors)
            p_y = p_new.gather(1, (y - old["K"])[:, None]).squeeze(1)
            terms["dec"] = -torch.log(p_y + 1e-12).mean()
            with torch.no_grad():
                info.update(p_y=float(p_y.mean()), p_old=float(p_old.mean()), p_null=float(p_null.mean()),
                            completeness_error=float((p_new.sum(1) + p_old + p_null - 1).abs().max()))
        elif cfg["dec"] == "gauss":
            logits = self.gauss(z, rows)
            terms["dec"] = F.cross_entropy(logits, y)
            with torch.no_grad():
                mass = torch.logsumexp(logits[:, :old["K"]], 1) - torch.logsumexp(logits, 1)
                info.update(p_old=float(mass.exp().mean()), p_y=float(torch.softmax(logits, 1).gather(
                    1, y[:, None]).mean()))
        for name, value in terms.items():
            info["loss_" + name] = float(value.detach())
        return terms, info

    # ---------------------------------------------------------------- training step
    def _pending(self):
        return any(v is None for v in self.lambdas.values())

    def _params(self, learner):
        return [p for p in learner._network_module_ptr.convnet.parameters() if p.requires_grad]

    def _grad_ratios(self, learner, loss_c, terms, scale):
        params = self._params(learner)
        g_c = _norm(torch.autograd.grad(loss_c, params, retain_graph=True, allow_unused=True))
        out = {}
        for name, term in terms.items():
            g = _norm(torch.autograd.grad(term, params, retain_graph=True, allow_unused=True))
            out[name] = (scale.get(name, 1.0) * g) / max(g_c, 1e-30)
        return out, g_c

    def _finalize_lambdas(self):
        record = {}
        for name, ratios in self.calib.items():
            if self.lambdas[name] is not None:
                continue
            if not ratios:
                raise RuntimeError("Holevo: no calibration step recorded for '{}'.".format(name))
            raw = self.cfg["target"][name] * statistics.median(ratios)
            lam = min(max(raw, LAMBDA_RANGE[0]), LAMBDA_RANGE[1])
            self.lambdas[name] = lam
            record[name] = {"lambda": lam, "raw": raw, "clamped": lam != raw, "steps": len(ratios),
                            "target": self.cfg["target"][name], "median_gc_over_go": statistics.median(ratios)}
            if lam != raw:
                logging.warning("HOLEVO lambda for %s clamped from %.4g to %.4g", name, raw, lam)
        logging.info("HOLEVO lambda %s", json.dumps(record))

    def step_loss(self, learner, features, targets, loss_c, epoch):
        """Auxiliary loss to add (None when it must not change the update: ramp weight 0 or diag)."""
        cfg = self.cfg
        weight = min(1.0, epoch / self.warmup) if self.warmup > 0 else 1.0
        active = weight > 0 and not cfg["diag"]
        calibrating = not cfg["diag"] and self.task == 1 and self._pending()
        if active and self._pending():
            if self.task == 1 and any(self.calib.values()):
                self._finalize_lambdas()
            else:
                raise RuntimeError("Holevo: lambdas are calibrated in epoch 0 of task 1.")
        log_now = cfg["holevo_log_every"] > 0 and self.step % cfg["holevo_log_every"] == 0
        total, info = None, None
        if active or calibrating:
            terms, info = self.compute(features, targets)
            if calibrating:
                params = self._params(learner)
                g_c = _norm(torch.autograd.grad(loss_c, params, retain_graph=True, allow_unused=True))
                for name, term in terms.items():
                    g = _norm(torch.autograd.grad(term, params, retain_graph=True, allow_unused=True))
                    self.calib[name].append(g_c / max(g, 1e-30))
                self.calib_count += 1
                if self.calib_count >= cfg["holevo_calib_steps"]:
                    self._finalize_lambdas()
            if active:
                total = sum(self.lambdas[name] * weight * term for name, term in terms.items())
                if log_now:
                    ratios, _ = self._grad_ratios(learner, loss_c, terms,
                                                  {n: self.lambdas[n] * weight for n in terms})
                    info.update({"r_" + n: v for n, v in ratios.items()})
        elif log_now:
            with torch.no_grad():
                _, info = self.compute(features.detach(), targets)
        if info is not None and (log_now or active):
            info["weight"] = weight
            for key, value in info.items():
                self.epoch_stats.setdefault(key, []).append(value)
        self.queue.commit(features, targets)
        self.step += 1
        return total

    # ---------------------------------------------------------------- diagnostics (read old images)
    def _extract(self, learner, classes, seed):
        from utils.stats_diagnostics import deterministic_loader
        dataset = learner.data_manager.get_dataset(classes, source="train", mode="test")
        loader = deterministic_loader(dataset, 128, int(learner.args.get("stats_num_workers", 2)), False, seed)
        network = learner._network.module if hasattr(learner._network, "module") else learner._network
        feats, labels = [], []
        for _, inputs, targets in loader:
            feats.append(network.extract_vector(inputs.to(self.device, non_blocking=True)).float())
            labels.append(torch.as_tensor(targets, device=self.device))
        return torch.cat(feats), torch.cat(labels)

    def oracle(self, learner, phase):
        """chi from the real features of every class seen (spec A 4.3) and, with a decoder, I_orc (spec B 4.3)."""
        from utils.qbridge_experiment import preserve_rng
        old, enc, eps = self.old, self.encoder, self.cfg["holevo_eps"]
        known, total = old["K"], old["C"]
        with preserve_rng(), torch.no_grad():
            modes = [(m, m.training) for m in learner._network.modules()]
            learner._network.eval()
            try:
                z, y = self._extract(learner, np.arange(0, total),
                                     ORACLE_SEED + 100 * self.task + (phase == "end"))
            finally:
                for m, training in modes:
                    m.training = training
            factors = [enc.factor(z[y == c]) for c in range(total)]
            stacked = torch.cat(factors, 1)
            rho_bar = stacked @ stacked.T / total
            log_c = math.log(total)
            s_bar = float(spectral_trace(rho_bar, "vn", eps))
            s_cls = statistics.mean(float(spectral_trace(f.T @ f, "vn", eps)) for f in factors)
            r_bar = float(spectral_trace(rho_bar, "rate", rate=old["rate"]))
            r_cls = statistics.mean(float(spectral_trace(f.T @ f, "rate", rate=old["rate"])) for f in factors)
            report = {"oracle": {"vn": (s_bar - s_cls) / log_c, "mcr2": (r_bar - r_cls) / log_c,
                                 "spread": s_bar / log_c}}
            queued = {c: enc.factor(self.queue.buf[c]) for c in self.queue.classes}
            report["stale"] = {kind: float(self.encoder_objective(queued, kind)) for kind in ("vn", "mcr2", "spread")}
            old_rows, new_rows = z[y < known], z[y >= known]
            if self.cfg["dec"] == "povm":
                intr, mass = [], []
                for start in range(0, old_rows.shape[0], 4096):
                    p_new, _, _ = self.povm(enc.pure(old_rows[start:start + 4096]), queued)
                    intr.append(p_new.sum(1))
                for start in range(0, new_rows.shape[0], 4096):
                    _, p_old, _ = self.povm(enc.pure(new_rows[start:start + 4096]), queued)
                    mass.append(p_old)
                report["I_orc"] = float(torch.cat(intr).mean())
                report["p_old_on_new"] = float(torch.cat(mass).mean())
            elif self.cfg["dec"] == "gauss":
                rows = {c: self.queue.buf[c] for c in self.queue.classes}
                intr = []
                for start in range(0, old_rows.shape[0], CHUNK):
                    probs = torch.softmax(self.gauss(old_rows[start:start + CHUNK], rows), 1)
                    intr.append(probs[:, known:].sum(1))
                report["I_orc"] = float(torch.cat(intr).mean())
            if phase == "end":  # stats states are ||v||^2-weighted: distance to the true mixture (spec B 2.7)
                dists = []
                for c in range(known, total):
                    rows_c = z[y == c]
                    f = enc.factor(rows_c)
                    psi = enc.pure(rows_c)
                    diff = f @ f.T - psi.T @ psi / psi.shape[0]
                    dists.append(0.5 * float(torch.linalg.eigvalsh(diff).abs().sum()))
                report["trace_distance"] = {"mean": statistics.mean(dists), "max": max(dists)}
        self.oracle_log.setdefault(self.task, {})[phase] = report
        logging.info("HOLEVO oracle task=%d phase=%s %s", self.task, phase, json.dumps(report))
        return report
