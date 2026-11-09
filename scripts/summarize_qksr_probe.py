"""Summarise the QKSR P0 probe runs (exps/qksr2/README.md, section P0).

    python scripts/summarize_qksr_probe.py [--root runs/logs] [--output x.md] [--json x.json]

Reads ``QKSRPROBE grad`` / ``QKSRPROBE end`` lines of the ``p0_*`` logs and applies the fixed reading
rules of the spec. It also checks that the probed runs reproduce the runs they copy (same accuracy curve).
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

GRAD = re.compile(r"QKSRPROBE grad task=(\d+) epoch=(\d+) (\{.*\})")
END = re.compile(r"QKSRPROBE end task=(\d+) (\{.*\})")
CURVE = re.compile(r"CNN top1 curve: \[([^\]]*)\]")
# Runs the probed runs copy, with their recorded CNN curves (same overlay without the probe).
REFERENCE = {
    "p0_qksr_cur": ("qksr_cur rep0 (exps/qksr/results)",
                    "exps/qksr/results/logs/adapter/imagenetr/0/20/qksr_cur_1993_*.log"),
    "p0_rsiat": ("hp_diag W03 (same Kaggle environment as W02/W03)",
                 "exps/holevo_povm/results/kaggle1/logs/adapter/imagenetr/0/20/hp_diag_1993_*.log"),
}
ROOT = Path(__file__).resolve().parents[1]

# Reading rules (spec P0.3), fixed before the runs.
RULES = {
    "conflict": -0.2,       # median cos(c, sep) below this, with >= 60% negative steps: P4 (surgery) worth testing
    "no_conflict": 0.1,     # |median cos(c, sep)| below this: conflict is not the cause, drop P4
    "same_selectivity": 0.02,   # |AUC_pqk - AUC_rbf_theta| below this: PQK selects like an RBF on the angles
    "bottleneck": 0.05,     # AUC_pqk below AUC on 768-d z by more than this: the projection loses class information
    "w1_factor": 5.0,       # between-class share < 5x random share: projector misses class directions
    "saturation": 0.2,      # share of |theta| > 0.9 pi above this: tanh saturates
    "cka_same": 0.9,        # CKA(PQK, RBF on theta) above this: PQK Gram ~ RBF Gram
    "hard_gap": 0.1,        # top-k minus all-pair kernel mean above this: hard pairs exist, uniform loss dilutes them
}


def curve(path):
    found = CURVE.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    return [float(v) for v in found[-1].split(",")] if found else None


def parse(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    grads, ends = [], {}
    for task, epoch, payload in GRAD.findall(text):
        grads.append((int(task), int(epoch), json.loads(payload)))
    for task, payload in END.findall(text):
        ends[int(task)] = json.loads(payload)
    return grads, ends


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(v, d=3):
    return "—" if v is None else ("{:.%df}" % d).format(v)


def summarise(name, path):
    grads, ends = parse(path)
    inc = [g for t, _, g in grads if t > 0]
    base = [g for t, _, g in grads if t == 0]
    out = {"run": name, "log": str(path)}

    def agg(rows, key, stat="median"):
        return med([r[key][stat] for r in rows if key in r])

    out["grad"] = {k: agg(inc, k) for k in ("cos_c_kd", "cos_c_sep", "cos_kd_sep", "cos_c_reg", "pcgrad_keep_sep")}
    out["grad"]["frac_neg_c_sep"] = agg(inc, "cos_c_sep", "frac_neg")
    out["grad"]["frac_neg_c_kd"] = agg(inc, "cos_c_kd", "frac_neg")
    out["grad"]["ratio_sep_c"] = med([r["norm_sep"]["median"] / r["norm_c"]["median"] for r in inc
                                      if "norm_sep" in r and r["norm_c"]["median"] > 0])
    out["grad"]["ratio_kd_c"] = med([r["norm_kd"]["median"] / r["norm_c"]["median"] for r in inc
                                     if "norm_kd" in r and r["norm_c"]["median"] > 0])
    out["grad"]["task0_cos_c_base"] = agg(base, "cos_c_base")
    later = [e for t, e in ends.items() if t > 0]
    pick = lambda f: med([f(e) for e in later if f(e) is not None]) if later else None
    get = lambda *keys: (lambda e: _dig(e, keys))
    out["end"] = {
        "auc_pqk": pick(get("pqk", "auc_class")),
        "auc_rbf_theta": pick(get("twin_euclid_theta", "auc_class")),
        "auc_euclid_z": pick(get("twin_euclid_z", "auc_class")),
        "auc_cosine_z": pick(get("twin_cosine_z", "auc_class")),
        "auc_pqk_cross_task": pick(get("pqk", "auc_cross_task_vs_same_class")),
        "auc_z_cross_task": pick(get("twin_euclid_z", "auc_cross_task_vs_same_class")),
        "kernel_same": pick(get("pqk", "kernel_mean_same")), "kernel_diff": pick(get("pqk", "kernel_mean_diff")),
        "w1_between": pick(get("w1", "between_share")), "w1_within": pick(get("w1", "within_share")),
        "w1_random": pick(get("w1", "random_share")),
        "saturated": pick(get("angles", "frac_saturated_0.9pi")),
        "cka_rbf_theta": pick(get("cka_pqk_vs", "rbf_theta")), "cka_rbf_z": pick(get("cka_pqk_vs", "rbf_z")),
        "pairs_all": pick(get("loss_pairs", "kernel_mean_all")), "pairs_topk": pick(get("loss_pairs", "kernel_mean_topk")),
    }
    if 0 in ends:
        out["end"]["task0"] = {k: _dig(ends[0], k) for k in (("pqk", "auc_class"), ("twin_euclid_theta", "auc_class"),
                                                             ("twin_euclid_z", "auc_class"), ("w1", "between_share"))}
        out["end"]["task0"] = {"/".join(k): v for k, v in out["end"]["task0"].items()}
    ref_name, pattern = REFERENCE.get(name, (None, None))
    own = curve(path)
    out["curve"] = own
    if pattern:
        refs = sorted(glob.glob(str(ROOT / pattern)))
        ref = curve(refs[0]) if refs else None
        out["reference"] = ref_name
        out["reproduces"] = None if ref is None or own is None else own == ref
        out["max_abs_diff"] = None if ref is None or own is None or len(ref) != len(own) else max(
            abs(a - b) for a, b in zip(own, ref))
    return out


def _dig(obj, keys):
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def verdicts(s):
    g, e, r = s["grad"], s["end"], RULES
    out = []
    c = g.get("cos_c_sep")
    if c is not None:
        if c < r["conflict"] and (g.get("frac_neg_c_sep") or 0) >= 0.6:
            out.append("W6: xung đột gradient rõ (median cos(c,sep) {:.3f}) → P4 đáng thử".format(c))
        elif abs(c) < r["no_conflict"]:
            out.append("W6: gần vuông góc (median cos(c,sep) {:.3f}) → xung đột KHÔNG phải nguyên nhân, bỏ P4".format(c))
        else:
            out.append("W6: vùng giữa (median cos(c,sep) {:.3f}) → P4 không ưu tiên".format(c))
    if e.get("auc_pqk") is not None and e.get("auc_rbf_theta") is not None:
        d = e["auc_pqk"] - e["auc_rbf_theta"]
        out.append("W7: AUC PQK − RBF(θ) = {:+.3f} → {}".format(
            d, "PQK chọn lọc như RBF trên góc" if abs(d) < r["same_selectivity"] else
            ("PQK tốt hơn" if d > 0 else "PQK kém hơn")))
    if e.get("auc_pqk") is not None and e.get("auc_euclid_z") is not None:
        d = e["auc_pqk"] - e["auc_euclid_z"]
        out.append("W1: AUC PQK − AUC 768-d = {:+.3f} → {}".format(
            d, "phép chiếu làm mất thông tin lớp" if d < -r["bottleneck"] else "phép chiếu giữ được thông tin lớp"))
    if e.get("w1_between") is not None and e.get("w1_random"):
        f = e["w1_between"] / e["w1_random"]
        out.append("W1: tán xạ giữa lớp trong không gian chiếu = {:.1f}× ngẫu nhiên → {}".format(
            f, "projector bỏ sót hướng phân lớp" if f < r["w1_factor"] else "projector bắt được hướng phân lớp"))
    if e.get("saturated") is not None:
        out.append("W2: góc bão hoà {:.1%} → {}".format(
            e["saturated"], "tanh bão hoà, cần scale" if e["saturated"] > r["saturation"] else "không bão hoà"))
    if e.get("cka_rbf_theta") is not None:
        out.append("W7: CKA(PQK, RBF θ) = {:.3f} → {}".format(
            e["cka_rbf_theta"], "Gram PQK ≈ RBF" if e["cka_rbf_theta"] > r["cka_same"] else "Gram PQK khác RBF"))
    if e.get("pairs_topk") is not None:
        gap = e["pairs_topk"] - e["pairs_all"]
        out.append("W4: kernel top-k − toàn bộ = {:+.3f} → {}".format(
            gap, "có cặp khó, loss trung bình đều làm loãng → P3 đáng thử" if gap > r["hard_gap"]
            else "ít khác biệt giữa cặp khó và cặp thường"))
    return out


def render(summaries):
    lines = ["# QKSR P0: phép đo chỉ đọc (`scripts/summarize_qksr_probe.py`)", ""]
    for s in summaries:
        lines += ["## " + s["run"], "", "Log: `{}`".format(s["log"]), ""]
        if "reference" in s:
            lines.append("- Tái lập {}: {} (lệch tối đa {} điểm)".format(
                s["reference"], {True: "TRÙNG", False: "KHÁC", None: "không có run tham chiếu"}[s["reproduces"]],
                fmt(s.get("max_abs_diff"), 3)))
        g, e = s["grad"], s["end"]
        lines += ["", "| Gradient adapter (task ≥ 1, trung vị) | Giá trị |", "|---|---:|"]
        for k in ("cos_c_kd", "frac_neg_c_kd", "cos_c_sep", "frac_neg_c_sep", "cos_kd_sep", "cos_c_reg",
                  "pcgrad_keep_sep", "ratio_sep_c", "ratio_kd_c", "task0_cos_c_base"):
            lines.append("| {} | {} |".format(k, fmt(g.get(k))))
        lines += ["", "| Cuối task (trung vị task ≥ 1) | Giá trị |", "|---|---:|"]
        for k, v in e.items():
            if k != "task0":
                lines.append("| {} | {} |".format(k, fmt(v)))
        if "task0" in e:
            lines.append("| task 0: " + ", ".join("{} {}".format(k, fmt(v)) for k, v in e["task0"].items()) + " | |")
        lines += ["", "**Đọc theo luật P0.3:**", ""] + ["- " + v for v in verdicts(s)] + [""]
    return "\n".join(lines) + "\n"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/logs")
    parser.add_argument("--output", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    paths = sorted(Path(args.root).rglob("p0_*.log"))
    summaries = []
    for path in paths:
        name = re.match(r"(p0_[a-z_]+?)_\d{4}_", path.name)
        summaries.append(summarise(name.group(1) if name else path.stem, path))
    text = render(summaries) if summaries else "Không tìm thấy log p0_* trong {}\n".format(args.root)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(summaries, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
