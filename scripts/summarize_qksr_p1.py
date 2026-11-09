"""Summarise QKSR-v2 steps P1 and P2 (exps/qksr2/README.md): paired comparisons of the p1_* / p2_* families.

    python scripts/summarize_qksr_p1.py --root runs/logs [--root ...] [--output x.md] [--json x.json]

Paired by replicate (``rng_replicate`` 0-2, ImageNet-R 1993) with ``scripts/run_level_stats.py``. Adds the
old/new halves of the per-task accuracy after the last task, and an environment check of the
``p1_qksr_cur`` runs against the earlier ``qksr_cur`` runs when their logs are reachable.
"""
import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_rls", ROOT / "scripts" / "run_level_stats.py")
rls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rls)

TASK_ACC = re.compile(r"Continual metrics after task \d+: .*task_acc=\[([^\]]*)\]")
QKSR = "p1_qksr_cur"
# (A, B, label): Δ = A − B. P1 rows test the quantum part; P2 rows test the refit (spec P2.3).
COMPARISONS = [
    ("p1_qksr_cur", "p1_qksr_rbf", "P1 · PQK − RBF trên cùng góc θ (T2)"),
    ("p1_qksr_cur", "p1_qksr_nocnot", "P1 · PQK − PQK bỏ CNOT (T4)"),
    ("p2_lda", "p1_qksr_cur", "P2 · LDA + γ − QKSR"),
    ("p2_gamma", "p1_qksr_cur", "P2 · chỉ γ − QKSR"),
    ("p2_lda", "p2_lda_rbf", "P2 · PQK − RBF (cùng LDA + γ)"),
    ("p1_qksr_cur", "p1_rsiat", "Đọc kèm · QKSR − RSIAT (cùng môi trường)"),
    ("p2_lda", "p1_rsiat", "Đọc kèm · P2 LDA + γ − RSIAT"),
]
# p0_rsiat (W04) is replicate 0 of RSIAT in the W04-W06 environment: the probe is read-only and the run
# matched hp_diag of W03 digit for digit, so it stands in for p1_rsiat rep 0.
ALIASES = {"p0_rsiat": "p1_rsiat"}
EQUIV = 0.1  # |Δ AIA| bound of the equivalence reading (spec P1.3)
OLD_LOGS = "exps/qksr/results/logs"


def halves(log):
    found = TASK_ACC.findall(Path(log).read_text(encoding="utf-8", errors="replace"))
    if not found:
        return None
    acc = [float(v) for v in found[-1].split(",")]
    half = len(acc) // 2
    return statistics.mean(acc[:half]), statistics.mean(acc[half:])


def paired(runs, base, cand):
    """Δ = base − cand per replicate for AIA, final, forgetting (run_level_stats) and old/new halves."""
    res = rls.compare(runs, cand, base)  # compare(baseline, candidate) gives candidate − baseline
    index = {(r["family"], r["replicate"]): r for r in runs}
    for name, pos in (("old", 0), ("new", 1)):
        values = []
        for _, _, rep in res["pairs"]:
            a, b = halves(index[(base, rep)]["log"]), halves(index[(cand, rep)]["log"])
            if a and b:
                values.append(a[pos] - b[pos])
        entry = {"mean": statistics.mean(values) if values else None, "n": len(values)}
        if len(values) >= 2:
            half = rls.t975(len(values) - 1) * statistics.stdev(values) / math.sqrt(len(values))
            entry["ci95"] = (entry["mean"] - half, entry["mean"] + half)
        res[name] = entry
    ci = res["aia"].get("ci95")
    if res["n"] < rls.MIN_PAIRS or ci is None:
        res["reading"] = "chưa đủ cặp"
    elif ci[0] > 0:
        res["reading"] = "A HƠN B (CI Δ AIA > 0)"
    elif ci[1] < 0:
        res["reading"] = "A KÉM B (CI Δ AIA < 0)"
    elif -EQUIV <= ci[0] and ci[1] <= EQUIV:
        res["reading"] = "TƯƠNG ĐƯƠNG (CI Δ AIA nằm trong ±{})".format(EQUIV)
    else:
        res["reading"] = "không kết luận (CI chứa 0, rộng hơn ±{})".format(EQUIV)
    return res


def env_check(runs):
    old_root = ROOT / OLD_LOGS
    if not old_root.is_dir():
        return None
    old = {r["replicate"]: r for r in rls.collect([old_root]) if r["family"] == "qksr_cur"}
    out = []
    for r in runs:
        if r["family"] == QKSR and r["replicate"] in old:
            out.append({"replicate": r["replicate"], "aia_new": r["aia"], "aia_old": old[r["replicate"]]["aia"],
                        "identical": abs(r["aia"] - old[r["replicate"]]["aia"]) < 1e-9})
    return out


def fmt_ci(e):
    if e.get("mean") is None:
        return "—"
    ci = e.get("ci95")
    return "{:+.3f}".format(e["mean"]) + (" [{:+.3f}, {:+.3f}]".format(*ci) if ci else "")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = rls.collect(args.root)
    for r in runs:
        r["family"] = ALIASES.get(r["family"], r["family"])
    runs = [r for r in runs if r["family"].startswith(("p1_", "p2_"))]
    lines = ["# QKSR-v2 P1/P2: so sánh ghép cặp (`scripts/summarize_qksr_p1.py`)", "",
             "Δ = A − B, ghép theo replicate; CI 95% t với n − 1 bậc tự do. P2 vượt đường đánh đổi khi "
             "Δ old > 0, Δ new > 0 và CI Δ AIA > 0 so với QKSR (spec P2.3).", "",
             "| Run | Replicate | AIA | Final | Forgetting |", "|---|---:|---:|---:|---:|"]
    for r in sorted(runs, key=lambda r: (r["family"], r["replicate"])):
        lines.append("| {} | {} | {:.3f} | {:.2f} | {} |".format(r["family"], r["replicate"], r["aia"], r["final"],
                                                               rls.fmt(r["forgetting"], 2)))
    results = {}
    lines += ["", "| So sánh | n | Δ AIA | Δ final | Δ forgetting | Δ old | Δ new | Đọc |",
              "|---|---:|---|---|---|---|---|---|"]
    for a, b, label in COMPARISONS:
        if not any(r["family"] == a for r in runs) or not any(r["family"] == b for r in runs):
            continue
        res = paired(runs, a, b)
        if a.startswith("p2_") and b == QKSR and res["n"] >= rls.MIN_PAIRS:
            ci = res["aia"].get("ci95")
            crosses = (res["old"]["mean"] or 0) > 0 and (res["new"]["mean"] or 0) > 0 and ci and ci[0] > 0
            res["reading"] += "; vượt đường đánh đổi: " + ("CÓ" if crosses else "KHÔNG")
        results["{} - {}".format(a, b)] = res
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            label, res["n"], fmt_ci(res["aia"]), fmt_ci(res["final"]), fmt_ci(res["forgetting"]),
            fmt_ci(res["old"]), fmt_ci(res["new"]), res["reading"]))
    env = env_check(runs)
    if env is not None:
        lines += ["", "**Môi trường:** `p1_qksr_cur` so với `qksr_cur` cũ: " + ", ".join(
            "rep {} {} ({:.3f} / {:.3f})".format(e["replicate"], "TRÙNG" if e["identical"] else "KHÁC",
                                                 e["aia_new"], e["aia_old"]) for e in env)]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": runs, "comparisons": results, "environment": env},
                                              indent=1, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
