"""Summarise statistics-classifier fusion (SCF) on the unified 3-seed protocol (round A3).

Collects every run of family ``scf`` (CIFAR-100, CUB-200) and replicate 0 of family
``qsd2`` (ImageNet-R/A, round A2) for seeds 1993, 2015 and 2026, and reports RSIAT and
SCF with the fusion weights frozen in round A2. Decision rules are pre-registered in
exps/scf/README.md and fixed below.

  python scripts/summarize_scf.py --root <logs> [--root <logs> ...] [--output summary.md] [--json summary.json]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_level_stats import RUN_NAME, split_family  # noqa: E402
from scripts.summarize_ca_heads import fmt, fmt_ci, interval  # noqa: E402

# Pre-registered (exps/scf/README.md). Do not edit after seeing results.
SEEDS = (1993, 2015, 2026)
FAMILIES = ("scf", "qsd2")
NEW_DATASETS = ("cifar224", "cub")
METHODS = (("linear", "RSIAT"), ("fuse_ridge_stats_b0p25", "SCF ridge"), ("fuse_pgm_stats_b0p5", "SCF PGM"),
           ("fuse_pgm_stats_hier_b1", "SCF PGM hier"))
CANDIDATES = ("fuse_ridge_stats_b0p25", "fuse_pgm_stats_b0p5")
STANDALONE = (("ridge_stats", "ridge alone"), ("pgm_stats", "PGM alone"))
ORDER = ("imagenetr", "imageneta", "cifar224", "cub")
NAMES = {"imagenetr": "ImageNet-R", "imageneta": "ImageNet-A", "cifar224": "CIFAR-100", "cub": "CUB-200"}

CURVE = re.compile(r"QSD curve head=(\w+): (\[.*\])")
TASKS = re.compile(r"QSD tasks head=(\w+): (\[.*\])")
CNN = re.compile(r"CNN top1 curve: (\[.*\])")


def continual(rows):
    """Forgetting and BWT exactly as trainer.py (rows: per-task accuracies after every stage)."""
    final, old = rows[-1], range(len(rows[-1]) - 1)
    if not old:
        return None, None
    forgetting = statistics.mean(max(row[j] for row in rows[j:-1]) - final[j] for j in old)
    bwt = statistics.mean(final[j] - rows[j][j] for j in old)
    return forgetting, bwt


def parse(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    family, replicate = split_family(match.group("prefix"))
    seed = int(match.group("seed"))
    if family not in FAMILIES or replicate != 0 or seed not in SEEDS:
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    curves = {h: json.loads(v) for h, v in CURVE.findall(text)}
    rows = {}
    for head, values in TASKS.findall(text):
        rows.setdefault(head, []).append(json.loads(values))
    cnn = [json.loads(v) for v in CNN.findall(text)]
    if not curves or "linear" not in curves:
        return None
    heads = {}
    for head, curve in curves.items():
        forgetting, bwt = continual(rows[head]) if head in rows else (None, None)
        heads[head] = {"aia": statistics.mean(curve), "final": curve[-1], "forgetting": forgetting, "bwt": bwt,
                       "curve": curve}
    return {"dataset": log.parent.parent.parent.name, "family": family, "seed": seed, "log": str(log),
            "tasks": len(curves["linear"]), "cnn_curve": cnn[-1] if cnn else None, "heads": heads}


def collect(roots):
    runs = {}
    for root in roots:
        for log in sorted(Path(root).rglob("*.log")):
            run = parse(log)
            if run:  # one run per (dataset, seed); keep the longest (completed) one
                key = (run["dataset"], run["seed"])
                if key not in runs or run["tasks"] > runs[key]["tasks"]:
                    runs[key] = run
    longest = {}
    for (ds, _), run in runs.items():
        longest[ds] = max(longest.get(ds, 0), run["tasks"])
    return [r for r in runs.values() if r["tasks"] == longest[r["dataset"]]]


def deltas(group, head, rival="linear", metric="aia"):
    return interval([r["heads"][head][metric] - r["heads"][rival][metric] for r in group
                     if head in r["heads"] and r["heads"][head][metric] is not None])


def decide(runs):
    out = {"transfer": {}, "quantum": None}
    new = [r for r in runs if r["dataset"] in NEW_DATASETS]
    for head in CANDIDATES:
        per = {ds: deltas([r for r in new if r["dataset"] == ds], head) for ds in NEW_DATASETS}
        pooled = deltas(new, head)
        ok = (pooled["n"] >= 2 * len(SEEDS) and "ci95" in pooled and pooled["ci95"][0] > 0
              and all(e["n"] == len(SEEDS) and e["mean"] > 0 for e in per.values()))
        out["transfer"][head] = {"per_dataset": per, "pooled": pooled, "holds": ok}
    q = deltas(runs, "fuse_pgm_stats_b0p5", "fuse_ridge_stats_b0p25")
    out["quantum"] = dict(q, holds="ci95" in q and q["ci95"][0] > 0)
    return out


def render(runs):
    datasets = [d for d in ORDER if any(r["dataset"] == d for r in runs)]
    lines = ["# SCF trên giao thức 3 seed (1993, 2015, 2026)", "",
             "Trọng số fusion khoá từ vòng A2: ridge 0.25, PGM 0.5, PGM phân cấp 1.", "",
             "## Bảng chính (mean ± SD trên seed)", "",
             "| Dataset | n | Phương pháp | AIA | Final | Forgetting ↓ |", "|---|---:|---|---:|---:|---:|"]
    for ds in datasets:
        group = [r for r in runs if r["dataset"] == ds]
        for head, label in METHODS + STANDALONE:
            if not all(head in r["heads"] for r in group):
                continue
            cells = []
            for metric in ("aia", "final", "forgetting"):
                v = [r["heads"][head][metric] for r in group if r["heads"][head][metric] is not None]
                cells.append("{:.2f} ± {:.2f}".format(statistics.mean(v), statistics.stdev(v)) if len(v) > 1
                             else (fmt(v[0], 2) if v else ""))
            lines.append("| {} | {} | {} | {} |".format(NAMES.get(ds, ds), len(group), label, " | ".join(cells)))
    lines += ["", "## Chênh lệch ghép cặp so với RSIAT (cùng run), CI 95%", "",
              "| Dataset | Phương pháp | Δ AIA | Δ Final | Δ Forgetting |", "|---|---|---|---|---|"]
    for ds in datasets:
        group = [r for r in runs if r["dataset"] == ds]
        for head, label in METHODS[1:]:
            if all(head in r["heads"] for r in group):
                cells = ["{} {}".format(fmt(e["mean"]), fmt_ci(e)) for e in
                         (deltas(group, head, metric=m) for m in ("aia", "final", "forgetting"))]
                lines.append("| {} | {} | {} |".format(NAMES.get(ds, ds), label, " | ".join(cells)))
    d = decide(runs)
    lines += ["", "## Quyết định (quy tắc đặt trước)", ""]
    for head, t in d["transfer"].items():
        per = ", ".join("{} {} (n={})".format(NAMES[ds], fmt(e["mean"]), e["n"]) for ds, e in t["per_dataset"].items())
        lines.append("- **Chuyển giao sang dataset mới** `{}`: gộp {} {} (n={}); {} → **{}**".format(
            head, fmt(t["pooled"]["mean"]), fmt_ci(t["pooled"]), t["pooled"]["n"], per,
            "ĐẠT" if t["holds"] else ("chưa đủ run" if t["pooled"]["n"] < 2 * len(SEEDS) else "KHÔNG ĐẠT")))
    q = d["quantum"]
    lines.append("- **Đóng góp lượng tử** (SCF PGM − SCF ridge, mọi run, n={}): {} {} → **{}**".format(
        q["n"], fmt(q["mean"]), fmt_ci(q), "giữ được" if q["holds"] else "không giữ được"))
    checks = ["{} {}: {}".format(NAMES.get(r["dataset"], r["dataset"]), r["seed"],
                                 "=" if r["heads"]["linear"]["curve"] == r["cnn_curve"] else "≠ (LỖI)")
              for r in sorted(runs, key=lambda r: (ORDER.index(r["dataset"]) if r["dataset"] in ORDER else 9, r["seed"]))]
    missing = ["{} {}".format(NAMES[ds], s) for ds in ORDER for s in SEEDS
               if not any(r["dataset"] == ds and r["seed"] == s for r in runs)]
    lines += ["", "**Kiểm tra hồi quy (head linear = CNN curve):** " + "; ".join(checks),
              "", "**Còn thiếu:** " + (", ".join(missing) if missing else "không"), ""]
    return "\n".join(lines) + "\n", d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = collect(args.root)
    text, decision = render(runs)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        slim = [{k: v for k, v in r.items() if k != "cnn_curve"} for r in runs]
        Path(args.json).write_text(json.dumps({"runs": slim, "decision": decision}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
