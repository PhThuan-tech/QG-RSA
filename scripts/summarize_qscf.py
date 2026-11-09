"""W07: quantum at training time (PQK steering + QKSR) x statistics-calibrated fusion (SCF).

2x2 design on ImageNet-R seed 1993, three replicates per arm, both arms in one wave:
train RSIAT (``qscf_rsiat_rep<k>``) or QKSR (``qscf_qksr_rep<k>``), evaluate the linear head or
SCF (ridge w=0.25, PGM w=0.5, frozen in round A2). Every head of one run is evaluated on the same
network, so the SCF gains are paired within a run; the arms are paired by replicate k.

Pre-registered (exps/qscf/README.md). Do not edit after seeing results.

  python scripts/summarize_qscf.py --root <logs> [--root <logs> ...] [--output summary.md] [--json summary.json]
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

ARMS = (("qscf_rsiat", "RSIAT"), ("qscf_qksr", "QKSR"))
HEADS = (("linear", "linear head"), ("fuse_ridge_stats_b0p25", "SCF ridge"), ("fuse_pgm_stats_b0p5", "SCF PGM"))
PRIMARY_HEAD = "fuse_ridge_stats_b0p25"  # primary test: (QKSR + SCF ridge) - (RSIAT + SCF ridge), A-bar
METRICS = (("aia", "Ā"), ("final", "A_B"))
CURVE = re.compile(r"QSD curve head=(\w+): (\[.*\])")


def parse(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    family, replicate = split_family(match.group("prefix"))
    if family not in dict(ARMS):
        return None
    curves = {h: json.loads(v) for h, v in CURVE.findall(log.read_text(encoding="utf-8", errors="replace"))}
    if "linear" not in curves:
        return None
    heads = {h: {"aia": statistics.mean(c), "final": c[-1], "tasks": len(c)} for h, c in curves.items()}
    return {"family": family, "replicate": replicate, "seed": int(match.group("seed")), "log": str(log),
            "tasks": len(curves["linear"]), "heads": heads}


def collect(roots):
    runs = {}
    for root in roots:
        for log in sorted(Path(root).rglob("*.log")):
            run = parse(log)
            if run:
                key = (run["family"], run["replicate"])
                if key not in runs or run["tasks"] > runs[key]["tasks"]:
                    runs[key] = run
    tasks = max((r["tasks"] for r in runs.values()), default=0)
    return {k: r for k, r in runs.items() if r["tasks"] == tasks}  # completed runs only


def value(runs, family, rep, head, metric):
    run = runs.get((family, rep))
    return None if run is None or head not in run["heads"] else run["heads"][head][metric]


def paired(runs, a, b, metric):
    """a, b: (family, head). Differences a - b over the replicates present in both."""
    diffs = []
    for rep in sorted({r for _, r in runs}):
        va, vb = value(runs, *a[:1], rep, a[1], metric), value(runs, *b[:1], rep, b[1], metric)
        if va is not None and vb is not None:
            diffs.append(va - vb)
    return interval(diffs)


def interaction(runs, head, metric):
    """(QKSR + SCF - QKSR) - (RSIAT + SCF - RSIAT): does the SCF gain change on the quantum-trained network?"""
    diffs = []
    for rep in sorted({r for _, r in runs}):
        vals = [value(runs, f, rep, h, metric) for f in ("qscf_qksr", "qscf_rsiat") for h in (head, "linear")]
        if None not in vals:
            diffs.append((vals[0] - vals[1]) - (vals[2] - vals[3]))
    return interval(diffs)


def analyse(runs):
    cells = {}
    for family, _ in ARMS:
        for head, _ in HEADS:
            for metric, _ in METRICS:
                vals = [value(runs, family, rep, head, metric) for rep in range(3)]
                cells["{}|{}|{}".format(family, head, metric)] = interval([v for v in vals if v is not None])
    comparisons = []
    for metric, _ in METRICS:
        for head, name in HEADS[1:]:
            for family, arm in ARMS:
                comparisons.append(("{} − linear, mạng {}".format(name, arm), metric,
                                    paired(runs, (family, head), (family, "linear"), metric)))
        comparisons.append(("QKSR − RSIAT, linear head", metric,
                            paired(runs, ("qscf_qksr", "linear"), ("qscf_rsiat", "linear"), metric)))
        for head, name in HEADS[1:]:
            comparisons.append(("(QKSR + {0}) − (RSIAT + {0})".format(name), metric,
                                paired(runs, ("qscf_qksr", head), ("qscf_rsiat", head), metric)))
            comparisons.append(("tương tác: gain {} trên QKSR − trên RSIAT".format(name), metric,
                                interaction(runs, head, metric)))
    primary = paired(runs, ("qscf_qksr", PRIMARY_HEAD), ("qscf_rsiat", PRIMARY_HEAD), "aia")
    if primary["n"] < 3:
        verdict = "CHƯA ĐỦ: cần 3 cặp replicate, mới có {}".format(primary["n"])
    elif primary["ci95"][0] > 0:
        verdict = "ĐẠT: QKSR + SCF hơn RSIAT + SCF (cận dưới CI 95% của ΔĀ > 0)"
    else:
        verdict = "KHÔNG ĐẠT: QKSR không thêm gì đáng kể khi đã có SCF (CI 95% của ΔĀ chứa 0 hoặc âm)"
    return {"cells": cells, "comparisons": comparisons, "primary": primary, "verdict": verdict}


def render(runs, result):
    lines = ["# W07: lượng tử lúc train × SCF (ImageNet-R, seed 1993)", "",
             "Run hoàn chỉnh: " + ", ".join("{}_rep{}".format(f, r) for f, r in sorted(runs)), ""]
    for metric, label in METRICS:
        lines += ["## Bảng 2×2, {} (mean ± SD, n = số replicate)".format(label), "",
                  "| Train \\ Head | " + " | ".join(n for _, n in HEADS) + " |",
                  "|---|" + "---:|" * len(HEADS)]
        for family, arm in ARMS:
            row = []
            for head, _ in HEADS:
                c = result["cells"]["{}|{}|{}".format(family, head, metric)]
                row.append("{} ± {} (n={})".format(fmt(c["mean"], 2), fmt(c.get("sd"), 2), c["n"])
                           if c["n"] else "—")
            lines.append("| {} | {} |".format(arm, " | ".join(row)))
        lines.append("")
    lines += ["## So sánh ghép cặp (theo replicate)", "", "| So sánh | Metric | Δ | CI 95% | n |", "|---|---|---:|---|---:|"]
    for name, metric, e in result["comparisons"]:
        lines.append("| {} | {} | {} | {} | {} |".format(name, dict(METRICS)[metric], fmt(e["mean"], 3), fmt_ci(e), e["n"]))
    p = result["primary"]
    lines += ["", "## Kiểm định chốt trước", "",
              "(QKSR + SCF ridge) − (RSIAT + SCF ridge), Ā: {} {} (n={})".format(fmt(p["mean"], 3), fmt_ci(p), p["n"]),
              "", "**{}**".format(result["verdict"]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = collect(args.root)
    result = analyse(runs)
    text = render(runs, result)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": list(runs.values()), **result}, indent=2, default=str),
                                   encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
