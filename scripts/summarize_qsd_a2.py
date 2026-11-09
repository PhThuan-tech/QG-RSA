"""Summarise QSD round A2: calibrated fusion, states from stored statistics, oracle control.

Fusion weights are chosen on the tuning group only (ImageNet-R, seed 1993) and then
frozen for the held-out runs (other seeds and ImageNet-A). Decision rules are
pre-registered in exps/qsd/README_A2.md and fixed below.

  python scripts/summarize_qsd_a2.py --root <logs> [--root <logs> ...] \
      [--stage-a-root <stage A logs> ...] [--output summary.md] [--json summary.json]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.summarize_ca_heads import fmt, fmt_ci  # noqa: E402
from scripts.summarize_qsd import collect, contrast  # noqa: E402
from utils.qsd import FUSE_SOURCES, FUSE_WEIGHTS, fuse_name  # noqa: E402

# Pre-registered (exps/qsd/README_A2.md). Do not edit after seeing results.
FAMILY = "qsd2"
TUNE = ("imagenetr", 1993)
MIN_RUNS = 3
MIN_GAIN = 0.3
METHOD, CLASSICAL = "pgm_stats", "ridge_stats"  # primary quantum source and its classical twin
TRANSPORT_GAIN = 1.0  # final-task pp: oracle states over stored states worth a transport method
BASE_HEADS = ("linear", "ridge_stats", "pgm_channel", "pgm_channel_hier", "pgm_fuse", "pgm_stats_stale",
              "pgm_stats", "pgm_stats_hier", "pgm_stats_oracle")
STAGE_A_SHARED = ("linear", "ridge_stats", "pgm_channel", "pgm_channel_hier", "pgm_fuse")


def is_tune(run):
    return (run["dataset"], run["seed"]) == TUNE


def tuned_weights(tune):
    """Per source: the weight with the best mean AIA on the tuning runs (ties -> smaller weight)."""
    chosen = {}
    for source in FUSE_SOURCES:
        best = None
        for weight in FUSE_WEIGHTS:
            head = fuse_name(source, weight)
            if tune and all(head in r["heads"] for r in tune):
                score = statistics.mean(r["heads"][head]["aia"] for r in tune)
                if best is None or score > best[1] + 1e-12:
                    best = (weight, score)
        if best:
            chosen[source] = best[0]
    return chosen


def holds(entry, min_gain=0.0, min_runs=MIN_RUNS):
    return entry["n"] >= min_runs and "ci95" in entry and entry["ci95"][0] > 0 and entry["mean"] >= min_gain


def held_out_contrast(heldout, head, rival):
    by_dataset = {ds: contrast([r for r in heldout if r["dataset"] == ds], head, rival)
                  for ds in sorted({r["dataset"] for r in heldout})}
    return by_dataset, contrast(heldout, head, rival)


def decide(runs):
    tune = [r for r in runs if is_tune(r)]
    heldout = [r for r in runs if not is_tune(r)]
    weights = tuned_weights(tune)
    out = {"weights": weights, "n_tune": len(tune), "n_heldout": len(heldout), "sources": {}, "claims": {}}
    for source, weight in weights.items():
        head = fuse_name(source, weight)
        per_ds, pooled = held_out_contrast(heldout, head, "linear")
        out["sources"][source] = {"head": head, "tune": contrast(tune, head, "linear"),
                                  "per_dataset": per_ds, "pooled": pooled}
    if METHOD in weights:
        m = out["sources"][METHOD]
        ok = holds(m["pooled"], MIN_GAIN) and all(e["mean"] is not None and e["mean"] > 0
                                                  for e in m["per_dataset"].values())
        out["method"] = "ĐẠT" if ok else ("chưa đủ run held-out" if m["pooled"]["n"] < MIN_RUNS else "KHÔNG ĐẠT")
    if METHOD in weights and CLASSICAL in weights:
        per_ds, pooled = held_out_contrast(heldout, fuse_name(METHOD, weights[METHOD]),
                                           fuse_name(CLASSICAL, weights[CLASSICAL]))
        ok = holds(pooled) and all(e["mean"] is not None and e["mean"] > 0 for e in per_ds.values())
        out["quantum"] = {"per_dataset": per_ds, "pooled": pooled, "holds": ok}
    for head, rival, label in (("pgm_stats", "pgm_stats_stale", "Trạng thái theo mean SSCA tốt hơn mean cũ"),
                               ("pgm_stats", "ridge_stats", "Phép đo PGM tốt hơn ridge trên cùng thống kê"),
                               ("pgm_stats", "pgm_channel", "Trạng thái từ thống kê so với từ feature từng mẫu"),
                               ("pgm_stats_hier", "pgm_stats", "Phân cấp task → lớp có lợi")):
        if all(head in r["heads"] and rival in r["heads"] for r in runs):
            c = contrast(runs, head, rival)
            c.update(label=label, holds=holds(c))
            out["claims"]["{} − {}".format(head, rival)] = c
    if all("pgm_stats_oracle" in r["heads"] and "pgm_stats" in r["heads"] for r in runs) and runs:
        c = contrast(runs, "pgm_stats_oracle", "pgm_stats", "final")
        c.update(label="Trần của vận chuyển trạng thái (final, oracle − thống kê)",
                 holds=c["mean"] is not None and c["mean"] >= TRANSPORT_GAIN)
        out["claims"]["pgm_stats_oracle − pgm_stats (final)"] = c
    return out


def stage_a_check(runs, stage_a):
    refs = {(r["dataset"], r["seed"], r["replicate"]): r for r in stage_a}
    lines = []
    for r in runs:
        ref = refs.get((r["dataset"], r["seed"], r["replicate"]))
        cnn = "=" if r["heads"].get("linear", {}).get("curve") == r["cnn_curve"] else "≠"
        if ref is None:
            lines.append("{} seed {} rep {}: linear {} CNN; không có run giai đoạn A".format(
                r["dataset"], r["seed"], r["replicate"], cnn))
            continue
        diff = [h for h in STAGE_A_SHARED if h in r["heads"] and h in ref["heads"]
                and r["heads"][h]["curve"] != ref["heads"][h]["curve"]]
        lines.append("{} seed {} rep {}: linear {} CNN; head giai đoạn A {}".format(
            r["dataset"], r["seed"], r["replicate"], cnn, "trùng hết" if not diff else "LỆCH: " + ", ".join(diff)))
    return lines


def render(runs, stage_a):
    d = decide(runs)
    lines = ["# QSD-CIL vòng A2", "",
             "Tuning: {} seed {} ({} run). Held-out: {} run.".format(TUNE[0], TUNE[1], d["n_tune"], d["n_heldout"]), ""]
    for ds in sorted({r["dataset"] for r in runs}):
        group = [r for r in runs if r["dataset"] == ds]
        lines += ["## {} | {} run (mọi seed)".format(ds, len(group)), "",
                  "| Head | AIA | Final | Task 0 | Dự đoán task (%) | Δ AIA vs linear | CI 95% |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for head in BASE_HEADS + tuple(fuse_name(s, w) for s, w in d["weights"].items()):
            if not all(head in r["heads"] for r in group):
                continue
            def mean(key):
                values = [r["heads"][head][key] for r in group if r["heads"][head][key] is not None]
                return statistics.mean(values) if values else None
            c = contrast(group, head, "linear")
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                head, fmt(mean("aia"), 2), fmt(mean("final"), 2), fmt(mean("task0"), 2),
                fmt(mean("task_prediction"), 2), fmt(c["mean"]) if head != "linear" else "",
                fmt_ci(c) if head != "linear" else ""))
        lines.append("")
    tune = [r for r in runs if is_tune(r)]
    lines += ["## Chọn trọng số fusion (chỉ trên nhóm tuning)", "",
              "| Nguồn | " + " | ".join("w={:g}".format(w) for w in FUSE_WEIGHTS) + " | Chọn |",
              "|---|" + "---:|" * len(FUSE_WEIGHTS) + "---|"]
    for source in FUSE_SOURCES:
        cells = []
        for w in FUSE_WEIGHTS:
            c = contrast(tune, fuse_name(source, w), "linear") if tune else {"mean": None}
            cells.append(fmt(c["mean"]))
        lines.append("| {} | {} | {} |".format(source, " | ".join(cells),
                                               "w={:g}".format(d["weights"][source]) if source in d["weights"] else ""))
    lines += ["", "## Held-out (trọng số cố định), Δ AIA so với linear", "",
              "| Nguồn | Head | Tuning | " + " | ".join(sorted({r["dataset"] for r in runs if not is_tune(r)}))
              + " | Gộp | CI 95% gộp |", "|---|---|---:|" + "---:|" * len({r["dataset"] for r in runs if not is_tune(r)})
              + "---:|---|"]
    for source, s in d["sources"].items():
        per = " | ".join("{} (n={})".format(fmt(e["mean"]), e["n"]) for _, e in sorted(s["per_dataset"].items()))
        lines.append("| {} | {} | {} | {} | {} | {} |".format(source, s["head"], fmt(s["tune"]["mean"]), per,
                                                           fmt(s["pooled"]["mean"]), fmt_ci(s["pooled"])))
    lines += ["", "## Quyết định", ""]
    if "method" in d:
        lines.append("- **Phương pháp** (`{}` fusion, held-out gộp: Δ ≥ {}, CI > 0, mọi dataset > 0): **{}**".format(
            METHOD, MIN_GAIN, d["method"]))
    if "quantum" in d:
        q = d["quantum"]
        lines.append("- **Đóng góp của phép đo lượng tử** (fusion `{}` − fusion `{}`, held-out): Δ {} {} → **{}**".format(
            METHOD, CLASSICAL, fmt(q["pooled"]["mean"]), fmt_ci(q["pooled"]), "giữ được" if q["holds"] else "không giữ được"))
    for name, c in d["claims"].items():
        lines.append("- {} (`{}`, mọi run, n={}): Δ {} {} → {}".format(
            c["label"], name, c["n"], fmt(c["mean"]), fmt_ci(c), "có" if c["holds"] else "không"))
    lines += ["", "**Kiểm tra hồi quy:** " + "; ".join(stage_a_check(runs, stage_a)), ""]
    return "\n".join(lines) + "\n", d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--stage-a-root", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = [r for r in collect(args.root) if r["family"] == FAMILY]
    stage_a = [r for r in collect(args.stage_a_root) if r["family"] == "qsd"] if args.stage_a_root else []
    text, decision = render(runs, stage_a)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": runs, "decision": decision}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
