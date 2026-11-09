"""Run-level statistics: training-noise calibration and paired candidate comparison.

The unit of analysis is one training run. Per-image tests (McNemar) only see
test-sampling variance; round 2 showed a pure-noise replicate reaching p=0.087,
so they cannot decide whether a method beats training noise.

Run families: a log prefix with an optional ``_rep<k>`` suffix, e.g. ``noise``,
``noise_rep1`` → family ``noise``, replicate 0 and 1. A candidate is compared
with the baseline on matched (dataset, seed, replicate) pairs, which share the
task-0 base and the same random stream (common random numbers).

  python scripts/run_level_stats.py noise   --root <logs> [--root <logs> ...]
  python scripts/run_level_stats.py compare --root <logs> --baseline noise --candidate r3_x
"""
import argparse
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

RUN_NAME = re.compile(r"^(?P<prefix>.+?)_(?P<seed>\d+)_(?P<convnet>.+)$")
REPLICATE = re.compile(r"^(?P<family>.+?)_rep(?P<k>\d+)$")
CURVE = re.compile(r"CNN top1 curve: (\[.*\])")
FORGETTING = re.compile(r"Continual metrics after task \d+: BWT=([-\d.]+), forgetting=([-\d.]+)")
STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
METRICS = ("aia", "final", "forgetting")

# Pre-registered constants (exps/measurement/README.md). Do not edit after seeing results.
TARGET_EFFECTS = (0.3, 0.5, 1.0)     # pp of AIA
PLANNING_EFFECT = 0.5
Z_ALPHA, Z_POWER = 1.959964, 0.841621  # two-sided 5%, 80% power
MIN_PAIRS = 3
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]


def t975(df):
    return T975[df - 1] if 1 <= df <= len(T975) else Z_ALPHA


def split_family(prefix):
    match = REPLICATE.match(prefix)
    return (match.group("family"), int(match.group("k"))) if match else (prefix, 0)


def parse_run(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    curve, forgetting, stamps = None, None, []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = CURVE.search(line)
        if m:
            curve = json.loads(m.group(1))
        m = FORGETTING.search(line)
        if m:
            forgetting = float(m.group(2))
        m = STAMP.match(line)
        if m:
            stamps.append(m.group(1))
    if not curve:
        return None
    family, replicate = split_family(match.group("prefix"))
    minutes = None
    if len(stamps) > 1:
        fmt = "%Y-%m-%d %H:%M:%S"
        minutes = (datetime.strptime(stamps[-1], fmt) - datetime.strptime(stamps[0], fmt)).total_seconds() / 60
    # logs/<model>/<dataset>/<init>/<increment>/<file>.log
    dataset = log.parent.parent.parent.name
    return {"dataset": dataset, "family": family, "replicate": replicate, "seed": int(match.group("seed")),
            "tasks": len(curve), "aia": sum(curve) / len(curve), "final": curve[-1],
            "forgetting": forgetting, "minutes": minutes, "log": str(log)}


def collect(roots):
    runs = []
    for root in roots:
        runs.extend(r for r in (parse_run(p) for p in sorted(Path(root).rglob("*.log"))) if r)
    complete = {}
    for run in runs:
        key = (run["dataset"], run["family"], run["seed"], run["replicate"])
        if key in complete:
            raise ValueError("Duplicate run {}: {} and {}".format(key, complete[key]["log"], run["log"]))
        complete[key] = run
    # Only runs that finished every task of their dataset count.
    longest = {}
    for run in runs:
        longest[run["dataset"]] = max(longest.get(run["dataset"], 0), run["tasks"])
    return [r for r in runs if r["tasks"] == longest[r["dataset"]]]


def pairs_needed(sd_diff, effect):
    return math.ceil(((Z_ALPHA + Z_POWER) * sd_diff / effect) ** 2)


def noise_table(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["dataset"], run["family"], run["seed"]), []).append(run)
    rows = []
    for (dataset, family, seed), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        row = {"dataset": dataset, "family": family, "seed": seed, "replicates": len(members),
               "minutes": statistics.mean(m["minutes"] for m in members if m["minutes"]) if any(
                   m["minutes"] for m in members) else None}
        for metric in METRICS:
            values = [m[metric] for m in members if m[metric] is not None]
            row[metric + "_mean"] = statistics.mean(values) if values else None
            row[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        # Conservative planning: independent noise in both arms, sd_diff = sqrt(2) * sd.
        sd_diff = math.sqrt(2) * row["aia_sd"]
        row["pairs_needed"] = {str(e): max(MIN_PAIRS, pairs_needed(sd_diff, e)) for e in TARGET_EFFECTS}
        row["cost_minutes"] = (row["pairs_needed"][str(PLANNING_EFFECT)] * 2 * row["minutes"]
                               if row["minutes"] else None)
        rows.append(row)
    return rows


def compare(runs, baseline, candidate, dataset=None):
    index = {(r["dataset"], r["family"], r["seed"], r["replicate"]): r for r in runs}
    diffs = {metric: [] for metric in METRICS}
    keys = []
    for (ds, family, seed, rep), run in sorted(index.items()):
        if family != candidate or (dataset and ds != dataset):
            continue
        ref = index.get((ds, baseline, seed, rep))
        if ref is None:
            continue
        keys.append((ds, seed, rep))
        for metric in METRICS:
            if run[metric] is not None and ref[metric] is not None:
                diffs[metric].append(run[metric] - ref[metric])
    result = {"baseline": baseline, "candidate": candidate, "pairs": keys, "n": len(keys)}
    for metric, values in diffs.items():
        n = len(values)
        entry = {"mean": statistics.mean(values) if values else None, "n": n}
        if n >= 2:
            sd = statistics.stdev(values)
            half = t975(n - 1) * sd / math.sqrt(n)
            entry.update(sd=sd, ci95=(entry["mean"] - half, entry["mean"] + half))
        result[metric] = entry
    aia = result["aia"]
    if result["n"] < MIN_PAIRS:
        result["verdict"] = "chưa đủ cặp (cần ≥ {})".format(MIN_PAIRS)
    elif aia["ci95"][0] > 0:
        result["verdict"] = "ĐẠT: CI 95% của Δ AIA nằm hoàn toàn trên 0"
    elif aia["ci95"][1] < 0:
        result["verdict"] = "KÉM HƠN: CI 95% của Δ AIA nằm hoàn toàn dưới 0"
    else:
        result["verdict"] = "không kết luận: CI 95% chứa 0"
    return result


def fmt(value, digits=3):
    return "" if value is None else "{:.{}f}".format(value, digits)


def render_noise(rows):
    lines = ["# Nhiễu huấn luyện theo benchmark", "",
             "SD giữa các replicate (cùng seed, cùng base task 0, khác ngẫu nhiên SGD từ task 1). "
             "Số cặp cần để phát hiện Δ AIA với power 80%, α = 5%, giả định thận trọng "
             "sd(Δ) = √2·SD (không tính lợi ích của common random numbers).", "",
             "| Dataset | Family | Seed | n | AIA mean ± SD | Final mean ± SD | Forgetting mean ± SD | "
             "Phút/run | Cặp cần (Δ=0.3 / 0.5 / 1.0) | Chi phí Δ=0.5 (giờ GPU tuần tự) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        need = r["pairs_needed"]
        lines.append("| {} | {} | {} | {} | {} ± {} | {} ± {} | {} ± {} | {} | {} / {} / {} | {} |".format(
            r["dataset"], r["family"], r["seed"], r["replicates"], fmt(r["aia_mean"]), fmt(r["aia_sd"]),
            fmt(r["final_mean"], 2), fmt(r["final_sd"], 2), fmt(r["forgetting_mean"], 2),
            fmt(r["forgetting_sd"], 2), fmt(r["minutes"], 0), need["0.3"], need["0.5"], need["1.0"],
            fmt(r["cost_minutes"] / 60 if r["cost_minutes"] else None, 1)))
    eligible = [r for r in rows if r["cost_minutes"] and r["replicates"] >= 3]
    if eligible:
        best = min(eligible, key=lambda r: r["cost_minutes"])
        lines += ["", "**Benchmark sàng lọc theo quy tắc đặt trước:** `{}` (chi phí thấp nhất để phát hiện "
                  "Δ AIA = {} với ≥ 3 replicate).".format(best["dataset"], PLANNING_EFFECT)]
    else:
        lines += ["", "Chưa benchmark nào có ≥ 3 replicate hoàn tất để áp quy tắc chọn."]
    lines += ["", "Lưu ý: phút/run đo trong điều kiện song song lúc chạy; chỉ so sánh tương đối "
              "giữa các benchmark chạy cùng điều kiện."]
    return "\n".join(lines) + "\n"


def render_compare(result):
    lines = ["# {} so với {}".format(result["candidate"], result["baseline"]), "",
             "Cặp ghép (dataset, seed, replicate): {}".format(result["pairs"]), "",
             "| Metric | n | Δ mean | SD(Δ) | CI 95% |", "|---|---:|---:|---:|---|"]
    for metric in METRICS:
        e = result[metric]
        ci = e.get("ci95")
        lines.append("| {} | {} | {} | {} | {} |".format(
            metric, e["n"], fmt(e["mean"]), fmt(e.get("sd")),
            "[{}, {}]".format(fmt(ci[0]), fmt(ci[1])) if ci else ""))
    lines += ["", "**Kết luận (Δ AIA):** " + result["verdict"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("noise", "compare"))
    parser.add_argument("--root", action="append", required=True, help="Log directory; repeatable")
    parser.add_argument("--baseline", default="noise")
    parser.add_argument("--candidate")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = collect(args.root)
    if args.mode == "noise":
        payload = noise_table(runs)
        text = render_noise(payload)
    else:
        if not args.candidate:
            parser.error("compare needs --candidate")
        payload = compare(runs, args.baseline, args.candidate, args.dataset)
        text = render_compare(payload)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
