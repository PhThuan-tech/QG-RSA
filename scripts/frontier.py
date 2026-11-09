"""Q-BOUND part A: old/new trade-off of every stored run and head (exps/qbound/README.md section 2).

Reads only the saved predictions (``<log>_stats/predictions_task_<k>.npz`` for the trained network's
linear head, ``qsd_predictions_task_<k>.npz`` / ``cahead_predictions_task_<k>.npz`` for extra heads on
the same network) and recomputes, for each (run, head): per-task accuracy after the last task, the
old / new halves, final accuracy and AIA. Deltas are taken against matched RSIAT references and scaled
by the run-to-run SD of RSIAT replicates (post-hoc, exploratory analysis: see the spec, 2.4).

    python scripts/frontier.py [--out exps/qbound/results/frontier]
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = re.compile(r"logs/adapter/(?P<ds>[^/]+)/(?P<init>\d+)/(?P<inc>\d+)/(?P<name>.+?)_(?P<seed>\d{4})"
                     r"_pretrained_.*_stats$")

# Pre-specified groups (spec 2.1). Families are run-name prefixes without the replicate suffix.
TRAINING = {"qksr_cur", "qksr_old", "holevo_vn", "holevo_mcr2", "holevo_spread", "hp_full", "hp_povm"}
STATS_CA = {"upper_oracle", "diag_oracle_mean", "diag_oracle_mean_cov", "diag_paired", "r2_ca_lognorm",
            "r2_ca_taskmargin", "r2_combo", "r2_sigma_legacy", "r2_sigma_paired"}
EXCLUDED = {"hp_twin", "upper_joint"}  # collapsed run; joint training is not incremental
# Families whose linear head is plain RSIAT (same training): replicates of the baseline.
RSIAT_FAMILIES = {"noise", "hp_diag", "qsd2", "qsd", "svf", "scf", "diag_legacy", "qrfca"}
# Inference heads of the main analysis (chosen in earlier rounds, not here): SCF ridge, PGM fusion, SVF.
INFERENCE_MAIN = {"fuse_ridge_stats_b0p25": "SCF ridge", "fuse_pgm_stats_b0p5": "SCF PGM",
                  "fuse_snap_ridge_b0p25": "SVF"}
INFERENCE_EXTRA_PREFIX = ("fuse_ridge_stats_b", "ridge_stats", "pgm_stats", "qda", "rff_ridge", "pqk_ridge")


def family(name):
    return re.sub(r"_rep\d+$", "", name)


def stage_files(stats, prefix):
    files = {}
    for path in stats.glob(prefix + "_task_*.npz"):
        k = int(path.stem.rsplit("_", 1)[1])
        files[k] = path
    return [files[k] for k in sorted(files)]


def curves(files, inc):
    """{head: (per-task accuracy after the last task, overall accuracy per stage)} from stage files."""
    out = {}
    last = len(files) - 1
    for k, path in enumerate(files):
        z = np.load(path, allow_pickle=False)
        target = z["target"]
        heads = [h for h in z.files if h not in ("target", "known_classes")]
        for head in heads:
            pred = z["pred"] if head == "pred" else z[head]
            name = "linear" if head == "pred" else head
            acc = 100.0 * float((pred == target).mean())
            entry = out.setdefault(name, {"stages": [], "tasks": None})
            entry["stages"].append(acc)
            if k == last:
                task = target // inc
                entry["tasks"] = [100.0 * float((pred[task == t] == target[task == t]).mean())
                                  for t in range(int(task.max()) + 1)]
    return {h: e for h, e in out.items() if len(e["stages"]) == len(files) and e["tasks"]}


def summary(entry):
    tasks = entry["tasks"]
    half = len(tasks) // 2
    return {"old": statistics.mean(tasks[:half]), "new": statistics.mean(tasks[half:]),
            "final": entry["stages"][-1], "aia": statistics.mean(entry["stages"]), "tasks": tasks}


def collect():
    rows = []
    for stats in sorted(ROOT.glob("exps/**/*_stats")):
        match = RUN_DIR.search(stats.as_posix())
        if not match or not stats.is_dir():
            continue
        name, inc = match["name"], int(match["inc"])
        fam = family(name)
        if fam in EXCLUDED:
            continue
        run = {"dataset": match["ds"], "seed": int(match["seed"]), "name": name, "family": fam,
               "source": stats.relative_to(ROOT).as_posix()}
        base = stage_files(stats, "predictions")
        heads = {}
        if base:
            heads.update(curves(base, inc))
        for prefix in ("qsd_predictions", "cahead_predictions"):
            extra = stage_files(stats, prefix)
            if extra and len(extra) == len(base):
                for head, entry in curves(extra, inc).items():
                    heads.setdefault(head, entry)
        for head, entry in heads.items():
            rows.append(dict(run, head=head, **summary(entry)))
    return rows


def dedup(rows):
    """Bit-identical networks (same per-task accuracy) count once as RSIAT replicates."""
    seen, out = set(), []
    for r in rows:
        key = (r["dataset"], r["seed"], tuple(round(a, 4) for a in r["tasks"]))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else None


def analyse(rows):
    linear = [r for r in rows if r["head"] == "linear"]
    refs = {}
    for r in dedup([r for r in linear if r["family"] in RSIAT_FAMILIES]):
        refs.setdefault((r["dataset"], r["seed"]), []).append(r)
    reference = {}
    for key, reps in refs.items():
        reference[key] = {m: statistics.mean(r[m] for r in reps) for m in ("old", "new", "final", "aia")}
        reference[key].update({"sd_" + m: sd([r[m] for r in reps]) for m in ("old", "new", "final", "aia")})
        reference[key]["n"] = len(reps)
        reference[key]["members"] = [r["source"].split("/logs/")[0] + ":" + r["name"] for r in reps]

    pooled = {}  # seeds with a single RSIAT run borrow the SD of the replicated seed of the same dataset
    for (ds, _), ref in reference.items():
        for m in ("old", "new", "final", "aia"):
            if ref["sd_" + m] is not None:
                pooled[(ds, m)] = ref["sd_" + m]
    points = []
    for r in rows:
        key = (r["dataset"], r["seed"])
        ref = reference.get(key)
        if ref is None:
            continue
        if r["head"] == "linear" and r["family"] in TRAINING:
            group = "training"
        elif r["head"] == "linear" and r["family"] in STATS_CA:
            group = "stats_ca"
        elif r["head"] in INFERENCE_MAIN and r["family"] in RSIAT_FAMILIES:
            group = "inference"
        elif r["head"] != "linear" and r["family"] in RSIAT_FAMILIES and r["head"].startswith(INFERENCE_EXTRA_PREFIX):
            group = "inference_extra"
        else:
            continue
        own = next((x for x in linear if x["source"] == r["source"]), None)
        point = dict(r, group=group, label=INFERENCE_MAIN.get(r["head"], r["head"] if r["head"] != "linear"
                                                               else r["name"]))
        for m in ("old", "new", "final", "aia"):
            point["d_" + m] = r[m] - ref[m]
            if own is not None and group.startswith("inference"):
                point["paired_d_" + m] = r[m] - own[m]
            s = ref["sd_" + m] or pooled.get((r["dataset"], m))
            point["z_" + m] = None if not s else point["d_" + m] / s
        points.append(point)
    unique, seen = [], set()
    for p in points:  # bit-identical networks (e.g. hp_diag and the SVF run) count once per head
        key = (p["dataset"], p["seed"], p["head"], tuple(round(a, 4) for a in p["tasks"]))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return reference, unique


def verdicts(reference, points, key=("imagenetr", 1993)):
    ref = reference[key]
    scope = [p for p in points if (p["dataset"], p["seed"]) == key]
    train = [p for p in scope if p["group"] == "training"]
    moved = [p for p in train if p["z_old"] is not None and abs(p["z_old"]) > 2]
    held = [p for p in moved if abs(p["z_final"]) < 2]
    infer = [p for p in scope if p["group"] == "inference"]
    out = {"scope": "{} {}".format(*key), "reference_n": ref["n"],
           "sd": {m: ref["sd_" + m] for m in ("old", "new", "final", "aia")},
           "training_n": len(train), "training_moved_old": len(moved), "moved_and_final_flat": len(held),
           "frontier_holds": bool(moved) and len(held) / len(moved) >= 0.8,
           "mean_z_final_training": statistics.mean(p["z_final"] for p in train) if train else None,
           "mean_z_final_inference": statistics.mean(p["z_final"] for p in infer) if infer else None}
    if train and infer:
        out["inference_crosses"] = out["mean_z_final_inference"] - out["mean_z_final_training"] >= 1
    rates = [-p["d_new"] / p["d_old"] for p in moved if p["d_old"]]
    out["exchange_rate_median"] = statistics.median(rates) if rates else None
    return out


def fmt(v, digits=2, sign=True):
    if v is None:
        return "—"
    return ("{:+.%df}" % digits if sign else "{:.%df}" % digits).format(v)


def render(reference, points, verdict):
    lines = ["# Q-BOUND phần A: đường đánh đổi cũ–mới (hậu kiểm, `scripts/frontier.py`)", ""]
    lines += ["Old = trung bình acc các task nửa đầu, new = nửa sau, sau task cuối; Δ so với trung bình RSIAT "
              "cùng dataset và seed; z = Δ / SD giữa các replicate RSIAT.", ""]
    lines += ["## Tham chiếu RSIAT", "", "| Dataset | Seed | n | Old | New | Final | AIA | SD old | SD new | SD final |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (ds, seed), ref in sorted(reference.items()):
        lines.append("| {} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} |".format(
            ds, seed, ref["n"], ref["old"], ref["new"], ref["final"], ref["aia"],
            *(fmt(ref["sd_" + m], 2, False) for m in ("old", "new", "final"))))
    for group, title in (("training", "Can thiệp lúc huấn luyện (adapter)"), ("stats_ca", "Can thiệp thống kê / CA"),
                         ("inference", "Can thiệp lúc suy luận (head chính)")):
        sel = sorted((p for p in points if p["group"] == group),
                     key=lambda p: (p["dataset"], p["seed"], p["label"]))
        if not sel:
            continue
        lines += ["", "## " + title, "",
                  "| Dataset | Seed | Run / head | Δ old (z) | Δ new (z) | Δ final (z) | Δ AIA (z) |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for p in sel:
            cells = ["{} ({})".format(fmt(p["d_" + m]), fmt(p["z_" + m], 1)) for m in ("old", "new", "final", "aia")]
            label = p["label"] if group != "inference" else "{} · {}".format(p["label"], p["name"])
            lines.append("| {} | {} | {} | {} |".format(p["dataset"], p["seed"], label, " | ".join(cells)))
    v = verdict
    lines += ["", "## Đọc theo tiêu chí ghi trước (spec 2.4, ImageNet-R 1993)", "",
              "- Tham chiếu: {} replicate RSIAT; SD old {}, new {}, final {}.".format(
                  v["reference_n"], *(fmt(v["sd"][m], 2, False) for m in ("old", "new", "final"))),
              "- Can thiệp huấn luyện: {} run, {} run có |z old| > 2, trong đó {} run có |z final| < 2 → "
              "**đường đánh đổi {}** (ngưỡng 80%).".format(
                  v["training_n"], v["training_moved_old"], v["moved_and_final_flat"],
                  "GIỮ" if v["frontier_holds"] else "KHÔNG GIỮ"),
              "- Tỉ giá trung vị −Δnew/Δold (run đã dịch old): {}.".format(fmt(v["exchange_rate_median"], 2, False)),
              "- z final trung bình: huấn luyện {}, suy luận {} → **suy luận {}vượt đường** (ngưỡng chênh ≥ 1).".format(
                  fmt(v["mean_z_final_training"]), fmt(v["mean_z_final_inference"]),
                  "" if v.get("inference_crosses") else "KHÔNG ")]
    return "\n".join(lines) + "\n"


def short_label(p):
    if p["group"] == "inference":
        return "SVF" if p["label"] == "SVF" else "SCF (ridge, PGM)"
    fam = p["family"]
    if fam.startswith("qksr"):
        return "QKSR (6 run)"
    if fam.startswith("upper_oracle"):
        return "oracle stats (2 run)"
    return fam


def plot(points, verdict, path, key=("imagenetr", 1993)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scope = [p for p in points if (p["dataset"], p["seed"]) == key]
    sd_old, sd_new = verdict["sd"]["old"], verdict["sd"]["new"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color="#d4d3cf", lw=0.8, zorder=1)
    ax.axvline(0, color="#d4d3cf", lw=0.8, zorder=1)
    ax.add_patch(matplotlib.patches.Rectangle((-2 * sd_old, -2 * sd_new), 4 * sd_old, 4 * sd_new,
                                              color="#bbbbbb", alpha=0.35, lw=0, label="RSIAT ±2 SD"))
    xs = np.linspace(-0.8, 1.8, 2)
    ax.plot(xs, -xs, color="#777777", ls="--", lw=1, label="Δfinal = 0 (old ≈ new)")
    style = {"training": ("#2a78d6", "o", "huấn luyện (adapter)"), "stats_ca": ("#1baf7a", "s", "thống kê / CA (oracle)"),
             "inference": ("#eb6834", "^", "suy luận (SCF / SVF)")}
    for group, (color, marker, label) in style.items():
        sel = [p for p in scope if p["group"] == group]
        if sel:
            ax.scatter([p["d_old"] for p in sel], [p["d_new"] for p in sel], c=color, marker=marker, s=42,
                       label=label, edgecolors="white", linewidths=0.6, zorder=3)
            names = {}  # one direct label per method family, at the centroid of its points
            for p in sel:
                names.setdefault(short_label(p), []).append((p["d_old"], p["d_new"]))
            for name, xy in names.items():
                x, y = np.mean(xy, axis=0)
                ax.annotate(name, (x, y), fontsize=7.5, xytext=(6, -10 if group == "training" else 5),
                            textcoords="offset points", color="#52514e")
    ax.set_xlim(-0.8, 2.7)
    ax.set_ylim(-1.8, 0.75)
    ax.set_xlabel("Δ acc task cũ (nửa đầu) so với RSIAT, điểm %")
    ax.set_ylabel("Δ acc task mới (nửa sau) so với RSIAT, điểm %")
    ax.set_title("ImageNet-R B0Inc20, seed 1993: đánh đổi cũ–mới", fontsize=10)
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="exps/qbound/results/frontier")
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = collect()
    reference, points = analyse(rows)
    verdict = verdicts(reference, points)
    text = render(reference, points, verdict)
    (out / "frontier.md").write_text(text, encoding="utf-8")
    payload = {"reference": {"{}_{}".format(*k): v for k, v in reference.items()}, "verdict": verdict,
               "points": [{k: v for k, v in p.items()} for p in points]}
    (out / "frontier.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    plot(points, verdict, out / "frontier_imagenetr_1993.png")
    print(text)


if __name__ == "__main__":
    main()
