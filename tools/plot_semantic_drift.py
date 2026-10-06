"""Plot task-level semantic-drift summaries produced by the opt-in observer."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Semantic-drift summary is empty.")

    tasks = [int(row["task_id"]) for row in rows]
    run_dir = args.summary.parent
    task_files = sorted(run_dir.glob("task_*.json"))
    task_data = []
    for path in task_files:
        with path.open(encoding="utf-8") as stream:
            task_data.append(json.load(stream))
    figure, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.ravel()

    def series(path, default=float("nan")):
        values = []
        for item in task_data:
            current = item
            for key in path:
                current = current.get(key, {}) if isinstance(current, dict) else {}
            values.append(current.get("mean", default) if isinstance(current, dict) else current)
        return values

    axes[0].plot(tasks, series(["old_sample_drift", "cosine"]), marker="o")
    axes[0].set_title("A: Old-class cosine drift")
    axes[1].plot(tasks, series(["new_task_shift", "cosine"]), marker="o")
    axes[1].set_title("B: New-task cosine shift")
    axes[2].plot(tasks, series(["old_new_drift_ratio", "cosine"]), marker="o")
    axes[2].set_title("C: Old/new cosine drift ratio")
    axes[3].plot(tasks, series(["ssca_diagnostic", "correction_error"]), marker="o")
    axes[3].set_title("D: SSCA correction error")
    axes[4].plot(tasks, series(["ssca_diagnostic", "direction_agreement"]), marker="o")
    axes[4].set_title("E: SSCA direction agreement")

    class_values = {}
    for item in task_data:
        for class_id, record in item.get("old_sample_drift", {}).get("per_class", {}).items():
            class_values.setdefault(class_id, []).append(
                record["sample_cosine"]["mean"]
            )
    if class_values:
        axes[5].boxplot(list(class_values.values()), labels=list(class_values))
        axes[5].set_title("F: Per-class old cosine drift")
        axes[5].tick_params(axis="x", labelrotation=90)
    else:
        axes[5].set_visible(False)

    pre_similarity = series(["old_new_separation", "pre_pairwise_cosine"])
    post_similarity = series(["old_new_separation", "post_pairwise_cosine"])
    axes[6].plot(tasks, pre_similarity, marker="o", label="pre")
    axes[6].plot(tasks, post_similarity, marker="o", label="post")
    axes[6].set_title("G: Old/new centroid cosine")
    axes[6].legend()
    axes[7].axis("off")
    for axis in axes[:7]:
        axis.set_xlabel("Incremental task")
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    output = args.output or args.summary.with_name("semantic_drift_plots.png")
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
