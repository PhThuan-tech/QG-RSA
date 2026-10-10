"""Collect real task artifacts for review; no metrics simulation or comparison report."""

import argparse
import json
from pathlib import Path
import zipfile


def collect(study, output):
    study = Path(study).resolve()
    plan = json.loads((study / "study_manifest.json").read_text(encoding="utf-8"))
    paths = [study / name for name in ("study_manifest.json", "preflight.json", "source_config.json")]
    paths.extend((study / "configs").glob("*.json"))
    completed, failed, absent = [], [], []
    for entry in plan["runs"]:
        directory = study / "runs" / entry["run_id"]
        if not directory.is_dir():
            absent.append(entry["run_id"])
            continue
        for name in ("tasks.csv", "tasks.jsonl", "run_manifest.json", "run_summary.json", "FAILED.json", "train.log", "runtime_output.log"):
            if (directory / name).is_file():
                paths.append(directory / name)
        if (directory / "run_summary.json").is_file():
            completed.append(entry["run_id"])
        else:
            failed.append(entry["run_id"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(study)).replace("\\", "/"))
        archive.writestr("collection_manifest.json", json.dumps({
            "status": "complete" if len(completed) == 9 else "partial",
            "completed": completed, "failed_or_incomplete": failed, "not_started": absent,
            "checkpoints_included": False,
        }, indent=2))
    print("Collected {} of 9 completed runs: {}".format(len(completed), output.resolve()))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    collect(args.study, args.output)
