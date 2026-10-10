"""Build a source-only Kaggle upload ZIP; never include historical artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(output):
    paths = [ROOT / name for name in (
        "main.py", "trainer.py", "LICENSE", "README.md", "requirements-kaggle-offline.txt",
        "docs/KAGGLE_OFFLINE_FAILURE_LOCALIZATION.md", "notebooks/QKSR_Failure_Localization_Kaggle_Offline.ipynb",
    )]
    for folder in ("data", "models", "network", "utils", "scripts", "tests"):
        paths.extend((ROOT / folder).rglob("*.py"))
    paths.extend((ROOT / "exps").rglob("*.json"))
    paths = sorted(set(paths))
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, "qg_rsa/" + str(path.relative_to(ROOT)).replace("\\", "/"))
        archive.writestr("qg_rsa/BUILD_MANIFEST.json", json.dumps({
            "git_commit": commit, "tracked_source_dirty": dirty, "source_sha256": manifest,
            "contains_results": False,
        }, indent=2))
    print("Source-only offline upload ZIP: {}".format(output.resolve()))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/kaggle_failure_localization/source.zip")
    package(parser.parse_args().output)
