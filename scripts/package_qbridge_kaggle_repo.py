"""Create the source-only ZIP attached to the offline Kaggle notebook."""
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "QG-RSA_QBRIDGE_Kaggle.zip"
EXCLUDED_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "logs", "ckpt", "reports", "dist", "result", "results",
                 "console", "paper", "_duplicates_to_delete"}
# Measured outputs and weights never belong in the code package (the notebook searches inputs for models).
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".npz", ".pt", ".pth", ".pkl", ".bin", ".safetensors"}


def included(path):
    relative = path.relative_to(ROOT)
    return (path.is_file() and not any(part in EXCLUDED_DIRS for part in relative.parts)
            and path.suffix.lower() not in EXCLUDED_SUFFIXES)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(ROOT.rglob("*")):
            if included(path):
                archive.write(path, (Path("QG-RSA") / path.relative_to(ROOT)).as_posix())
    temporary.replace(OUTPUT)
    with zipfile.ZipFile(OUTPUT) as archive:
        names = set(archive.namelist())
        required = {"QG-RSA/main.py", "QG-RSA/utils/qbridge.py", "QG-RSA/utils/stats_diagnostics.py",
                    "QG-RSA/scripts/run_level_stats.py", "QG-RSA/QGRSA_EXPERIMENTS.ipynb",
                    "QG-RSA/scripts/qbridge_prepare.py", "QG-RSA/QBRIDGE_Kaggle.ipynb"}
        if required - names:
            raise RuntimeError("Incomplete package: " + str(required - names))
        bad = [name for name in names if "/.git/" in name or "__pycache__" in name]
        if bad:
            raise RuntimeError("Generated files leaked into package: " + str(bad[:10]))
    print(OUTPUT)


if __name__ == "__main__":
    main()
