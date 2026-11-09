"""Lock a commit as the Kaggle code base (upload once, reuse for every wave).

  python scripts/pack_base.py [--commit HEAD]   ->  dist/qgrsa-base-<sha7>.zip

The zip is ``git archive`` of the commit (committed files only, never the working tree)
without results, paper and notebooks, plus BASE_COMMIT at its root. Upload it as a new
Kaggle dataset (or a new version of it). Waves built against this commit then only carry
their patch (scripts/build_wave.py).
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Never in the base nor in a patch: results, writing, packaging, notebooks.
EXCLUDE = ("exps/**/results/**", "exps/archive/**", "paper/**", "docs/**", "dist/**", "images/**",
           "kaggle/waves/**", "*.ipynb")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def resolve(commit):
    return git("rev-parse", commit).decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    commit = resolve(args.commit)
    out = ROOT / "dist" / "qgrsa-base-{}.zip".format(commit[:7])
    out.parent.mkdir(exist_ok=True)
    git("-c", "core.autocrlf=false", "archive", "--format=zip", "-o", str(out), commit, "--", ".",
        *(":(exclude,glob){}".format(p) for p in EXCLUDE))
    with zipfile.ZipFile(out, "a") as z:
        z.writestr("BASE_COMMIT", commit + "\n")
        count = len(z.namelist())
    print("{} ({} files, {:.1f} MB): upload to Kaggle as the code-base dataset".format(
        out, count, out.stat().st_size / 2 ** 20))


if __name__ == "__main__":
    main()
