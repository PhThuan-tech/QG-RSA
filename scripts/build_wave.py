"""Build the Kaggle notebook of one wave: locked code base + embedded patch.

  python scripts/build_wave.py kaggle/waves/<wave>/wave.json

wave.json names the base commit (packed once by scripts/pack_base.py) and the jobs. The
patch is every file that differs between that commit and the working tree (tracked or
new, same exclusions as the base, and no Markdown), stored with the hash of its original
so the runner refuses a wrong base. Output: <wave dir>/<NAME>_kaggle.ipynb with three
cells (what to attach and send back; run.py = PLAN + PATCH + kaggle/runner.py; run).
"""
import base64
import hashlib
import json
import pprint
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pack_base import ROOT, git, resolve  # noqa: E402

PLAN_KEYS = ("name", "jobs", "max_parallel", "overrides", "tests", "summaries", "data_sources", "model_file")


def excluded(path):
    """Same exclusions as the base (pack_base.EXCLUDE), plus Markdown and the runner itself."""
    parts = path.split("/")
    return (path.endswith((".md", ".ipynb")) or parts[0] in ("paper", "docs", "dist", "images", "kaggle")
            or path.startswith("exps/archive/") or (parts[0] == "exps" and "results" in parts[2:-1]))


def build_patch(base):
    changed = git("diff", "--name-status", "--no-renames", base, "--").decode().splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    files, base_sha, deleted = {}, {}, []
    entries = [line.split("\t", 1) for line in changed] + [["A", p] for p in untracked]
    for status, path in sorted(entries, key=lambda e: e[1]):
        if excluded(path):
            continue
        if status == "D":
            deleted.append(path)
            continue
        data = (ROOT / path).read_bytes()
        if b"\0" not in data:  # text: LF like the blobs in the base (Windows checkouts use CRLF)
            data = data.replace(b"\r\n", b"\n")
        files[path] = base64.b64encode(data).decode()
        if status == "M":
            base_sha[path] = hashlib.sha256(git("show", "{}:{}".format(base, path))).hexdigest()
    return {"base": base, "built": time.strftime("%Y-%m-%d %H:%M"), "files": files, "base_sha": base_sha,
            "deleted": deleted}


def lines(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base = resolve(spec["base"])
    patch = build_patch(base)
    plan = {k: spec[k] for k in PLAN_KEYS if k in spec}
    runner = (ROOT / "kaggle" / "runner.py").read_text(encoding="utf-8")
    run_py = "%%writefile /kaggle/working/run.py\n" + "\n".join([
        '"""{} (base {}). Sửa PLAN ở đây để đổi job; đổi code thì build lại wave."""'.format(spec["name"], base[:7]),
        "PLAN = " + pprint.pformat(plan, width=110, sort_dicts=False),
        "PATCH = " + repr(patch), "", runner])
    patched = "\n".join("- `{}`".format(p) for p in sorted(patch["files"])) or "- (không có)"
    intro = "\n".join([
        "# {}: {}".format(spec["name"], spec["title"]), "",
        *spec.get("description", []), "",
        "**Code:** base `{}` (dataset `qgrsa-base-{}`, tạo bằng `python scripts/pack_base.py --commit {}`) "
        "+ patch {} file nhúng trong cell 2 (build {}):".format(base[:7], base[:7], base[:7], len(patch["files"]),
                                                                 patch["built"]),
        patched, "",
        "**Trước khi chạy:** GPU **RTX PRO 6000**; *Add Input* > " + ", ".join("`{}`".format(i) for i in spec["inputs"])
        + "; Internet có thể Off. Rồi *Save Version > Save & Run All (Commit)*.", "",
        "**Gửi lại:** `results_{}.zip` (hoặc `runs/SUMMARY.txt`).".format(spec["name"])])
    cells = [{"cell_type": "markdown", "metadata": {}, "source": lines(intro)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(run_py)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": lines("!cd /kaggle/working && python3 -u run.py 2>&1 | tee -a /kaggle/working/run.log\n"
                              "!rm -rf /kaggle/working/repo /kaggle/working/*_split /kaggle/working/vit_b16_224.pt")}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = spec_path.parent / "{}_kaggle.ipynb".format(spec["name"])
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("{} | base {} | patch: {}{}".format(out, base[:7], ", ".join(sorted(patch["files"])) or "-",
                                             " | xoá: " + ", ".join(patch["deleted"]) if patch["deleted"] else ""))
    return out


if __name__ == "__main__":
    build(sys.argv[1])
