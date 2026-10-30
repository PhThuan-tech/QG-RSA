"""Kaggle runner for one wave (template).

scripts/build_wave.py writes run.py = ``PLAN`` + ``PATCH`` + this file. The code base is
the locked zip made by scripts/pack_base.py (uploaded once as a Kaggle dataset); this
runner checks its commit, applies the wave's patch (files changed since that commit),
then runs every job and packs the logs. Datasets and the ViT weights are attached inputs.

Edit PLAN in the notebook cell to change jobs; anything else needs a rebuild.
"""
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input") if Path("/kaggle/input").is_dir() else Path.cwd() / "input"
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "work"
WORK.mkdir(parents=True, exist_ok=True)
REPO, OUT = WORK / "repo", WORK / "runs"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TIMM_FUSED_ATTN="0")
N_CLASSES = {"imageneta": 200, "imagenetr": 200, "cifar224": 100, "cub": 200}
NAMES = {"imageneta": r"imagenet[-_]?a\b|imagenet-a|adversarial", "imagenetr": r"imagenet[-_]?r\b|imagenet-r|rendition"}
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SHOW = re.compile(r"CNN top1 curve|Average Accuracy|Error|Traceback|WARNING")
JOBS = [(ds, v, s) for datasets, seeds, variants in PLAN["jobs"] for ds in datasets for s in seeds for v in variants]


def say(*args):
    print(*args, flush=True)


def walk(root, dirs=False):
    """Files (or dirs) under root, following symlinks (Kaggle mounts some inputs as symlinks)."""
    seen = set()
    for top, dnames, fnames in os.walk(root, followlinks=True):
        real = os.path.realpath(top)
        if real in seen:
            dnames[:] = []
            continue
        seen.add(real)
        for name in (dnames if dirs else fnames):
            yield Path(top) / name


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------------------- environment
def check_gpu():
    probe = ("import torch, json; ok = torch.cuda.is_available(); cap = torch.cuda.get_device_capability(0) if ok else None;"
             "print(json.dumps({'torch': torch.__version__, 'ok': ok, 'name': torch.cuda.get_device_name(0) if ok else None,"
             "'cap': cap, 'arch': torch.cuda.get_arch_list()}))")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, "Không import được torch:\n" + r.stderr[-2000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    say("GPU:", info["name"], "| torch", info["torch"])
    assert info["ok"], "Không thấy GPU CUDA: bật accelerator"
    need = "sm_%d%d" % tuple(info["cap"])
    wheels = sorted({p.parent for p in walk(INPUT) if p.suffix == ".whl"})
    missing = [m for m in ("timm", "tqdm", "scipy") if subprocess.run([sys.executable, "-c", "import " + m]).returncode]
    if need not in info["arch"] or missing:
        assert wheels, "Thiếu {} và không có wheel offline trong input".format(missing or ["torch cho " + need])
        cmd = [sys.executable, "-m", "pip", "install", "--no-index"] + sum((["--find-links", str(d)] for d in wheels), [])
        subprocess.run(cmd + (["torch", "torchvision"] if need not in info["arch"] else []) + missing, check=True)
        say("Đã cài package từ wheel → chạy lại run.py trong tiến trình mới")
        os.execv(sys.executable, [sys.executable] + sys.argv)


# ----------------------------------------------------------------------------- code: locked base + patch
def find_base():
    """Folder (or zip) of the locked base: it holds BASE_COMMIT at its root."""
    for p in walk(INPUT):
        if p.name == "BASE_COMMIT" and (p.parent / "main.py").is_file():
            return p.parent
    for p in walk(INPUT):
        if p.suffix == ".zip" and zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                if "BASE_COMMIT" in z.namelist():
                    target = WORK / "base_unzipped"
                    shutil.rmtree(target, ignore_errors=True)
                    z.extractall(target)
                    return target
    raise SystemExit("STOP: không thấy code base (zip/folder có BASE_COMMIT). Add Input dataset từ dist/qgrsa-base-*.zip")


def prepare_code():
    base = find_base()
    commit = (base / "BASE_COMMIT").read_text().strip()
    if commit != PATCH["base"]:
        raise SystemExit("STOP: code base là commit {} nhưng wave cần {} → attach đúng version dataset base".format(
            commit[:7], PATCH["base"][:7]))
    shutil.rmtree(REPO, ignore_errors=True)
    shutil.copytree(base, REPO, copy_function=shutil.copyfile, ignore=shutil.ignore_patterns("__pycache__"))
    for path, expected in PATCH["base_sha"].items():  # the patch is made for exactly these originals
        target = REPO / path
        actual = sha256(target.read_bytes()) if target.is_file() else None
        if actual != expected:
            raise SystemExit("STOP: {} trong base khác bản patch được build từ đó".format(path))
    for path, blob in PATCH["files"].items():
        target = REPO / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(blob))
    for path in PATCH["deleted"]:
        (REPO / path).unlink(missing_ok=True)
    say("Code: base {} + patch {} file, xoá {} (build {})".format(commit[:7], len(PATCH["files"]), len(PATCH["deleted"]),
                                                               PATCH["built"]))
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))


# ----------------------------------------------------------------------------- model and data
def prepare_model():
    import torch
    cands = [p for p in walk(INPUT) if p.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors", ".npz"}
             and p.stat().st_size > 50 * 2 ** 20]
    if PLAN.get("model_file"):
        cands = [Path(PLAN["model_file"])]
    assert len(cands) == 1, "Cần đúng 1 file ViT-B/16 in21k trong input, thấy {} → điền PLAN['model_file']".format(cands)
    src = cands[0]
    if src.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(src))
    elif src.suffix == ".npz":
        from timm.models.vision_transformer import VisionTransformer, _load_weights
        vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0)
        _load_weights(vit, str(src))
        state = vit.state_dict()
    else:
        state = torch.load(src, map_location="cpu", weights_only=False)
        for k in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and isinstance(state.get(k), dict):
                state = state[k]
                break
    state = {k.removeprefix("module.").removeprefix("model.").removeprefix("vit."): v
             for k, v in state.items() if torch.is_tensor(v)}
    if "embeddings.patch_embeddings.projection.weight" in state:  # Hugging Face -> timm
        hf, state = state, {}
        for a, b in {"embeddings.cls_token": "cls_token", "embeddings.position_embeddings": "pos_embed",
                     "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
                     "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
                     "layernorm.weight": "norm.weight", "layernorm.bias": "norm.bias"}.items():
            state[b] = hf[a]
        ren = {"layernorm_before": "norm1", "layernorm_after": "norm2", "attention.output.dense": "attn.proj",
               "intermediate.dense": "mlp.fc1", "output.dense": "mlp.fc2"}
        for i in range(12):
            b = "encoder.layer.{}.".format(i)
            for s in ("weight", "bias"):
                for o, n in ren.items():
                    state["blocks.{}.{}.{}".format(i, n, s)] = hf[b + o + "." + s]
                state["blocks.{}.attn.qkv.{}".format(i, s)] = torch.cat(
                    [hf[b + "attention.attention.{}.{}".format(q, s)] for q in ("query", "key", "value")])
    state = {k: v for k, v in state.items() if not k.startswith(("head", "pre_logits", "classifier", "pooler"))}
    assert tuple(state["pos_embed"].shape) == (1, 197, 768), "Cần ViT-B/16 224 (định dạng timm, npz hoặc safetensors)"
    out = WORK / "vit_b16_224.pt"
    torch.save(state, out)
    say("Model:", src)
    return out


def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest, filter="data")
    return dest


def raw_200(dataset):
    pattern = re.compile(NAMES[dataset], re.I)

    def search(root):
        return [p for p in walk(root, dirs=True) if WORK not in p.parents and pattern.search(str(p))
                and sum(c.is_dir() for c in p.iterdir()) == 200]
    found = search(INPUT)
    if not found:
        archives = [p for p in walk(INPUT) if pattern.search(p.name) and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))]
        assert archives, "Không thấy {} (folder 200 lớp hoặc archive) → điền PLAN['data_sources']".format(dataset)
        found = [p for p in walk(extract(archives[0], WORK / (dataset + "_raw")), dirs=True)
                 if sum(c.is_dir() for c in p.iterdir()) == 200]
    presplit = {p.parent for p in found if p.name == "train" and p.parent / "test" in found}
    if len(presplit) == 1 and len(found) == 2:
        return presplit.pop()  # e.g. Kaggle zaphat206/imagenet-r
    assert len(found) == 1, "{}: cần đúng 1 folder 200 lớp, thấy {}".format(dataset, found)
    return found[0]


def split_80_20(raw, dataset):
    """ImageNet-A protocol of every round: 80/20 per class, seed 67 (symlinks)."""
    import numpy as np
    root = WORK / (dataset + "_split")
    if not (root / "test").is_dir():
        rng = np.random.default_rng(67)
        for cls in sorted(c for c in raw.iterdir() if c.is_dir()):
            files = sorted(f for f in cls.iterdir() if f.suffix.lower() in IMG)
            order = rng.permutation(len(files)).tolist()
            n_tr = min(max(1, round(len(files) * 0.8)), len(files) - 1)
            for split, idx in (("train", order[:n_tr]), ("test", order[n_tr:])):
                d = root / split / cls.name
                d.mkdir(parents=True, exist_ok=True)
                for i in idx:
                    (d / files[i].name).symlink_to(files[i].resolve())
    return root


def cifar_root():
    found = [p.parent for p in walk(INPUT, dirs=True) if p.name == "cifar-100-python" and (p / "train").is_file()]
    if not found:
        archives = [p for p in walk(INPUT) if "cifar" in p.name.lower() and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))]
        assert archives, "Không thấy cifar-100-python"
        found = [p.parent for p in walk(extract(archives[0], WORK / "cifar_raw"), dirs=True)
                 if p.name == "cifar-100-python" and (p / "train").is_file()]
    return found[0]


def cub_root():
    raw = [p for p in walk(INPUT, dirs=True) if (p / "train_test_split.txt").is_file() and (p / "images").is_dir()]
    if raw:
        from data.data import prepare_cub_split
        say("CUB split chính thức:", prepare_cub_split(str(raw[0]), str(WORK / "cub_split")))
        return WORK / "cub_split"
    pre = [p for p in walk(INPUT, dirs=True) if re.search(r"cub|bird", str(p), re.I) and (p / "train").is_dir()
           and (p / "test").is_dir()]
    assert len(pre) == 1, "CUB: cần bản gốc có train_test_split.txt hoặc đúng 1 bản train/+test/, thấy {}".format(pre)
    return pre[0]


def prepare_data():
    roots = {}
    for ds in sorted({ds for ds, _, _ in JOBS}):
        given = PLAN.get("data_sources", {}).get(ds)
        if given:
            src = Path(given)
        elif ds == "cifar224":
            src = cifar_root()
        elif ds == "cub":
            src = cub_root()
        else:
            src = raw_200(ds)
        roots[ds] = src if ds in ("cifar224", "cub") or (src / "test").is_dir() else split_80_20(src, ds)
        say("Data", ds, "→", roots[ds])
    return roots


# ----------------------------------------------------------------------------- jobs
def run_tests():
    for pattern in PLAN.get("tests", []):
        r = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", pattern],
                           capture_output=True, text=True)
        say("test", pattern, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
        if r.returncode:
            raise SystemExit("STOP: test lỗi\n" + r.stderr[-3000:])


def run_config(ds, variant):
    from utils.qbridge_experiment import merge_dict
    found = list(REPO.glob("exps/*/{}.json".format(variant)))
    assert len(found) == 1, "Overlay {}: thấy {}".format(variant, found)
    base = json.loads((REPO / "exps/rsiat" / "{}.json".format(ds)).read_text())
    return merge_dict(merge_dict(base, json.loads(found[0].read_text())), PLAN.get("overrides", {}))


def log_path(ds, variant, seed):
    cfg = run_config(ds, variant)
    init = 0 if cfg["init_cls"] == cfg["increment"] else cfg["init_cls"]
    return OUT / "logs/adapter" / ds / str(init) / str(cfg["increment"]) / "{}_{}_{}.log".format(
        cfg["prefix"], seed, cfg["convnet_type"])


def done(ds, variant, seed):
    cfg, p = run_config(ds, variant), log_path(ds, variant, seed)
    tasks = 1 + (N_CLASSES[ds] - cfg["init_cls"]) // cfg["increment"]
    return p.is_file() and any(l.count(",") == tasks - 1 for l in p.read_text().splitlines() if "CNN top1 curve" in l)


def start(job, roots, pretrained, threads):
    ds, variant, seed = job
    log_path(*job).unlink(missing_ok=True)
    cfg = run_config(ds, variant)
    cfg.update(seed=[seed], device=[0], data_root=str(roots[ds]), pretrained_path=str(pretrained),
               output_root=str(OUT), resume=False)
    name = "{}_{}_{}".format(*job)
    path = OUT / "configs" / (name + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    (OUT / "console").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", "-B", "main.py", "--config", str(path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump():
        with open(OUT / "console" / (name + ".txt"), "w") as log:
            for line in proc.stdout:
                log.write(line)
                if SHOW.search(line):
                    say("[{}]".format(name), line.rstrip()[-200:])
    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    say("START [{}]".format(name))
    return proc, thread


def run_all(roots, pretrained):
    queue = [j for j in JOBS if not done(*j)]
    parallel = int(PLAN.get("max_parallel", 3))
    threads = max(1, (os.cpu_count() or 1) // parallel // 2)
    running, failed, t0 = {}, [], time.time()
    say("Jobs: {} (bỏ qua {} đã xong), {} song song".format(len(queue), len(JOBS) - len(queue), parallel))
    while queue or running:
        while queue and len(running) < parallel:
            job = queue.pop(0)
            running[job] = start(job, roots, pretrained, threads)
            time.sleep(20)  # stagger start-up
        for job, (proc, thread) in list(running.items()):
            if proc.poll() is not None:
                thread.join()
                ok = proc.returncode == 0 and done(*job)
                say(("DONE " if ok else "FAILED ") + "[{}_{}_{}]".format(*job) + " sau {:.0f} phút".format((time.time() - t0) / 60))
                if not ok:
                    failed.append(job)
                del running[job]
        time.sleep(10)
    return failed


def summarize(failed):
    lines = ["===== SUMMARY {} | {} =====".format(PLAN["name"], time.strftime("%Y-%m-%d %H:%M"))]
    for job in JOBS:
        p = log_path(*job)
        last = [l.split("=> ")[-1] for l in p.read_text().splitlines() if "Average Accuracy (CNN)" in l][-1:] if p.is_file() else []
        lines.append("{}_{}_{}: {}".format(*job, last[0] if last else "KHÔNG CÓ KẾT QUẢ"))
    if failed:
        lines.append("FAILED: {} → xem runs/console/".format(failed))
    for script in PLAN.get("summaries", []):
        out = OUT / (Path(script).stem + ".md")
        r = subprocess.run([sys.executable, "-B", script, "--root", str(OUT / "logs"), "--output", str(out),
                            "--json", str(out.with_suffix(".json"))], capture_output=True, text=True)
        lines.append(out.read_text(encoding="utf-8") if r.returncode == 0 else "{} lỗi:\n{}".format(script, r.stderr[-2000:]))
    text = "\n".join(lines) + "\n"
    (OUT / "SUMMARY.txt").write_text(text, encoding="utf-8")
    say(text)


def package():
    import torch
    import timm
    pkg = WORK / "package"
    shutil.rmtree(pkg, ignore_errors=True)
    shutil.copytree(OUT, pkg, ignore=shutil.ignore_patterns("*.pkl", "*.pt", "ckpt"))
    (pkg / "environment.json").write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "timm": timm.__version__, "plan": PLAN, "jobs": JOBS, "base": PATCH["base"], "patch_built": PATCH["built"],
        "patch_files": sorted(PATCH["files"])}, indent=2))
    say("Kết quả:", shutil.make_archive(str(WORK / "results_{}".format(PLAN["name"])), "zip", pkg))


def main():
    check_gpu()
    prepare_code()
    run_tests()
    roots, pretrained = prepare_data(), prepare_model()
    failed = run_all(roots, pretrained)
    summarize(failed)
    package()
    shutil.rmtree(WORK / "base_unzipped", ignore_errors=True)


if __name__ == "__main__":
    main()
