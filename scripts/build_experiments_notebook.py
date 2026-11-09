"""Build QGRSA_EXPERIMENTS.ipynb (the Kaggle notebook for every experiment round).

  python scripts/build_experiments_notebook.py QGRSA_EXPERIMENTS.ipynb
"""
import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# QG-RSA experiments — RSIAT offline (RTX PRO 6000 Blackwell)

Một notebook cho mọi vòng thí nghiệm. Chọn `PLAN` ở cell 1: mỗi plan là danh sách nhóm (dataset, seed, variant). Tên variant là tên file overlay `exps/<thí nghiệm>/<variant>.json`. Danh mục thí nghiệm và kết quả: `exps/README.md`.

| Vòng | Variant | Dataset | Mục đích |
|---|---|---|---|
| **SVF giai đoạn A (hiện tại)** | `SVF` | `imagenetr`, `imageneta` × 3 seed | view plastic (RSIAT) + view snapshot task 0 đóng băng, fusion có hiệu chỉnh, chỉ suy luận (`exps/svf/README.md`) |
| SCF vòng A3 | `SCF` | `cifar224`, `cub` theo `PLAN` | RSIAT + fusion thống kê–classifier với trọng số đã khoá, 3 seed 1993/2015/2026 (`exps/scf/README.md`) |
| QSD-CIL vòng A2 | `QSD2` | R + A (đã chạy: `exps/qsd/results/a2/`) | fusion có hiệu chỉnh (5 nguồn × 5 trọng số), trạng thái từ thống kê đã lưu (mean SSCA / mean cũ), oracle đối chứng, lưu điểm số (`exps/qsd/README_A2.md`) |
| QSD-CIL giai đoạn A | `QSD` | R + A (đã chạy: `exps/qsd/results/stage_a/`) | 9 head PGM/kênh/phân cấp/fusion trong cùng run (`exps/qsd/README.md`) |
| QRF-CA | `QRFCA` | `imageneta` (đã chạy: `exps/qrfca/results/`) | 6 head CA (linear, QDA, RFF, ReLU-RP, PQK, PQK không CNOT) trong cùng run (`exps/qrfca/README.md`) |
| Phát triển QKSR | `DEVELOP` | `imagenetr` (đã chạy: `exps/qksr/results/`) | `qksr_cur` so với `qksr_old`, 3 replicate mỗi bên (`exps/qksr/README.md`) |
| Cận trên | `UPPER` | `imagenetr` (đã chạy: `exps/upper/`) | oracle thống kê lớp cũ + huấn luyện chung 200 lớp (`exps/upper/README.md`) |
| Đo nhiễu | `MEASURE` | `imagenetr` (đã chạy: `exps/measurement/results/`) | 4 replicate baseline |
| Vòng 1 / 2 | `ROUND1` / `ROUND2` | `imageneta` (đã chạy: `exps/diagnostics/results/`) | SSCA ghép cặp, oracle, σ tự động, CA logit_norm / task margin |

**Inputs (không cần Internet):** (1) ZIP repo QG-RSA mới nhất; (2) dataset raw cho mỗi benchmark: `imagenet-r/<200 lớp>`, `imagenet-a/<200 lớp>`, hoặc `cifar-100-python/` (folder hoặc ZIP/TAR); (3) pretrained ViT-B/16 224 in21k; (4) *tuỳ chọn* wheel offline.

ImageNet-A được chia 80/20 theo lớp với seed 67; ImageNet-R dùng bản chia sẵn Kaggle `zaphat206/imagenet-r` — luôn dùng cùng bản này để ghép cặp với baseline đã chạy. Blackwell (sm_120) cần torch ≥ 2.7 với CUDA ≥ 12.8; cell 2 kiểm tra trước khi import torch.
""")

md("## 1. Config")
code(r'''
# Dataset: "imageneta" = input natural-adversarial-examples-imageneta (tự chia 80/20 seed 67); "imagenetr" = zaphat206/imagenet-r
MEASURE  = ["noise_rep0", "noise_rep1", "noise_rep2", "noise_rep3"]
ROUND1   = ["ssca_legacy_observed", "oracle_mean_cov", "ssca_paired", "noise_replicate", "oracle_mean"]
ROUND2   = ["ssca_legacy_observed", "noise_replicate", "noise_replicate2", "r2_sigma_paired",
            "r2_ca_taskmargin", "r2_ca_lognorm", "r2_sigma_legacy", "r2_combo"]
UPPER    = ["upper_oracle_rep0", "upper_oracle_rep1", "upper_joint"]
DEVELOP  = [f"{f}_rep{k}" for f in ("qksr_cur", "qksr_old") for k in range(3)]
QRFCA    = ["qrfca_rep0", "qrfca_rep1", "qrfca_rep2"]
QSD      = ["qsd_rep0", "qsd_rep1", "qsd_rep2"]
QSD2     = ["qsd2_rep0", "qsd2_rep1", "qsd2_rep2"]
SCF      = ["scf"]
SVF      = ["svf"]
SEEDS3   = [1993, 2015, 2026]   # quy ước chung cho mọi dataset (exps/README.md)
# Mỗi plan = danh sách nhóm (datasets, seeds, variants); job = tích của từng nhóm.
PLANS = {
    # SVF giai đoạn A: một tài khoản chạy svf_all; nếu chia 2 tài khoản thì dùng svf_r / svf_a.
    "svf_all": [(["imagenetr", "imageneta"], SEEDS3, SVF)],                # 6 run, ~3–4 giờ (ước tính)
    "svf_r":   [(["imagenetr"], SEEDS3, SVF)],                             # gồm run tuning R 1993
    "svf_a":   [(["imageneta"], SEEDS3, SVF)],
    # Vòng A3 (SCF, đã chạy), giữ để tái lập:
    "kaggle1": [(["cifar224"], [1993, 2015], SCF)],
    "kaggle2": [(["cifar224"], [2026], SCF), (["cub"], SEEDS3, SCF)],
    # Vòng A2 (đã chạy), giữ để tái lập:
    "a2_kaggle1": [(["imagenetr"], [1993], QSD2)],
    "a2_kaggle2": [(["imageneta"], [1993], QSD2), (["imagenetr"], [2042], ["qsd2_rep0"])],
    "a2_kaggle3": [(["imagenetr", "imageneta"], [2015, 2026], ["qsd2_rep0"])],
}
PLAN = "svf_all"
JOBS = [(ds, v, s) for datasets, seeds, variants in PLANS[PLAN] for ds in datasets for s in seeds for v in variants]
DATASETS = sorted({ds for ds, _, _ in JOBS})
SEEDS = sorted({s for _, _, s in JOBS})
VARIANTS = sorted({v for _, v, _ in JOBS})
OVERRIDES = {"save_checkpoints": False}  # KHÔNG đổi batch_size/epochs nếu muốn so với run cũ
MAX_PARALLEL = 3   # đo thực tế ImageNet-A: 6 song song ≈ 55 phút/run (nghẽn CPU), 2 ≈ 15 phút/run

INPUT_ROOT = ""     # rỗng = /kaggle/input nếu có, ngược lại thư mục hiện tại
WORK_ROOT  = ""     # rỗng = /kaggle/working nếu có, ngược lại ./qgrsa_work
REPO_ZIP   = ""     # rỗng = tự tìm repo/ZIP có scripts/run_level_stats.py
MODEL_FILE = ""     # rỗng = tự tìm .pt/.pth/.bin/.safetensors duy nhất
DATA_SOURCES = {"imageneta": "", "imagenetr": "", "cifar224": "", "cub": ""}  # rỗng = tự tìm theo tên
WHEEL_DIR  = ""     # rỗng = tự tìm thư mục chứa *.whl (chỉ dùng khi thiếu package)
SPLIT_SEED, TRAIN_FRACTION = 67, 0.8
DEVICE = 0
RUN_TESTS = True
''')

md(r"""
## 2. Môi trường offline + kiểm tra Blackwell

Probe torch trong **tiến trình con**. Nếu cell phải cài torch mới từ wheel thì **Restart kernel** rồi chạy lại từ cell 1.
""")
code(r'''
import os, sys, json, shutil, zipfile, tarfile, subprocess, re, importlib.util
from pathlib import Path

INPUT = Path(INPUT_ROOT) if INPUT_ROOT else (Path("/kaggle/input") if Path("/kaggle/input").is_dir() else Path.cwd())
WORK = Path(WORK_ROOT) if WORK_ROOT else (Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()/"qgrsa_work")
WORK.mkdir(parents=True, exist_ok=True)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TIMM_FUSED_ATTN="0")
print("INPUT:", INPUT, "| WORK:", WORK, "| Python", sys.version.split()[0])

def walk(root, dirs=False):
    """Files (or dirs) under root, following symlinks. Python 3.13's Path.rglob no
    longer enters symlinked directories, and Kaggle mounts some inputs as symlinks."""
    seen = set()
    for top, dnames, fnames in os.walk(root, followlinks=True):
        real = os.path.realpath(top)
        if real in seen:
            dnames[:] = []
            continue
        seen.add(real)
        for name in (dnames if dirs else fnames):
            yield Path(top)/name

def show_tree(root, depth=3, limit=60):
    """Print what is actually mounted, for error messages."""
    root, shown = Path(root), 0
    for p in sorted(walk(root, dirs=True)) + sorted(walk(root)):
        if len(p.relative_to(root).parts) <= depth and shown < limit:
            print("   ", p.relative_to(root), "/" if p.is_dir() else ""); shown += 1

wheel_dirs = [Path(WHEEL_DIR)] if WHEEL_DIR else sorted({p.parent for p in walk(INPUT) if p.suffix == ".whl"})
def pip_offline(*packages):
    assert wheel_dirs, f"Thiếu {packages} và không có thư mục wheel offline (WHEEL_DIR)"
    cmd = [sys.executable, "-m", "pip", "install", "--no-index"]
    for d in wheel_dirs: cmd += ["--find-links", str(d)]
    subprocess.run(cmd + list(packages), check=True)

PROBE = ("import torch, json; ok = torch.cuda.is_available(); "
         "cap = torch.cuda.get_device_capability(0) if ok else None; "
         "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'ok': ok, "
         "'name': torch.cuda.get_device_name(0) if ok else None, 'cap': cap, "
         "'arch': torch.cuda.get_arch_list()}))")
probe = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True)
assert probe.returncode == 0, "Không import được torch:\n" + probe.stderr[-2000:]
info = json.loads(probe.stdout.strip().splitlines()[-1])
print(info["name"], "| torch", info["torch"], "| CUDA", info["cuda"], "| arch", info["arch"])
assert info["ok"], "Không thấy GPU CUDA"
need = "sm_%d%d" % tuple(info["cap"])
if need not in info["arch"]:
    print(f"torch {info['torch']} KHÔNG có kernel cho {need} (Blackwell cần CUDA ≥ 12.8, torch ≥ 2.7).")
    pip_offline("torch", "torchvision")
    raise SystemExit("Đã cài torch mới từ wheel → Restart kernel rồi chạy lại từ cell 1.")

missing = [m for m in ("timm", "tqdm", "scipy") if importlib.util.find_spec(m) is None]
if missing:
    pip_offline(*missing)
print("Môi trường OK")
''')

code(r'''
import numpy as np, torch, torchvision, timm
torch.cuda.set_device(DEVICE)
# Phép thử thật trên GPU: bắt lỗi "no kernel image" ngay, không phải sau 20 phút train.
x = torch.randn(8, 3, 224, 224, device="cuda")
conv = torch.nn.Conv2d(3, 768, 16, 16).cuda()
torch.cuda.synchronize(); y = conv(x).flatten(2) @ torch.randn(196, 196, device="cuda"); torch.cuda.synchronize()
del x, y, conv; torch.cuda.empty_cache()

vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / 2**30
cpus = os.cpu_count() or 1
try:
    ram_gb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 2**20
except (OSError, StopIteration):
    ram_gb = float("nan")
auto = max(1, min(len(JOBS), int(vram_gb // 14), cpus // 6,
                  int(ram_gb // 10) if ram_gb == ram_gb else 99))
PARALLEL = auto if MAX_PARALLEL == "auto" else int(MAX_PARALLEL)
print(f"GPU {torch.cuda.get_device_name(DEVICE)} | VRAM {vram_gb:.0f} GB | CPU {cpus} | RAM khả dụng {ram_gb:.0f} GB")
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | timm {timm.__version__} | PARALLEL={PARALLEL}")
''')

md("## 3. Repo")
code(r'''
MARKER = "scripts/run_level_stats.py"
def find_repo(root, exclude=None):
    return next((p.parent for p in walk(root) if p.name == "main.py"
                 and (p.parent/MARKER).is_file() and not (exclude and Path(exclude) in p.parents)), None)

def find_repo_zip():
    for p in (q for q in walk(INPUT) if q.suffix == ".zip"):
        try:
            with zipfile.ZipFile(p) as z:
                if any(n.endswith(MARKER) for n in z.namelist()): yield p
        except zipfile.BadZipFile: pass

REPO = WORK/"repo"
shutil.rmtree(REPO, ignore_errors=True)
src = Path(REPO_ZIP) if REPO_ZIP else None
if src is not None and src.is_file():
    with zipfile.ZipFile(src) as z: z.extractall(REPO)
else:
    folder = find_repo(src or INPUT, exclude=WORK)   # nhiều nền tảng tự giải nén ZIP thành folder
    if folder is None:
        src = next(find_repo_zip(), None)
        assert src is not None, f"Không thấy repo có {MARKER} (ZIP cũ?) → upload dist/QG-RSA_QBRIDGE_Kaggle.zip mới"
        with zipfile.ZipFile(src) as z: z.extractall(REPO)
    else:   # input thường chỉ đọc → chép sang WORK
        shutil.copytree(folder, REPO, ignore=shutil.ignore_patterns("logs", "ckpt", ".git", "__pycache__", "results"))
REPO = find_repo(REPO)
os.chdir(REPO); sys.path.insert(0, str(REPO))
OUT = WORK/"runs"; OUT.mkdir(exist_ok=True)
print("Repo:", REPO)
''')

md("## 4. Pretrained ViT → state dict timm")
code(r'''
if MODEL_FILE: model_src = Path(MODEL_FILE)
else:
    # ViT-B/16 weights are ~330 MB; the size floor skips small .npz/.pt files (logs, predictions) in uploaded repos.
    cands = [p for p in walk(INPUT) if p.suffix.lower() in {".pt", ".pth", ".bin", ".safetensors", ".npz"}
             and WORK not in p.parents and p.stat().st_size > 50 * 2**20]
    if len(cands) != 1:
        print(f"Tìm thấy {len(cands)} file model: {cands[:10]}. Nội dung {INPUT}:"); show_tree(INPUT)
        raise AssertionError("Cần đúng 1 file model (.pt/.pth/.bin/.safetensors/.npz). Kiểm tra đã Add Input "
                             "model chưa; nếu có nhiều file hoặc file nằm trong ZIP thì điền MODEL_FILE.")
    model_src = cands[0]
if model_src.suffix == ".safetensors":
    from safetensors.torch import load_file; state = load_file(str(model_src))
elif model_src.suffix == ".npz":   # checkpoint gốc kiểu Google/JAX (vd. ViT-B_16.npz in21k)
    from timm.models.vision_transformer import VisionTransformer, _load_weights
    vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0)
    _load_weights(vit, str(model_src))
    state = vit.state_dict(); del vit
else:
    state = torch.load(model_src, map_location="cpu", weights_only=False)
    for k in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and isinstance(state.get(k), dict): state = state[k]; break
state = {k.removeprefix("module.").removeprefix("model.").removeprefix("vit."): v
         for k, v in state.items() if torch.is_tensor(v)}
if "embeddings.patch_embeddings.projection.weight" in state:   # Hugging Face → timm
    hf, state = state, {}
    for a, b in {"embeddings.cls_token": "cls_token", "embeddings.position_embeddings": "pos_embed",
                 "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
                 "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
                 "layernorm.weight": "norm.weight", "layernorm.bias": "norm.bias"}.items():
        state[b] = hf[a]
    ren = {"layernorm_before": "norm1", "layernorm_after": "norm2", "attention.output.dense": "attn.proj",
           "intermediate.dense": "mlp.fc1", "output.dense": "mlp.fc2"}
    for i in range(12):
        b = f"encoder.layer.{i}."
        for s in ("weight", "bias"):
            for o, n in ren.items(): state[f"blocks.{i}.{n}.{s}"] = hf[b + o + "." + s]
            state[f"blocks.{i}.attn.qkv.{s}"] = torch.cat([hf[b + f"attention.attention.{q}.{s}"]
                                                         for q in ("query", "key", "value")])
state = {k: v for k, v in state.items() if not k.startswith(("head", "pre_logits", "classifier", "pooler"))}
assert tuple(state["pos_embed"].shape) == (1, 197, 768), "Cần ViT-B/16 224"
PRETRAINED = WORK/"vit_b16_224.pt"; torch.save(state, PRETRAINED); del state
print("Pretrained:", model_src, "→", PRETRAINED)
''')

md(r"""
## 5. Dataset

ImageNet-A/R: tìm folder có đúng 200 lớp con, phân biệt bằng tên (`imagenet-a` / `imagenet-r`); nếu chỉ có ZIP/TAR thì giải nén vào WORK. Bản gốc chưa chia được chia 80/20 theo lớp với seed 67 (symlink, không chép ảnh); bản đã chia sẵn `train/` + `test/` cùng 200 lớp (vd. Kaggle `zaphat206/imagenet-r`, 23 918 / 6 082 ảnh) được dùng nguyên. CIFAR-100: folder `cifar-100-python/` hoặc archive của nó. CUB-200-2011: bản gốc (`images/`, `images.txt`, `train_test_split.txt`) được chia theo **split chính thức** 5 994 / 5 794 (symlink); nếu chỉ có bản đã chia sẵn `train/` + `test/` 200 lớp thì dùng nguyên.
""")
code(r'''
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {"imageneta": r"imagenet[-_]?a\b|imagenet-a|adversarial", "imagenetr": r"imagenet[-_]?r\b|imagenet-r|rendition"}

def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z: z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try: t.extractall(dest, filter="data")   # Python ≥ 3.12: chặn path traversal
            except TypeError: t.extractall(dest)
    return dest

def raw_200(dataset):
    pattern = re.compile(NAMES[dataset], re.I)
    def search(root, skip=None):
        return [p for p in walk(root, dirs=True) if not (skip and Path(skip) in p.parents)
                and pattern.search(str(p)) and sum(c.is_dir() for c in p.iterdir()) == 200]
    found = search(INPUT, skip=WORK) or search(WORK/f"{dataset}_raw")
    if not found:
        near = [(p, n) for p in walk(INPUT, dirs=True) if not (Path(WORK) in p.parents)
                and 100 <= (n := sum(c.is_dir() for c in p.iterdir())) <= 400]
        for p, n in near[:10]:
            print(f"  folder {p} có {n} lớp con (cần đúng 200, tên chứa '{dataset[-1]}' như imagenet-{dataset[-1]})")
        archives = [p for p in walk(INPUT) if pattern.search(p.name)
                    and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))]
        assert archives, f"Không thấy {dataset} (folder 200 lớp hoặc archive) → điền DATA_SOURCES['{dataset}']"
        extract(archives[0], WORK/f"{dataset}_raw")
        found = search(WORK/f"{dataset}_raw")
    # Bản chia sẵn <root>/train + <root>/test (vd. Kaggle zaphat206/imagenet-r) → trả về <root>.
    presplit = {p.parent for p in found if p.name == "train" and p.parent/"test" in found}
    if len(presplit) == 1 and len(found) == 2:
        return presplit.pop()
    assert len(found) == 1, f"{dataset}: cần đúng 1 folder 200 lớp, thấy {found} → điền DATA_SOURCES"
    return found[0]

def split_80_20(raw, dataset):
    root = WORK/f"{dataset}_split"
    if not (root/"test").is_dir():
        rng = np.random.default_rng(SPLIT_SEED)
        for cls in sorted(c for c in raw.iterdir() if c.is_dir()):
            files = sorted(f for f in cls.iterdir() if f.suffix.lower() in IMG)
            order = rng.permutation(len(files)).tolist()
            n_tr = min(max(1, round(len(files) * TRAIN_FRACTION)), len(files) - 1)
            for split, idx in (("train", order[:n_tr]), ("test", order[n_tr:])):
                d = root/split/cls.name; d.mkdir(parents=True, exist_ok=True)
                for i in idx:
                    try: (d/files[i].name).symlink_to(files[i].resolve())
                    except OSError: shutil.copy2(files[i], d/files[i].name)
    return root

def cifar_root():
    found = [p.parent for p in walk(INPUT, dirs=True) if p.name == "cifar-100-python" and (p/"train").is_file()]
    if not found:
        archives = [p for p in walk(INPUT) if "cifar" in p.name.lower()
                    and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))]
        assert archives, "Không thấy cifar-100-python → điền DATA_SOURCES['cifar224']"
        extract(archives[0], WORK/"cifar_raw")
        found = [p.parent for p in walk(WORK/"cifar_raw", dirs=True) if p.name == "cifar-100-python"
                 and (p/"train").is_file()]
    return found[0]

def cub_root():
    """CUB-200-2011: raw folder with the official split files (preferred), or a pre-split train/ + test/."""
    def scan(root):
        raw = [p for p in walk(root, dirs=True) if (p/"train_test_split.txt").is_file()
               and (p/"images.txt").is_file() and (p/"images").is_dir()]
        pre = [p for p in walk(root, dirs=True) if re.search(r"cub|bird", str(p), re.I) and (p/"train").is_dir()
               and (p/"test").is_dir() and sum(c.is_dir() for c in (p/"train").iterdir()) == 200]
        return raw, pre
    raw, pre = scan(INPUT)
    if not raw and not pre:
        archives = [p for p in walk(INPUT) if re.search(r"cub|bird", p.name, re.I)
                    and (zipfile.is_zipfile(p) or tarfile.is_tarfile(p))]
        assert archives, "Không thấy CUB-200-2011 (bản gốc có train_test_split.txt hoặc train/+test/) → điền DATA_SOURCES['cub']"
        extract(archives[0], WORK/"cub_raw")
        raw, pre = scan(WORK/"cub_raw")
    if raw:
        from data.data import prepare_cub_split
        counts = prepare_cub_split(str(raw[0]), str(WORK/"cub_split"))
        print("CUB: split chính thức từ", raw[0], counts)
        if counts != {"train": 5994, "test": 5794}:
            print("CẢNH BÁO: số ảnh khác split chính thức 5 994 / 5 794")
        return WORK/"cub_split"
    assert len(pre) == 1, f"CUB: thấy nhiều bản chia sẵn {pre} → điền DATA_SOURCES['cub']"
    return pre[0]

DATA_ROOTS = {}
for ds in DATASETS:
    if DATA_SOURCES.get(ds):
        src = Path(DATA_SOURCES[ds])
    elif ds == "cifar224":
        src = cifar_root()
    elif ds == "cub":
        src = cub_root()
    else:
        src = raw_200(ds)
    if ds != "cifar224" and (src/"test").is_dir():
        tr = sorted(c.name for c in (src/"train").iterdir() if c.is_dir())
        te = sorted(c.name for c in (src/"test").iterdir() if c.is_dir())
        assert len(tr) == 200 and tr == te, (
            f"{src} đã chia sẵn nhưng train {len(tr)} lớp / test {len(te)} lớp, lệch {sorted(set(tr) ^ set(te))[:5]}"
            " → dùng bản gốc chưa chia (<200 lớp>, 30 000 ảnh) để notebook tự chia 80/20 seed 67")
        print(ds, "dùng split có sẵn:", src)
    DATA_ROOTS[ds] = src if ds == "cifar224" or (src/"test").is_dir() else split_80_20(src, ds)
    if ds != "cifar224":
        count = lambda s: sum(1 for f in (DATA_ROOTS[ds]/s).rglob("*") if not f.is_dir())
        print(ds, DATA_ROOTS[ds], "| train", count("train"), "| test", count("test"))
    else:
        print(ds, DATA_ROOTS[ds])
''')

md("## 6. Test nhanh (CPU, ~15 giây)")
code(r'''
if RUN_TESTS:
    for pattern in ("test_stats_diagnostics.py", "test_run_level_stats.py", "test_ca_heads.py", "test_qsd.py", "test_scf.py", "test_svf.py"):
        r = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", pattern],
                           capture_output=True, text=True)
        print(pattern, r.stderr.strip().splitlines()[-1]); assert r.returncode == 0, r.stderr[-3000:]
''')

md(r"""
## 7. Chạy (song song)

Mỗi run = `exps/rsiat/<dataset>.json` + overlay + `OVERRIDES`, chạy trong một tiến trình riêng. Log đầy đủ nằm ở `runs/console/`. Run đã xong được bỏ qua, nên có thể chạy lại cell sau khi bị ngắt; một run lỗi không làm dừng các run khác.
""")
code(r'''
import threading, time
from utils.qbridge_experiment import merge_dict
SHOW = re.compile(r"STATS task|CNN top1 curve|Average Accuracy|Error|Traceback|WARNING")
N_CLASSES = {"imageneta": 200, "imagenetr": 200, "cifar224": 100, "cub": 200}

def overlay(variant):
    found = [p for p in REPO.glob(f"exps/*/{variant}.json")]
    assert len(found) == 1, f"Overlay {variant}: thấy {found}"
    return json.loads(found[0].read_text())

def base_config(ds):
    return json.loads((REPO/"exps/rsiat"/f"{ds}.json").read_text())

def run_config(ds, variant):
    return merge_dict(merge_dict(base_config(ds), overlay(variant)), OVERRIDES)

def log_path(ds, variant, seed):
    cfg = run_config(ds, variant)   # overlay có thể đổi init_cls/increment (vd. upper_joint)
    init = 0 if cfg["init_cls"] == cfg["increment"] else cfg["init_cls"]
    return (OUT/"logs/adapter"/ds/str(init)/str(cfg["increment"])
            / f"{overlay(variant)['prefix']}_{seed}_{cfg['convnet_type']}.log")

def done(ds, variant, seed):
    cfg, p = run_config(ds, variant), log_path(ds, variant, seed)
    tasks = 1 + (N_CLASSES[ds] - cfg["init_cls"]) // cfg["increment"]
    return p.is_file() and any(l.count(",") == tasks - 1 for l in p.read_text().splitlines() if "CNN top1 curve" in l)

def start(ds, variant, seed, threads):
    log_path(ds, variant, seed).unlink(missing_ok=True)   # bỏ log dở dang của lần trước
    cfg = run_config(ds, variant)
    cfg.update(seed=[seed], device=[DEVICE], data_root=str(DATA_ROOTS[ds]), pretrained_path=str(PRETRAINED),
               output_root=str(OUT), resume=False)
    name = f"{ds}_{variant}_{seed}"
    path = OUT/"configs"/f"{name}.json"; path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2))
    (OUT/"console").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
    proc = subprocess.Popen([sys.executable, "-u", "-B", "main.py", "--config", str(path)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    def pump():
        with open(OUT/"console"/f"{name}.txt", "w") as log:
            for line in proc.stdout:
                log.write(line)
                if SHOW.search(line): print(f"[{name}]", line.rstrip()[-200:], flush=True)
    thread = threading.Thread(target=pump, daemon=True); thread.start()
    print(f"START [{name}]", flush=True)
    return proc, thread

def run_all(parallel=PARALLEL):
    queue = [j for j in JOBS if not done(*j)]
    skipped = [j for j in JOBS if done(*j)]
    if skipped: print("SKIP (đã xong):", skipped)
    threads = max(1, (os.cpu_count() or 1) // max(1, parallel) // 2)
    running, failed, t0 = {}, [], time.time()
    while queue or running:
        while queue and len(running) < parallel:
            job = queue.pop(0); running[job] = start(*job, threads)
            time.sleep(20)   # lệch pha khởi động
        for job, (proc, thread) in list(running.items()):
            if proc.poll() is not None:
                thread.join()
                ok = proc.returncode == 0 and done(*job)
                print(("DONE " if ok else "FAILED ") + "[{}_{}_{}]".format(*job) + f" sau {(time.time()-t0)/60:.0f} phút", flush=True)
                if not ok: failed.append(job)
                del running[job]
        time.sleep(10)
    print("Hoàn tất." if not failed else f"Lỗi: {failed} → xem runs/console/<dataset>_<variant>_<seed>.txt")
    return failed
''')
code(r'''
failed = run_all()
''')

md(r"""
## 8. Tổng hợp

- **Nhiễu theo benchmark** (`scripts/run_level_stats.py noise`): SD giữa các replicate, số cặp run cần cho từng mức hiệu ứng, và benchmark sàng lọc được chọn theo quy tắc đặt trước.
- **Chẩn đoán SSCA/CA** (`summarize_stats_diagnostics.py`): khi chạy `ROUND1`/`ROUND2`/`UPPER`.
- `UPPER` được so cặp với baseline `noise` ở máy local (baseline nằm trong `exps/measurement/results/`), nên ở đây chỉ in AIA từng run.
""")
code(r'''
from IPython.display import Markdown, display
if any(v.startswith("noise_rep") for v in VARIANTS):
    r = subprocess.run([sys.executable, "-B", "scripts/run_level_stats.py", "noise", "--root", str(OUT/"logs"),
                        "--output", str(OUT/"noise.md"), "--json", str(OUT/"noise.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    display(Markdown((OUT/"noise.md").read_text(encoding="utf-8")))
for ds, v, s in JOBS:
    c = log_path(ds, v, s)
    if c.is_file():
        print(ds, v, s, [l.split("=> ")[-1] for l in c.read_text().splitlines() if "Average Accuracy (CNN)" in l][-1:])
for prefix, script in (("qsd_rep", "summarize_qsd.py"), ("qsd2", "summarize_qsd_a2.py")):
    if any(v.startswith(prefix) for v in VARIANTS):   # mỗi tài khoản chỉ có một phần run: bảng ở đây là tạm
        r = subprocess.run([sys.executable, "-B", f"scripts/{script}", "--root", str(OUT/"logs"),
                            "--output", str(OUT/"qsd_summary.md"), "--json", str(OUT/"qsd_summary.json")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        display(Markdown((OUT/"qsd_summary.md").read_text(encoding="utf-8")))
if "scf" in VARIANTS:   # mỗi tài khoản chỉ có một phần run: bảng ở đây là tạm
    r = subprocess.run([sys.executable, "-B", "scripts/summarize_scf.py", "--root", str(OUT/"logs"),
                        "--output", str(OUT/"scf_summary.md"), "--json", str(OUT/"scf_summary.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    display(Markdown((OUT/"scf_summary.md").read_text(encoding="utf-8")))
if "svf" in VARIANTS:   # chia nhiều tài khoản thì bảng ở đây là tạm
    r = subprocess.run([sys.executable, "-B", "scripts/summarize_svf.py", "--root", str(OUT/"logs"),
                        "--output", str(OUT/"svf_summary.md"), "--json", str(OUT/"svf_summary.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    display(Markdown((OUT/"svf_summary.md").read_text(encoding="utf-8")))
if any(v.startswith("qrfca") for v in VARIANTS):
    r = subprocess.run([sys.executable, "-B", "scripts/summarize_ca_heads.py", "--root", str(OUT/"logs"),
                        "--output", str(OUT/"qrfca_summary.md"), "--json", str(OUT/"qrfca_summary.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    display(Markdown((OUT/"qrfca_summary.md").read_text(encoding="utf-8")))
if any(v.startswith(("ssca", "oracle", "r2_", "noise_replicate", "upper_oracle")) for v in VARIANTS):
    for logs in sorted((OUT/"logs/adapter").glob("*/*/*")):
        r = subprocess.run([sys.executable, "-B", "scripts/summarize_stats_diagnostics.py", "--root", str(logs),
                            "--output", str(logs/"summary.md")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        display(Markdown((logs/"summary.md").read_text(encoding="utf-8")))
''')

md("## 9. Sandbox phân tích")
code(r'''
import ast
import matplotlib.pyplot as plt

def curve(log):
    lines = [l for l in Path(log).read_text().splitlines() if "CNN top1 curve" in l]
    return ast.literal_eval(lines[-1].split("CNN top1 curve: ")[1]) if lines else None

fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 3.5), squeeze=False)
for ax, ds in zip(axes[0], DATASETS):
    for _, v, s in (j for j in JOBS if j[0] == ds):
        c = curve(log_path(ds, v, s)) if log_path(ds, v, s).is_file() else None
        if c: ax.plot(range(len(c)), c, marker=".", label=f"{v}|{s}")
    ax.set_title(ds); ax.set_xlabel("task"); ax.set_ylabel("top-1"); ax.legend(fontsize=7)
plt.show()
''')

md("## 10. Đóng gói kết quả (không gồm checkpoint)")
code(r'''
pkg = WORK/"package"; shutil.rmtree(pkg, ignore_errors=True)
shutil.copytree(OUT, pkg, ignore=shutil.ignore_patterns("*.pkl", "*.pt", "ckpt"))
(pkg/"environment.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(DEVICE), "torch": torch.__version__,
    "cuda": torch.version.cuda, "timm": timm.__version__, "parallel": PARALLEL, "seeds": SEEDS,
    "plan": PLAN, "jobs": JOBS, "datasets": DATASETS, "variants": VARIANTS, "overrides": OVERRIDES}, indent=2))
print(shutil.make_archive(str(WORK/f"qgrsa_results_{PLAN}"), "zip", pkg))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    lines = c["source"].split("\n"); c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
