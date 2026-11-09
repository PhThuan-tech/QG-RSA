"""Build the committed offline Kaggle notebook without requiring nbformat."""
import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


cells = [
markdown("""# Q-BRIDGE v1.1 trên Kaggle — offline base → incremental

Notebook này nhận ba nhóm asset do bạn tự **Add Input**: ZIP repo QG-RSA, dataset CIFAR-100 hoặc ImageNet-A, và tensor pretrained ViT-B/16 224. Internet giữ **OFF**.

Luồng chạy: kiểm tra assets → train QKSR task 0 → fit artifact base-only nếu variant cần → chạy Q-BRIDGE từ đúng base checkpoint → tổng hợp và đóng gói. Đây là runner thí nghiệm; unit tests hay smoke không phải benchmark accuracy."""),
markdown("""## 1. Cấu hình duy nhất cần sửa

Nếu auto-discovery tìm nhiều file/folder, điền đường dẫn `/kaggle/input/...` tương ứng. `R5` là full raw Q-BRIDGE; `R6` là full calibrated Q-BRIDGE. Có thể chạy thêm R0–R10."""),
code(r'''# ===== USER CONFIG =====
DATASET = "imageneta"          # "imageneta" hoặc "cifar224"
SEEDS = [1993]                 # full study: [1993, 2015, 2026]
VARIANTS = ["R5", "R6"]       # full raw + full calibrated

REPO_ZIP = ""                  # repo ZIP/TAR hoặc repo folder; rỗng = auto-discover
MODEL_FILE = ""                # .safetensors/.pt/.pth/.bin ViT-B/16; rỗng = auto-discover
DATASET_SOURCE = ""            # dataset archive/folder; rỗng = auto-discover
WHEEL_SOURCE = ""              # folder/archive chứa wheel nếu Kaggle thiếu timm/safetensors
RAW_IMAGENETA_TRAIN_FRACTION = 0.8  # dùng khi ImageNet-A chỉ có 200 class folders
DATA_SPLIT_SEED = 67

DEVICE = "cuda:0"
RUN_SOURCE_TESTS = True
RUN_TRAINING = True
PACKAGE_CHECKPOINTS = False     # True có thể tạo ZIP rất lớn
BASE_CONFIG_OVERRIDES = {}      # cố định trước khi chạy; ví dụ {"batch_size": 24}
ARTIFACT_OPTIONS = {"calibration_steps": 500, "mlp_epochs": 10,
                    "mlp_batch_size": 64, "mlp_lr": 0.01}

if DATASET not in {"cifar224", "imageneta"}:
    raise ValueError("DATASET chỉ nhận cifar224 hoặc imageneta")
if not SEEDS or len(SEEDS) != len(set(SEEDS)):
    raise ValueError("SEEDS phải không rỗng và không lặp")
valid_variants = {f"R{i}" for i in range(11)}
if not VARIANTS or not set(VARIANTS) <= valid_variants:
    raise ValueError("VARIANTS chỉ nhận R0...R10")
'''),
markdown("""## 2. Session, GPU và helpers an toàn

Mỗi lần Run All tạo một session mới trong `/kaggle/working`; không ghi vào inputs chỉ đọc. Archive bị từ chối nếu có path traversal hoặc symlink."""),
code(r'''import os, sys, json, re, gc, stat, shutil, hashlib, zipfile, tarfile, subprocess, importlib
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from collections import deque

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
if not INPUT_ROOT.is_dir():
    raise RuntimeError("Notebook này dành cho Kaggle: không có /kaggle/input")
SESSION_ROOT = WORK_ROOT / ("qbridge_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
SESSION_ROOT.mkdir(parents=True, exist_ok=False)
for name in ("assets", "metadata", "logs", "base", "artifacts", "configs", "runs"):
    (SESSION_ROOT/name).mkdir()

import torch
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("Chưa có CUDA. Bật Settings → Accelerator → GPU rồi chạy lại.")
if DEVICE.startswith("cuda"):
    torch.cuda.set_device(int(DEVICE.split(":")[-1]))
    print("GPU:", torch.cuda.get_device_name(), "| torch:", torch.__version__)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def choose_one(candidates, label):
    candidates = sorted({Path(p).resolve() for p in candidates})
    if len(candidates) != 1:
        raise RuntimeError(f"{label}: cần đúng 1 candidate, tìm thấy {len(candidates)}\n" +
                           "\n".join(str(p) for p in candidates[:30]) +
                           "\nĐiền đường dẫn explicit trong USER CONFIG rồi chạy lại từ đầu.")
    return candidates[0]

def is_archive(path):
    path = Path(path)
    return path.is_file() and (zipfile.is_zipfile(path) or tarfile.is_tarfile(path))

def safe_extract(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def target(name):
        parts = PurePosixPath(name.replace("\\", "/"))
        if parts.is_absolute() or ".." in parts.parts or (parts.parts and ":" in parts.parts[0]):
            raise ValueError(f"Archive path không an toàn: {name}")
        result = destination.joinpath(*parts.parts).resolve()
        if not result.is_relative_to(destination):
            raise ValueError(f"Archive member nằm ngoài destination: {name}")
        return result
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for item in handle.infolist():
                output = target(item.filename)
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("ZIP symlink không được hỗ trợ: " + item.filename)
                if item.is_dir(): output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(item) as source, output.open("wb") as sink: shutil.copyfileobj(source, sink)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as handle:
            for item in handle:
                output = target(item.name)
                if item.isdir(): output.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with handle.extractfile(item) as source, output.open("wb") as sink: shutil.copyfileobj(source, sink)
                else: raise ValueError("TAR link/device không được hỗ trợ: " + item.name)
    else: raise ValueError("Không phải ZIP/TAR hỗ trợ: " + str(archive))
    return destination

def run_logged(command, name, cwd=None):
    log = SESSION_ROOT/"logs"/(name + ".log")
    tail = deque(maxlen=60)
    env = dict(os.environ); env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("w", encoding="utf-8") as output:
        output.write("COMMAND: " + json.dumps([str(x) for x in command]) + "\n")
        process = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            output.write(line); output.flush(); tail.append(line.rstrip())
            if line.startswith("QBRIDGE") or "Average" in line or "ERROR" in line:
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode:
        print("\n".join(tail))
        raise RuntimeError(f"Command failed ({returncode}); xem {log}")
    return log
'''),
markdown("""## 3. Giải nén và kiểm tra repo QG-RSA

Repo ZIP phải chứa implementation Q-BRIDGE v1.1, không phải repo `research` của notebook mẫu."""),
code(r'''REPO_MARKERS = ("main.py", "utils/qbridge.py", "scripts/qbridge_prepare.py")

def repo_roots(scope):
    scope = Path(scope).resolve()
    if not scope.is_dir(): return []
    candidates = [scope]
    candidates.extend(path.parent for path in scope.rglob("main.py") if path.is_file())
    return sorted({root.resolve() for root in candidates
                   if all((root/marker).is_file() for marker in REPO_MARKERS)})

def repo_diagnostic(scope):
    scope = Path(scope)
    nearby = []
    if scope.is_dir():
        for main_file in list(scope.rglob("main.py"))[:30]:
            root = main_file.parent
            missing = [marker for marker in REPO_MARKERS if not (root/marker).is_file()]
            nearby.append({"candidate":str(root), "missing":missing})
    return nearby

repo_request = Path(REPO_ZIP).expanduser() if REPO_ZIP else INPUT_ROOT
if not repo_request.exists(): raise FileNotFoundError(repo_request)
roots = repo_roots(repo_request) if repo_request.is_dir() else []
if len(roots) == 1:
    shutil.copytree(roots[0], SESSION_ROOT/"repository", ignore=shutil.ignore_patterns(
        ".git", "__pycache__", "*.pyc", "logs", "ckpt", "reports"))
elif len(roots) > 1:
    raise RuntimeError("Tìm thấy nhiều repo roots. Điền REPO_ZIP bằng đúng một path:\n" +
                       "\n".join(str(path) for path in roots))
else:
    archives = [repo_request] if repo_request.is_file() else [p for p in repo_request.rglob("*") if is_archive(p)]
    plausible = []
    for archive in archives:
        try:
            names = []
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as h: names = h.namelist()
            elif tarfile.is_tarfile(archive):
                with tarfile.open(archive, "r:*") as h: names = h.getnames()
            if any(n.endswith("utils/qbridge.py") for n in names) and any(n.endswith("main.py") for n in names):
                plausible.append(archive)
        except Exception: pass
    if not plausible:
        raise RuntimeError("Không nhận diện được repo Q-BRIDGE trong " + str(repo_request) +
                           ". Repo cần đủ " + str(REPO_MARKERS) +
                           ". Các main.py gần đó: " + json.dumps(repo_diagnostic(repo_request), ensure_ascii=False) +
                           ". Nếu repo đang mount dạng folder, đặt REPO_ZIP='/kaggle/input/<slug>/QG-RSA'.")
    archive = choose_one(plausible, "QG-RSA repo archive")
    safe_extract(archive, SESSION_ROOT/"repository")
REPO_ROOT = choose_one(repo_roots(SESSION_ROOT/"repository"), "QG-RSA repo root")
required = ["docs/QBRIDGE_Implementation_Spec_v1.1.md", "utils/qbridge_experiment.py",
            "scripts/generate_qbridge_configs.py", "scripts/summarize_qbridge.py",
            "exps/qbridge/sources/cifar224.json",
            "exps/qbridge/sources/imageneta.json"]
missing = [name for name in required if not (REPO_ROOT/name).is_file()]
if missing: raise RuntimeError("Repo ZIP thiếu: " + str(missing))
source_text = (REPO_ROOT/"utils/inc_net.py").read_text(encoding="utf-8")
if "pretrained_path" not in source_text or "_load_local_adapter_backbone" not in source_text:
    raise RuntimeError("Repo ZIP chưa có hỗ trợ pretrained tensor offline. Hãy ZIP checkout QG-RSA mới nhất.")
os.chdir(REPO_ROOT); sys.path.insert(0, str(REPO_ROOT))
print("Repo:", REPO_ROOT, "| SHA tree marker:", sha256_file(REPO_ROOT/"utils/qbridge.py")[:12])
'''),
markdown("""## 4. Dependencies offline

Notebook chỉ cài package còn thiếu từ wheel đã upload. Nó không dùng PyPI khi Internet OFF."""),
code(r'''required = {"timm": "timm", "numpy": "numpy", "scipy": "scipy", "tqdm": "tqdm"}
if Path(MODEL_FILE).suffix.lower() == ".safetensors" or (not MODEL_FILE and any(INPUT_ROOT.rglob("*.safetensors"))):
    required["safetensors"] = "safetensors"
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    wheel_scope = Path(WHEEL_SOURCE).expanduser() if WHEEL_SOURCE else INPUT_ROOT
    wheel_dirs = sorted({str(p.parent) for p in wheel_scope.rglob("*.whl")})
    if not wheel_dirs: raise RuntimeError("Thiếu package/wheel offline: " + str(missing))
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps"]
    for folder in wheel_dirs: command += ["--find-links", folder]
    subprocess.run(command + missing, check=True)
    importlib.invalidate_caches()
still_missing = [module for module in required if importlib.util.find_spec(module) is None]
if still_missing: raise RuntimeError("Vẫn thiếu module: " + str(still_missing))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TIMM_FUSED_ATTN="0")
import numpy as np, torchvision, timm
print("Dependencies OK | torchvision", torchvision.__version__, "| timm", timm.__version__)
'''),
markdown("""## 5. Chuẩn hóa tensor pretrained ViT

Nhận state dict timm hoặc Hugging Face ViT-B/16 224. Cell loại classifier head, kiểm tra kích thước và lưu state chuẩn cục bộ. Tensor này dùng để train QKSR base; Q-BRIDGE incremental sau đó strict-load full base checkpoint."""),
code(r'''MODEL_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin"}
request = Path(MODEL_FILE).expanduser() if MODEL_FILE else INPUT_ROOT
if not request.exists(): raise FileNotFoundError(request)
files = [request] if request.is_file() else [p for p in request.rglob("*") if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES]
# Avoid selecting experiment/checkpoint files embedded inside the repo archive extraction.
files = [p for p in files if not p.resolve().is_relative_to(SESSION_ROOT.resolve())]
model_source = choose_one(files, "pretrained ViT tensor")
if model_source.suffix.lower() == ".safetensors":
    from safetensors.torch import load_file
    state = load_file(str(model_source), device="cpu")
else:
    try: state = torch.load(model_source, map_location="cpu", weights_only=True)
    except TypeError: state = torch.load(model_source, map_location="cpu")
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict): state = state[key]; break
if not isinstance(state, dict): raise ValueError("Model file không chứa state_dict")
state = {k.removeprefix("module.").removeprefix("model."): v for k,v in state.items() if torch.is_tensor(v)}
hf = {k.removeprefix("vit."): v for k,v in state.items()}
if "embeddings.patch_embeddings.projection.weight" in hf:
    mapped = {}
    direct = {"embeddings.cls_token":"cls_token", "embeddings.position_embeddings":"pos_embed",
              "embeddings.patch_embeddings.projection.weight":"patch_embed.proj.weight",
              "embeddings.patch_embeddings.projection.bias":"patch_embed.proj.bias",
              "layernorm.weight":"norm.weight", "layernorm.bias":"norm.bias"}
    for source,dest in direct.items():
        if source in hf: mapped[dest] = hf[source]
    replacements = {"layernorm_before":"norm1", "layernorm_after":"norm2",
                    "attention.output.dense":"attn.proj", "intermediate.dense":"mlp.fc1",
                    "output.dense":"mlp.fc2"}
    layers = sorted({int(k.split(".")[2]) for k in hf if k.startswith("encoder.layer.")})
    for i in layers:
        base = f"encoder.layer.{i}."
        for old,new in replacements.items():
            for suffix in ("weight","bias"):
                if base+old+"."+suffix in hf: mapped[f"blocks.{i}.{new}.{suffix}"] = hf[base+old+"."+suffix]
        for suffix in ("weight","bias"):
            tensors = [hf.get(base+f"attention.attention.{name}.{suffix}") for name in ("query","key","value")]
            if all(t is not None for t in tensors): mapped[f"blocks.{i}.attn.qkv.{suffix}"] = torch.cat(tensors)
    state = mapped
state = {k:v for k,v in state.items() if not k.startswith(("head.","head_dist.","pre_logits.","classifier.","pooler."))}
required_keys = ["patch_embed.proj.weight", "cls_token", "pos_embed", "norm.weight",
                 "blocks.0.attn.qkv.weight", "blocks.11.mlp.fc2.weight"]
missing = [k for k in required_keys if k not in state]
if missing: raise RuntimeError("Không phải ViT-B/16 timm/HF tương thích; thiếu " + str(missing))
if tuple(state["patch_embed.proj.weight"].shape) != (768,3,16,16) or tuple(state["pos_embed"].shape) != (1,197,768):
    raise RuntimeError("Tensor phải là ViT-B/16 224 (embed=768, 197 position tokens)")
PRETRAINED = SESSION_ROOT/"assets/vit_b16_224_state.pt"
torch.save(state, PRETRAINED); del state, hf; gc.collect()
print("Model:", model_source, "→", PRETRAINED, "| sha256", sha256_file(PRETRAINED)[:16])
'''),
markdown("""## 6. Chuẩn bị dataset

CIFAR cần format gốc `cifar-100-python/{train,test,meta}`. ImageNet-A nhận cả hai dạng: `train/<class>` + `test/<class>`, hoặc dataset chuẩn chỉ có `imagenet-a/<class>`. Với dạng raw, notebook tạo split 80/20 xác định trong working directory và ghi rõ protocol/seed vào manifest."""),
code(r'''def dataset_scope(request, label):
    request = Path(request).expanduser() if request else INPUT_ROOT
    if not request.exists(): raise FileNotFoundError(request)
    if request.is_file(): return safe_extract(request, SESSION_ROOT/"assets"/label)
    return request

def cifar_roots(scope):
    found = []
    if (scope/"cifar-100-python/train").is_file(): found.append(scope)
    if scope.name == "cifar-100-python" and (scope/"train").is_file(): found.append(scope.parent)
    found += [p.parent for p in scope.rglob("cifar-100-python") if (p/"train").is_file() and (p/"test").is_file()]
    return found

def imageneta_roots(scope):
    found = []
    if (scope/"train").is_dir() and (scope/"test").is_dir(): found.append(scope)
    found += [p.parent for p in scope.rglob("train") if p.is_dir() and (p.parent/"test").is_dir()]
    return found

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def raw_imageneta_roots(scope):
    scope = Path(scope)
    candidates = [scope]
    candidates.extend(path for path in scope.rglob("*") if path.is_dir()
                      and path.name.lower().replace("_", "-") == "imagenet-a")
    valid = []
    for root in candidates:
        try:
            classes = [path for path in root.iterdir() if path.is_dir()]
            if len(classes) == 200 and all(any(file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
                                                for file in class_dir.iterdir()) for class_dir in classes):
                valid.append(root.resolve())
        except (OSError, PermissionError):
            pass
    return sorted(set(valid))

def make_deterministic_imageneta_split(raw_root):
    if not 0 < float(RAW_IMAGENETA_TRAIN_FRACTION) < 1:
        raise ValueError("RAW_IMAGENETA_TRAIN_FRACTION phải nằm trong (0,1)")
    output = SESSION_ROOT/"assets/imageneta_split"
    generator = np.random.default_rng(int(DATA_SPLIT_SEED))
    counts = {}
    for class_dir in sorted(path for path in Path(raw_root).iterdir() if path.is_dir()):
        files = sorted(path for path in class_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(files) < 2:
            raise RuntimeError(f"Lớp {class_dir.name} cần ít nhất 2 ảnh để chia train/test")
        order = generator.permutation(len(files)).tolist()
        n_train = min(max(1, int(round(len(files)*RAW_IMAGENETA_TRAIN_FRACTION))), len(files)-1)
        split_indices = {"train":order[:n_train], "test":order[n_train:]}
        counts[class_dir.name] = {name:len(indices) for name,indices in split_indices.items()}
        for split, indices in split_indices.items():
            destination = output/split/class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for index in indices:
                source = files[index].resolve()
                target = destination/source.name
                try: target.symlink_to(source)
                except OSError: shutil.copy2(source, target)
    write_json(SESSION_ROOT/"metadata/imageneta_split.json", {
        "source":str(Path(raw_root).resolve()), "destination":str(output),
        "method":"per-class deterministic random split", "seed":int(DATA_SPLIT_SEED),
        "train_fraction":float(RAW_IMAGENETA_TRAIN_FRACTION), "counts":counts})
    return output

scope = dataset_scope(DATASET_SOURCE, DATASET)
if DATASET == "cifar224":
    roots = cifar_roots(scope)
    if not roots and not DATASET_SOURCE:
        archives = [p for p in INPUT_ROOT.rglob("*") if is_archive(p) and "cifar" in p.name.lower()]
        scope = safe_extract(choose_one(archives, "CIFAR archive"), SESSION_ROOT/"assets/cifar224")
        roots = cifar_roots(scope)
    DATA_ROOT = choose_one(roots, "CIFAR-100 root")
    train = torchvision.datasets.CIFAR100(str(DATA_ROOT), train=True, download=False)
    test = torchvision.datasets.CIFAR100(str(DATA_ROOT), train=False, download=False)
    dataset_manifest = {"dataset":DATASET, "protocol":"official CIFAR-100 train/test",
                        "train":len(train), "test":len(test), "classes":100}
else:
    roots = imageneta_roots(scope)
    if not roots and not DATASET_SOURCE:
        archives = [p for p in INPUT_ROOT.rglob("*") if is_archive(p) and "imagenet" in p.name.lower()]
        if archives:
            scope = safe_extract(choose_one(archives, "ImageNet-A archive"), SESSION_ROOT/"assets/imageneta")
            roots = imageneta_roots(scope)
    roots = [p for p in roots if len([x for x in (p/"train").iterdir() if x.is_dir()]) == 200]
    split_protocol = "provided fixed train/test"
    if roots:
        DATA_ROOT = choose_one(roots, "ImageNet-A train/test root")
    else:
        raw_roots = raw_imageneta_roots(scope)
        if not raw_roots and scope != INPUT_ROOT:
            raw_roots = raw_imageneta_roots(INPUT_ROOT)
        raw_root = choose_one(raw_roots, "raw ImageNet-A root with 200 class folders")
        DATA_ROOT = make_deterministic_imageneta_split(raw_root)
        split_protocol = (f"derived per-class split train={RAW_IMAGENETA_TRAIN_FRACTION:.3f}; "
                          f"seed={DATA_SPLIT_SEED}")
    train = torchvision.datasets.ImageFolder(str(DATA_ROOT/"train"))
    test = torchvision.datasets.ImageFolder(str(DATA_ROOT/"test"))
    if len(train.classes) != 200 or train.class_to_idx != test.class_to_idx:
        raise RuntimeError("ImageNet-A cần 200 lớp và mapping train/test giống nhau")
    dataset_manifest = {"dataset":DATASET, "protocol":split_protocol,
                        "train":len(train), "test":len(test), "classes":200}
del train, test
dataset_manifest.update(data_root=str(DATA_ROOT), source=str(scope), source_tree_unhashed=True)
write_json(SESSION_ROOT/"metadata/dataset.json", dataset_manifest)
print(json.dumps(dataset_manifest, indent=2, ensure_ascii=False))
'''),
markdown("""## 7. Khóa config base

Task 0 dùng QKSR/PQK và tensor pretrained offline. Mỗi seed tạo một base checkpoint riêng. `max_tasks_per_run=1` chỉ áp dụng cho bước base; incremental config sau đó đặt lại `null`."""),
code(r'''base_template = json.loads((REPO_ROOT/"exps/qbridge/sources"/f"{DATASET}.json").read_text(encoding="utf-8"))
base_template.update(BASE_CONFIG_OVERRIDES)
base_template.update({
    "prefix":"qbridge_base", "dataset":DATASET, "device":[int(DEVICE.split(":")[-1])],
    "data_root":str(DATA_ROOT), "offline":True, "pretrained_path":str(PRETRAINED),
    "output_root":str(SESSION_ROOT/"base"), "save_checkpoints":True, "isolate_runs":True,
    "keep_last_checkpoint":True, "compact_diagonal_checkpoint":False,
    "resume":False, "max_tasks_per_run":1,
    "use_quantum_kernel_base":True, "use_quantum_kernel_inc":False,
    "q_kernel_type":"pqk", "q_kernel_order":1, "q_order2_weight":1.0,
    "q_num_qubits":8, "q_num_layers":2, "q_reupload":False, "q_dtype":"float32",
    "q_gamma_mode":"bounded_learned", "q_calib_samples":512, "q_init_seed":1234,
    "q_metric_lr_mult":0.1, "q_inc_train_mode":"frozen", "q_inc_pair":"old_proj"
})
if BASE_CONFIG_OVERRIDES.get("use_quantum_kernel_base") is False:
    raise ValueError("Q-BRIDGE yêu cầu QKSR base: không được tắt use_quantum_kernel_base")
config_paths = {}
for seed in SEEDS:
    cfg = dict(base_template); cfg["seed"] = [seed]
    path = SESSION_ROOT/"configs"/f"{DATASET}_base_seed{seed}.json"
    write_json(path, cfg); config_paths[seed] = path
write_json(SESSION_ROOT/"metadata/experiment_plan.json", {
    "dataset":DATASET, "seeds":SEEDS, "variants":VARIANTS, "offline":True,
    "pretrained_sha256":sha256_file(PRETRAINED), "base_overrides":BASE_CONFIG_OVERRIDES,
    "artifact_options":ARTIFACT_OPTIONS, "benchmark_status":"not_started"})
print("Base configs:", config_paths)
'''),
markdown("""## 8. Source tests

Chạy fixture CPU trước khi dùng GPU nhiều giờ. Test pass chỉ xác nhận implementation, không xác nhận CIFAR/ImageNet-A accuracy."""),
code(r'''if RUN_SOURCE_TESTS:
    run_logged([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
               "source_tests", cwd=REPO_ROOT)
    write_json(SESSION_ROOT/"metadata/source_tests.json", {"status":"passed", "scope":"repository fixtures; not benchmark"})
    print("Source tests: PASSED")
else:
    write_json(SESSION_ROOT/"metadata/source_tests.json", {"status":"skipped_by_user"})
    print("Source tests: skipped")
'''),
markdown("""## 9. Train QKSR base — task 0

Cell này train base từ pretrained ViT cho từng seed. Checkpoint phải chứa PQK, gamma, class means và full covariance; cell kiểm tra ngay sau khi train."""),
code(r'''BASE_CHECKPOINTS = {}
if RUN_TRAINING:
    for seed in SEEDS:
        run_logged([sys.executable, "-u", "-B", "main.py", "--config", str(config_paths[seed])],
                   f"base_{DATASET}_{seed}", cwd=REPO_ROOT)
        cfg = json.loads(config_paths[seed].read_text())
        checkpoint = (SESSION_ROOT/"base/ckpt/qbridge_base"/DATASET/
                      f"{cfg['init_cls']}_{cfg['increment']}"/f"seed_{seed}/task_0.pkl")
        if not checkpoint.is_file(): raise RuntimeError("Không tìm thấy base checkpoint: " + str(checkpoint))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"model_state_dict","quantum_kernel_state_dict","class_means","class_covs",
                    "class_order","run_metadata","known_classes","total_classes"}
        if required-set(state): raise RuntimeError("Base checkpoint thiếu " + str(required-set(state)))
        if state["cur_task"] != 0 or state["class_covs"].shape[-2:] != (768,768):
            raise RuntimeError("Base checkpoint không phải completed task 0/full covariance")
        BASE_CHECKPOINTS[seed] = checkpoint
        print(f"BASE OK seed={seed} sha={sha256_file(checkpoint)[:16]}")
    write_json(SESSION_ROOT/"metadata/base_checkpoints.json",
               {str(k):{"path":str(v),"sha256":sha256_file(v)} for k,v in BASE_CHECKPOINTS.items()})
else:
    print("RUN_TRAINING=False: chưa tạo base checkpoint")
'''),
markdown("""## 10. Fit base-only artifacts cho R6/R7/R8

R6 cần calibrator qua gate; R7 cần RBF selector; R8 cần MLP selector. Mỗi artifact bị ràng buộc bởi SHA-256 của base checkpoint cùng seed. Không dùng test/incremental data để fit."""),
code(r'''if RUN_TRAINING:
    need_cache = bool(set(VARIANTS) & {"R6","R7","R8"})
    for seed, base in BASE_CHECKPOINTS.items():
        folder = SESSION_ROOT/"artifacts"/DATASET/str(seed); folder.mkdir(parents=True, exist_ok=True)
        common = ["--config",str(config_paths[seed]),"--base",str(base),"--seed",str(seed),"--device",DEVICE]
        cache = folder/"features.pt"
        if need_cache:
            run_logged([sys.executable,"-u","-B","scripts/qbridge_prepare.py","cache",*common,
                        "--data-root",str(DATA_ROOT),"--output",str(cache)], f"cache_{DATASET}_{seed}", REPO_ROOT)
        if "R6" in VARIANTS:
            run_logged([sys.executable,"-u","-B","scripts/qbridge_prepare.py","calibrate",*common,
                        "--cache",str(cache),"--steps",str(ARTIFACT_OPTIONS["calibration_steps"]),
                        "--output",str(folder/"calibrator.pt")], f"calibrator_{DATASET}_{seed}", REPO_ROOT)
        if "R7" in VARIANTS:
            run_logged([sys.executable,"-u","-B","scripts/qbridge_prepare.py","rbf",*common,
                        "--cache",str(cache),"--output",str(folder/"rbf.pt")], f"rbf_{DATASET}_{seed}", REPO_ROOT)
        if "R8" in VARIANTS:
            run_logged([sys.executable,"-u","-B","scripts/qbridge_prepare.py","mlp",*common,
                        "--cache",str(cache),"--epochs",str(ARTIFACT_OPTIONS["mlp_epochs"]),
                        "--batch-size",str(ARTIFACT_OPTIONS["mlp_batch_size"]),
                        "--lr",str(ARTIFACT_OPTIONS["mlp_lr"]),"--output",str(folder/"mlp.pt")],
                       f"mlp_{DATASET}_{seed}", REPO_ROOT)
    print("Artifacts prepared:", SESSION_ROOT/"artifacts")
'''),
markdown("""## 11. Sinh config khóa và chạy incremental

Mỗi variant/seed khởi động lại từ đúng base checkpoint. Output đã tồn tại sẽ bị từ chối để tránh trộn run. Chạy notebook mới nếu muốn lặp lại."""),
code(r'''RUN_CONFIGS = {}
if RUN_TRAINING:
    for seed, base in BASE_CHECKPOINTS.items():
        source = json.loads(config_paths[seed].read_text())
        source.update({"seed":[seed], "max_tasks_per_run":None,
                       "output_root":str(SESSION_ROOT/"runs"), "pretrained_path":str(PRETRAINED)})
        source_path = SESSION_ROOT/"configs"/f"{DATASET}_incremental_source_seed{seed}.json"
        write_json(source_path, source)
        output = SESSION_ROOT/"configs"/f"seed_{seed}"
        run_logged([sys.executable,"-u","-B","scripts/generate_qbridge_configs.py",
                    "--source",str(source_path),"--output-dir",str(output),"--base",str(base),
                    "--artifact-root",str(SESSION_ROOT/"artifacts"),"--seeds",str(seed)],
                   f"configs_{DATASET}_{seed}", REPO_ROOT)
        for variant in VARIANTS: RUN_CONFIGS[(seed,variant)] = output/f"{variant}.json"
    write_json(SESSION_ROOT/"metadata/run_configs.json",
               {f"{seed}|{variant}":{"path":str(path),"sha256":sha256_file(path)}
                for (seed,variant),path in RUN_CONFIGS.items()})

    for seed in SEEDS:
        for variant in VARIANTS:
            print(f"START {DATASET} seed={seed} variant={variant}")
            run_logged([sys.executable,"-u","-B","main.py","--config",str(RUN_CONFIGS[(seed,variant)])],
                       f"run_{DATASET}_{seed}_{variant}", REPO_ROOT)
            print(f"DONE  {DATASET} seed={seed} variant={variant}")
'''),
markdown("""## 12. Tổng hợp measured runs

Chỉ run `completed` được tổng hợp; partial/failed không được điền metric. Paired contrast phải cùng base hash và cùng training/data budget."""),
code(r'''SUMMARY = SESSION_ROOT/"qbridge_summary.json"
if RUN_TRAINING:
    run_logged([sys.executable,"-u","-B","scripts/summarize_qbridge.py",
                "--root",str(SESSION_ROOT/"runs"),"--output",str(SUMMARY)],
               "summarize", REPO_ROOT)
    summary = json.loads(SUMMARY.read_text())
    print("Status:", summary["status"], "| completed runs:", len(summary["runs"]))
    for run in summary["runs"]:
        print(run["dataset"], run["seed"], run["variant"],
              "AIA=",run["aia"],"final=",run["final_top1"],"forgetting=",run["forgetting"])
    for contrast in summary["paired_contrasts"]:
        print(contrast["dataset"], contrast["contrast"], contrast["summary"])
'''),
markdown("""## 13. Đóng gói kết quả

ZIP mặc định chứa configs, logs, JSON reports và artifact metadata; bỏ tensor/checkpoint lớn. Base và final checkpoint vẫn nằm trong session để bạn Save Version/Download riêng. Bật `PACKAGE_CHECKPOINTS=True` nếu muốn đưa chúng vào ZIP."""),
code(r'''package_root = SESSION_ROOT/"package"
package_root.mkdir()
for name in ("metadata","logs","configs"):
    shutil.copytree(SESSION_ROOT/name, package_root/name)
if (SESSION_ROOT/"qbridge_summary.json").is_file():
    shutil.copy2(SESSION_ROOT/"qbridge_summary.json", package_root/"qbridge_summary.json")
for report in (SESSION_ROOT/"runs").rglob("*.json"):
    target = package_root/"run_reports"/report.relative_to(SESSION_ROOT/"runs")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(report, target)
for report in (SESSION_ROOT/"artifacts").rglob("*.json"):
    target = package_root/"artifact_reports"/report.relative_to(SESSION_ROOT/"artifacts")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(report, target)
if PACKAGE_CHECKPOINTS:
    for root in (SESSION_ROOT/"base", SESSION_ROOT/"runs", SESSION_ROOT/"artifacts"):
        for binary in root.rglob("*"):
            if binary.is_file() and binary.suffix.lower() in {".pt",".pth",".pkl",".safetensors"}:
                target = package_root/"binaries"/binary.relative_to(SESSION_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(binary, target)
write_json(package_root/"PACKAGE_MANIFEST.json", {
    "created_utc":datetime.now(timezone.utc).isoformat(), "session":str(SESSION_ROOT),
    "dataset":DATASET, "seeds":SEEDS, "variants":VARIANTS,
    "includes_checkpoints":PACKAGE_CHECKPOINTS,
    "claim":"Measured run artifacts only; tests are not benchmark evidence"})
archive = Path(shutil.make_archive(str(WORK_ROOT/f"qbridge_{DATASET}_results"), "zip", package_root))
print("Results ZIP:", archive, "|", round(archive.stat().st_size/1024/1024,2), "MiB")
print("Session (Save Version để giữ checkpoints):", SESSION_ROOT)
try:
    from IPython.display import display, FileLink
    display(FileLink(str(archive)))
except Exception: pass
''')]

notebook = {"cells": cells,
            "metadata": {"accelerator": "GPU", "kaggle": {"accelerator": "gpu", "internet": False},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3"}},
            "nbformat": 4, "nbformat_minor": 5}

output = Path(__file__).resolve().parents[1] / "QBRIDGE_Kaggle.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(output)
