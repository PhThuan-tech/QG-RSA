"""Local-only asset discovery and strict ViT backbone loading (no downloads)."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import socket
import tarfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(paths, description):
    paths = sorted({Path(p).resolve() for p in paths})
    if len(paths) != 1:
        raise ValueError("Expected one {} (found {}); specify its path explicitly: {}".format(
            description, len(paths), [str(p) for p in paths],
        ))
    return paths[0]


def resolve_data_root(dataset, input_root, explicit=None):
    """CIFAR root is the parent of cifar-100-python; ImageNet requires train/test."""
    search = Path(explicit or input_root)
    if not search.is_dir():
        raise FileNotFoundError("Missing local dataset directory: {}".format(search))
    if dataset == "cifar224":
        markers = [search] if search.name == "cifar-100-python" else []
        markers += list(search.rglob("cifar-100-python"))
        roots = [p.parent for p in markers if all((p / f).is_file() for f in ("train", "test", "meta"))]
        return _unique(roots, "extracted CIFAR-100 root")
    if dataset not in {"imageneta", "imagenetr"}:
        raise ValueError("Supported datasets: cifar224, imageneta, imagenetr.")
    roots = [search] + [p.parent for p in search.rglob("train") if p.is_dir()]
    roots = [p for p in roots if (p / "train").is_dir() and (p / "test").is_dir()]
    return _unique(roots, "ImageFolder train/test root")


def resolve_weights(input_root, explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError("Missing local pretrained weights: {}".format(path))
        return path
    root = Path(input_root)
    candidates = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {
        ".pth", ".pt", ".bin", ".safetensors", ".npz",
    }]
    return _unique(candidates, "pretrained weight file")


def extract_cifar_archive(archive, destination):
    """Extract the uploaded official archive safely into a NEW writable directory."""
    destination = Path(destination).resolve()
    with tarfile.open(archive, "r:*") as bundle:
        members = bundle.getmembers()
        names = {m.name.rstrip("/") for m in members}
        expected = {"cifar-100-python/" + name for name in ("train", "test", "meta")}
        if not expected.issubset(names):
            raise ValueError("Archive is not the official CIFAR-100 Python layout.")
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe archive member: {}".format(member.name))
        destination.mkdir(parents=True, exist_ok=False)
        # Already validated paths and disallowed links/devices. Copy only bytes;
        # do not apply archive permissions/ownership to the Kaggle workspace.
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
    return destination


@contextmanager
def no_network():
    """Fail closed for IP connections; allow local Unix IPC for DataLoader workers."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Network connections are disabled for this offline study.")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Network connections are disabled for this offline study.")
        return original_connect_ex(sock, address)

    def create(*args, **kwargs):
        raise RuntimeError("Network connections are disabled for this offline study.")

    socket.socket.connect, socket.socket.connect_ex, socket.create_connection = connect, connect_ex, create
    try:
        yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original_connect, original_connect_ex, original_create


def read_vit_state(path, model_name):
    """Support timm/HF state dictionaries, safetensors and local Google ViT NPZ."""
    import torch

    path = Path(path)
    if path.suffix.lower() == ".npz":
        import timm
        model = timm.create_model(model_name, pretrained=False, num_classes=0)
        model.load_pretrained(str(path))
        state = model.state_dict()
    elif path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model"):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
    if not isinstance(state, dict) or not state:
        raise ValueError("Pretrained file must contain a non-empty ViT state dictionary.")
    normalized = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Invalid tensor state entry in pretrained file.")
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith(("head.", "head_dist.", "pre_logits.")):
            continue
        if key in normalized:
            raise ValueError("Duplicate normalized pretrained key: {}".format(key))
        normalized[key] = value
    return normalized


def convert_vit_state(state):
    """Keep the original timm fused-QKV and MLP conversion for the adapter ViT."""
    converted = {}
    for key, value in state.items():
        if key.endswith(("qkv.weight", "qkv.bias")):
            if value.shape[0] != 3 * 768:
                raise ValueError("Expected ViT-B/16 fused QKV size 2304.")
            for projection, tensor in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                converted[key.replace("qkv", projection)] = tensor
        else:
            converted[key.replace("mlp.fc", "fc")] = value
    return converted


def load_adapter_pretrained(model, model_name, path=None):
    if path is None:
        import timm
        state = timm.create_model(model_name, pretrained=True, num_classes=0).state_dict()
    else:
        state = read_vit_state(path, model_name)
    state = convert_vit_state(state)
    if path is not None:
        own = model.state_dict()
        # Only newly initialized adapter parameters may be absent. Silently
        # missing backbone weights would turn an offline run into another model.
        required = {key for key in own if ".adaptmlp." not in key}
        missing = required - state.keys()
        unexpected = state.keys() - own.keys()
        mismatched = [key for key in state.keys() & own.keys() if state[key].shape != own[key].shape]
        if missing or unexpected or mismatched:
            raise ValueError("Incompatible pretrained backbone: missing={}, unexpected={}, shape_mismatch={}".format(
                sorted(missing), sorted(unexpected), sorted(mismatched),
            ))
        if any(".adaptmlp." in key for key in state):
            raise ValueError("Expected backbone-only weights, not a trained adapter checkpoint.")
        import torch
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Pretrained backbone contains non-finite weights.")
        import torch
        if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Pretrained backbone contains non-finite weights.")
    message = model.load_state_dict(state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name in message.missing_keys
    return model
