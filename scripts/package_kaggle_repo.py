"""Validate and create a minimal, auditable QG-RSA ZIP for Kaggle Input."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "QG-RSA"

REQUIRED_ROOT_FILES = {
    "main.py",
    "trainer.py",
    "requirements-colab.txt",
    "QKSR_Kaggle_Ablations.ipynb",
    "QKSR_Kaggle_ImageNetR.ipynb",
}
OPTIONAL_ROOT_FILES = {"README.md", "LICENSE", ".gitignore"}
INCLUDED_DIRECTORIES = {
    "data",
    "docs",
    "exps",
    "models",
    "network",
    "scripts",
    "tests",
    "utils",
}
REQUIRED_CONFIGS = {
    "exps/adapter_imageneta.json",
    "exps/adapter_imagenetr.json",
    "exps/adapter_cifar224.json",
}

# These are never valid source-package members, even if accidentally placed
# under a whitelisted directory.
FORBIDDEN_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "datasets", "logs", "ckpt",
    "checkpoints", "analysis_outputs", "dist", "artifacts", ".ipynb_checkpoints",
}
FORBIDDEN_SUFFIXES = {
    ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".safetensors",
    ".npy", ".npz", ".log", ".zip", ".tar", ".gz", ".7z",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}
FORBIDDEN_NAMES = {".env", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_PACKAGE_BYTES = 50 * 1024 * 1024


def relative_posix(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _forbidden(relative: PurePosixPath) -> str | None:
    lowered_parts = {part.lower() for part in relative.parts}
    if lowered_parts & FORBIDDEN_PARTS:
        return "forbidden directory"
    if relative.name.lower() in FORBIDDEN_NAMES:
        return "credential-like filename"
    if Path(relative.name).suffix.lower() in FORBIDDEN_SUFFIXES:
        return "binary/data/output suffix"
    return None


def collect_source_files() -> list[Path]:
    """Return a deterministic whitelist; reject unsafe or oversized members."""
    missing = sorted(name for name in REQUIRED_ROOT_FILES if not (PROJECT_ROOT / name).is_file())
    if missing:
        raise RuntimeError("Missing required root files: {}".format(missing))

    candidates = [PROJECT_ROOT / name for name in REQUIRED_ROOT_FILES | OPTIONAL_ROOT_FILES]
    for directory_name in sorted(INCLUDED_DIRECTORIES):
        directory = PROJECT_ROOT / directory_name
        if not directory.is_dir():
            raise RuntimeError("Missing required source directory: {}".format(directory_name))
        candidates.extend(path for path in directory.rglob("*") if path.is_file())

    selected: list[Path] = []
    total_bytes = 0
    for path in sorted(set(candidates), key=lambda item: relative_posix(item)):
        relative = PurePosixPath(relative_posix(path))
        if _forbidden(relative):
            continue
        if path.is_symlink():
            raise RuntimeError("Source package must not contain symlink: {}".format(relative))
        resolved = path.resolve()
        if PROJECT_ROOT.resolve() not in resolved.parents:
            raise RuntimeError("Package member escapes project root: {}".format(relative))
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise RuntimeError("Unexpectedly large source file ({} bytes): {}".format(size, relative))
        total_bytes += size
        selected.append(path)

    if total_bytes > MAX_PACKAGE_BYTES:
        raise RuntimeError("Selected source is too large: {} bytes".format(total_bytes))
    selected_names = {relative_posix(path) for path in selected}
    missing_configs = sorted(REQUIRED_CONFIGS - selected_names)
    if missing_configs:
        raise RuntimeError("Missing dataset configs: {}".format(missing_configs))
    return selected


def _compile_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError("Invalid notebook structure: {}".format(relative_posix(path)))
    ids = []
    for index, cell in enumerate(notebook["cells"]):
        cell_id = cell.get("id")
        if not cell_id or cell_id in ids:
            raise ValueError("Missing/duplicate cell id in {} at {}".format(relative_posix(path), index))
        ids.append(cell_id)
        if cell.get("cell_type") != "code":
            continue
        source_lines = []
        for line in "".join(cell.get("source", [])).splitlines():
            source_lines.append("pass" if line.lstrip().startswith("%") else line)
        compile("\n".join(source_lines), "{}:{}".format(path.name, cell_id), "exec")


def validate_source_files(files: list[Path]) -> None:
    """Parse every source/config/notebook before expensive runtime tests."""
    names = {relative_posix(path) for path in files}
    for required in REQUIRED_ROOT_FILES | REQUIRED_CONFIGS:
        if required not in names:
            raise RuntimeError("Required member was not selected: {}".format(required))
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=relative_posix(path))
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix == ".ipynb":
            _compile_notebook(path)


def run_test_suite() -> None:
    """Tests must pass before an archive is written."""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable, "-B", "-m", "unittest", "discover",
        "-s", "tests", "-p", "test_*.py", "-q",
    ]
    print("Running validation tests:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(files: list[Path]) -> dict:
    return {
        "format_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "validation": {
            "python_ast": True,
            "json_and_notebooks": True,
            "unittest_discovery": "passed",
        },
        "files": [
            {
                "path": relative_posix(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def _zip_info(member: str) -> zipfile.ZipInfo:
    # Stable metadata gives the same ZIP bytes for the same source bytes.
    info = zipfile.ZipInfo(member, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def write_archive(files: list[Path], output: Path, manifest: dict) -> None:
    output = output.resolve()
    root = PROJECT_ROOT.resolve()
    if output == root or root not in output.parents:
        raise ValueError("Output ZIP must stay inside the project workspace.")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must use a .zip filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        prefix=".qksr_package_", suffix=".zip", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in files:
                member = "{}/{}".format(ARCHIVE_ROOT, relative_posix(path))
                archive.writestr(_zip_info(member), path.read_bytes(), compresslevel=9)
            manifest_member = "{}/_KAGGLE_PACKAGE_MANIFEST.json".format(ARCHIVE_ROOT)
            archive.writestr(_zip_info(manifest_member), manifest_bytes, compresslevel=9)
        verify_archive(temporary_path, files, manifest)
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_archive(archive_path: Path, files: list[Path], manifest: dict) -> None:
    expected = {"{}/{}".format(ARCHIVE_ROOT, relative_posix(path)) for path in files}
    manifest_name = "{}/_KAGGLE_PACKAGE_MANIFEST.json".format(ARCHIVE_ROOT)
    expected.add(manifest_name)
    manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP contains duplicate member names.")
        if set(names) != expected:
            raise RuntimeError("ZIP member list differs from validated source selection.")
        damaged = archive.testzip()
        if damaged is not None:
            raise RuntimeError("ZIP CRC failure: {}".format(damaged))
        embedded = json.loads(archive.read(manifest_name).decode("utf-8"))
        if embedded != manifest:
            raise RuntimeError("Embedded manifest differs from validated manifest.")
        for relative, expected_entry in manifest_by_path.items():
            content = archive.read("{}/{}".format(ARCHIVE_ROOT, relative))
            if len(content) != expected_entry["size"]:
                raise RuntimeError("Size mismatch after ZIP write: {}".format(relative))
            if hashlib.sha256(content).hexdigest() != expected_entry["sha256"]:
                raise RuntimeError("SHA-256 mismatch after ZIP write: {}".format(relative))
            if _forbidden(PurePosixPath(relative)):
                raise RuntimeError("Forbidden file entered archive: {}".format(relative))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "artifacts" / "QG-RSA_Kaggle.zip",
        help="ZIP destination inside this workspace.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Run all validation and tests without writing a ZIP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = collect_source_files()
    print("Selected {} source files ({:.2f} MiB).".format(
        len(files), sum(path.stat().st_size for path in files) / (1024 ** 2)
    ))
    validate_source_files(files)
    print("Static validation passed.")
    run_test_suite()
    if args.check_only:
        print("Check-only completed; no archive written.")
        return
    manifest = build_manifest(files)
    write_archive(files, args.output, manifest)
    output = args.output.resolve()
    print("Archive verified:", output)
    print("Files:", len(files), "+ manifest")
    print("ZIP size: {:.2f} MiB".format(output.stat().st_size / (1024 ** 2)))
    print("ZIP SHA-256:", sha256_file(output))


if __name__ == "__main__":
    main()
