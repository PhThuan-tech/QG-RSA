import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from scripts.package_kaggle_repo import (
    FORBIDDEN_PARTS,
    FORBIDDEN_SUFFIXES,
    PROJECT_ROOT,
    REQUIRED_CONFIGS,
    REQUIRED_ROOT_FILES,
    _forbidden,
    build_manifest,
    collect_source_files,
    validate_source_files,
    verify_archive,
    write_archive,
)


class KagglePackageTests(unittest.TestCase):
    def test_whitelist_contains_required_source_and_no_runtime_artifacts(self):
        files = collect_source_files()
        names = {path.relative_to(PROJECT_ROOT).as_posix() for path in files}
        self.assertTrue((REQUIRED_ROOT_FILES | REQUIRED_CONFIGS) <= names)
        self.assertIn("scripts/package_kaggle_repo.py", names)
        self.assertIn("tests/test_package_kaggle_repo.py", names)
        for name in names:
            relative = PurePosixPath(name)
            self.assertIsNone(_forbidden(relative), name)
            self.assertFalse({part.lower() for part in relative.parts} & FORBIDDEN_PARTS)
            self.assertNotIn(Path(name).suffix.lower(), FORBIDDEN_SUFFIXES)

    def test_static_validation_and_verified_archive_round_trip(self):
        files = collect_source_files()
        validate_source_files(files)
        manifest = build_manifest(files)
        with tempfile.TemporaryDirectory(prefix=".qksr_test_package_", dir=PROJECT_ROOT) as directory:
            destination = Path(directory) / "source.zip"
            write_archive(files, destination, manifest)
            verify_archive(destination, files, manifest)
            self.assertGreater(destination.stat().st_size, 0)
            self.assertEqual(len(manifest["files"]), len(files))
            self.assertEqual(
                len({entry["path"] for entry in manifest["files"]}),
                len(files),
                "Duplicate source path in the minimal package",
            )

    def test_manifest_is_json_serializable(self):
        encoded = json.dumps(build_manifest(collect_source_files()), ensure_ascii=False)
        self.assertIn("QKSR_Kaggle_Ablations.ipynb", encoded)


if __name__ == "__main__":
    unittest.main()
