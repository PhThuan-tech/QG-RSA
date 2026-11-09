"""Static checks for the uploadable offline Kaggle workflow."""
import ast
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


class QBridgeKaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "QBRIDGE_Kaggle.ipynb"
        cls.notebook = json.loads(cls.path.read_text(encoding="utf-8"))
        cls.source = "\n".join("".join(cell.get("source", [])) for cell in cls.notebook["cells"])

    def test_code_cells_parse_and_outputs_are_empty(self):
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]), filename="notebook_cell_{}".format(index))
                self.assertEqual(cell.get("outputs"), [])
                self.assertIsNone(cell.get("execution_count"))

    def test_offline_assets_and_dataset_switch_are_explicit(self):
        for marker in ("DATASET = \"imageneta\"", "cifar224", "REPO_ZIP", "MODEL_FILE",
                       "DATASET_SOURCE", "--no-index", "HF_HUB_OFFLINE", "safe_extract",
                       "raw_imageneta_roots", "make_deterministic_imageneta_split",
                       "DATA_SPLIT_SEED", "RAW_IMAGENETA_TRAIN_FRACTION",
                       "exps/qbridge/sources"):
            self.assertIn(marker, self.source)
        self.assertNotIn("ALLOW_PIP_INTERNET", self.source)
        self.assertNotIn("exps/adapter_", self.source)

    def test_base_artifacts_variants_and_reports_are_staged(self):
        for marker in ("max_tasks_per_run", "task_0.pkl", "qbridge_prepare.py", "calibrate",
                       "generate_qbridge_configs.py", "R5", "R6", "summarize_qbridge.py",
                       "PACKAGE_CHECKPOINTS"):
            self.assertIn(marker, self.source)

    def test_raw_imageneta_split_accepts_200_class_folders(self):
        dataset_cell = next("".join(cell["source"]) for cell in self.notebook["cells"]
                            if "def raw_imageneta_roots" in "".join(cell.get("source", [])))
        definitions = dataset_cell.split("scope = dataset_scope", 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/"imagenet-a"
            for index in range(200):
                class_dir = raw/("n%08d" % index)
                class_dir.mkdir(parents=True)
                (class_dir/"a.jpg").write_bytes(b"a")
                (class_dir/"b.jpg").write_bytes(b"b")
            metadata = root/"session"/"metadata"
            metadata.mkdir(parents=True)
            def write_json(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")
            namespace = {"Path": Path, "np": np, "shutil": shutil,
                         "SESSION_ROOT": root/"session", "DATA_SPLIT_SEED": 67,
                         "RAW_IMAGENETA_TRAIN_FRACTION": .8, "write_json": write_json}
            exec(definitions, namespace)
            self.assertEqual(namespace["raw_imageneta_roots"](root), [raw.resolve()])
            output = namespace["make_deterministic_imageneta_split"](raw)
            self.assertEqual(len(list((output/"train").iterdir())), 200)
            self.assertEqual(len(list((output/"test").iterdir())), 200)
            self.assertTrue((metadata/"imageneta_split.json").is_file())


if __name__ == "__main__":
    unittest.main()
