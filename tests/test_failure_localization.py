"""Offline contracts and synthetic runner checks; no benchmark results."""

import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import socket
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

from data.data import iCIFAR224, iImageNetR
from scripts import run_failure_localization as runner
from scripts.collect_failure_localization import collect
from scripts.package_failure_localization import package
from utils.failure_localization import SEEDS, VARIANTS, build_configs, validate_configs, forgetting, TaskRecorder, task_line, write_json
from utils.offline_assets import resolve_data_root, resolve_weights, extract_cifar_archive, no_network, convert_vit_state, read_vit_state, load_adapter_pretrained
from test_experimental_integrity import make_manager, make_learner, small_autoencoder


class FailureLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def source(self):
        return json.loads((runner.REPO_ROOT / "exps/adapter_cifar224.json").read_text())

    def test_configs_change_only_flags_and_operational_policy(self):
        source = self.source()
        specification = json.loads((runner.REPO_ROOT / "exps/failure_localization_cifar224.json").read_text())
        self.assertEqual(specification["seeds"], list(SEEDS))
        self.assertEqual({key: (value["use_quantum_kernel_base"], value["use_quantum_kernel_inc"])
                          for key, value in specification["variants"].items()}, VARIANTS)
        configs = build_configs(source)
        validate_configs(configs)
        self.assertEqual({(c["prefix"], c["seed"][0]) for c in configs}, {(v, s) for v in VARIANTS for s in SEEDS})
        operational = {"seed", "prefix", "use_quantum_kernel_base", "use_quantum_kernel_inc",
                       "save_checkpoints", "resume", "keep_last_checkpoint", "compact_diagonal_checkpoint"}
        for config in configs:
            for key, value in source.items():
                if key not in operational:
                    self.assertEqual(config[key], value, key)
        self.assertTrue(all(c["checkpoint_policy"] == "final_only" for c in configs))
        self.assertTrue(all(c["q_inc_train_mode"] == "frozen" and c["q_inc_pair"] == "old_proj" for c in configs))

    def test_rejects_budget_flags_duplicates_and_tuning_changes(self):
        for mutate in (lambda c: c[1].update(init_epochs=11),
                       lambda c: c[1].update(use_quantum_kernel_inc=True),
                       lambda c: c.__setitem__(1, copy.deepcopy(c[0]))):
            configs = build_configs(self.source())
            mutate(configs)
            with self.assertRaises(ValueError):
                validate_configs(configs)
        for override in ({"resume": True}, {"val_ratio": 0.2}, {"max_tasks_per_run": 2}, {"q_kernel_type": "rbf_proj"}):
            with self.assertRaises(ValueError):
                build_configs({**self.source(), **override})

    def test_offline_cifar_never_requests_download(self):
        fake = type("Dataset", (), {"data": np.zeros((2, 2, 2, 3), dtype=np.uint8), "targets": [0, 1]})()
        dataset = iCIFAR224()
        dataset.data_root, dataset.offline = "/uploaded/cifar", True
        with patch("data.data.datasets.cifar.CIFAR100", return_value=fake) as constructor:
            dataset.download_data()
        self.assertEqual(constructor.call_count, 2)
        self.assertEqual([call.kwargs["download"] for call in constructor.call_args_list], [False, False])
        self.assertTrue(all(call.args[0] == "/uploaded/cifar" for call in constructor.call_args_list))

    def test_missing_cifar_fails_without_download_fallback(self):
        with tempfile.TemporaryDirectory() as directory, no_network():
            dataset = iCIFAR224()
            dataset.data_root, dataset.offline = directory, True
            with patch("data.data.datasets.cifar.CIFAR100.download", side_effect=AssertionError("download")), \
                 self.assertRaisesRegex(RuntimeError, "not found or corrupted"):
                dataset.download_data()

    def test_imagenet_rejects_mismatched_class_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            for split, classes in (("train", ("a", "b")), ("test", ("a", "c"))):
                for name in classes:
                    folder = Path(root) / split / name
                    folder.mkdir(parents=True)
                    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(folder / "image.png")
            dataset = iImageNetR()
            dataset.data_root = root
            with self.assertRaisesRegex(ValueError, "mappings differ"):
                dataset.download_data()

    def test_discovers_assets_without_upload_folder_names_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            parent = root / "arbitrary-upload" / "nested"
            marker = parent / "cifar-100-python"
            marker.mkdir(parents=True)
            for name in ("train", "test", "meta"):
                (marker / name).write_bytes(b"fixture")
            weights = root / "other-upload" / "pytorch_model.bin"
            weights.parent.mkdir()
            weights.write_bytes(b"fixture")
            self.assertEqual(resolve_data_root("cifar224", root), parent)
            self.assertEqual(resolve_data_root("cifar224", root, marker), parent)
            self.assertEqual(resolve_weights(root), weights)
            (weights.parent / "second.pth").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "found 2"):
                resolve_weights(root)
            self.assertEqual(resolve_weights(root, weights), weights)
            with self.assertRaises(FileNotFoundError):
                resolve_weights(root, root / "missing.pt")

    def test_cifar_archive_safe_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / "data.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for name in ("train", "test", "meta"):
                    member = tarfile.TarInfo("cifar-100-python/" + name)
                    member.size = 3
                    stream.addfile(member, io.BytesIO(b"abc"))
            extract_cifar_archive(archive, root / "extracted")
            self.assertEqual((root / "extracted/cifar-100-python/train").read_bytes(), b"abc")
            with self.assertRaises(FileExistsError):
                extract_cifar_archive(archive, root / "extracted")
            unsafe = root / "unsafe.tar"
            with tarfile.open(archive, "r:gz") as source, tarfile.open(unsafe, "w") as target:
                for member in source.getmembers():
                    target.addfile(member, source.extractfile(member))
                escape = tarfile.TarInfo("../escape")
                escape.size = 1
                target.addfile(escape, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                extract_cifar_archive(unsafe, root / "unsafe")
            self.assertFalse((root / "unsafe").exists())

    def test_network_guard_blocks_ip_and_restores_even_on_error(self):
        original = socket.create_connection
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            with no_network():
                socket.create_connection(("example.invalid", 443))
        self.assertIs(socket.create_connection, original)
        with no_network(), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            with self.assertRaises(RuntimeError):
                connection.connect(("127.0.0.1", 1))
            with self.assertRaises(RuntimeError):
                connection.connect_ex(("127.0.0.1", 1))

    def test_weight_conversion_matches_original_qkv_and_mlp_mapping(self):
        qkv = torch.arange(2304 * 2).reshape(2304, 2)
        state = {"blocks.0.attn.qkv.weight": qkv, "blocks.0.attn.qkv.bias": torch.arange(2304),
                 "blocks.0.mlp.fc1.weight": torch.ones(2, 2)}
        converted = convert_vit_state(state)
        for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
            self.assertTrue(torch.equal(converted["blocks.0.attn." + projection + ".weight"], qkv[index * 768:(index + 1) * 768]))
        self.assertIn("blocks.0.fc1.weight", converted)
        self.assertIn("blocks.0.attn.qkv.weight", state)
        with self.assertRaises(ValueError):
            convert_vit_state({"qkv.weight": torch.zeros(3, 2)})

    def test_local_backbone_loading_is_strict_and_preserves_freeze_policy(self):
        from network.vision_transformer_adapter import VisionTransformer
        from timm.models.vision_transformer import VisionTransformer as TimmViT
        tuning = SimpleNamespace(ffn_adapt=True, ffn_option="parallel", ffn_adapter_layernorm_option="none",
                          ffn_adapter_init_option="lora", ffn_adapter_scalar="0.1", ffn_num=4,
                          d_model=768, vpt_on=False, vpt_num=0)
        options = dict(img_size=224, patch_size=16, embed_dim=768, depth=1, num_heads=12, num_classes=0, qkv_bias=True)
        with redirect_stdout(io.StringIO()):
            local = VisionTransformer(**options, tuning_config=tuning)
            reference = copy.deepcopy(local)
        state = TimmViT(**options).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"state_dict": {"module." + k: v for k, v in state.items()}}, path)
            with no_network(), patch("timm.create_model", side_effect=AssertionError("download route")):
                load_adapter_pretrained(local, "vit_base_patch16_224_in21k", path)
            with patch("timm.create_model") as create:
                create.return_value.state_dict.return_value = state
                load_adapter_pretrained(reference, "vit_base_patch16_224_in21k")
            for key, value in local.state_dict().items():
                self.assertTrue(torch.equal(value, reference.state_dict()[key]), key)
            for name, parameter in local.named_parameters():
                self.assertEqual(parameter.requires_grad, ".adaptmlp." in name, name)
            local.eval(), reference.eval()
            with torch.no_grad():
                image = torch.ones(1, 3, 224, 224)
                self.assertTrue(torch.equal(local(image), reference(image)))
            broken = {k: v for k, v in state.items() if k != "cls_token"}
            torch.save(broken, path)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_adapter_pretrained(local, "vit_base_patch16_224_in21k", path)
            torch.save({**state, "blocks.0.adaptmlp.down_proj.weight": torch.ones(4, 768)}, path)
            with self.assertRaisesRegex(ValueError, "backbone-only"):
                load_adapter_pretrained(local, "vit_base_patch16_224_in21k", path)
            bad_values = dict(state)
            bad_values["cls_token"] = torch.full_like(state["cls_token"], float("nan"))
            torch.save(bad_values, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_adapter_pretrained(local, "vit_base_patch16_224_in21k", path)

    def test_npz_loader_uses_local_path_without_pretrained_download(self):
        with patch("timm.create_model") as create:
            create.return_value.state_dict.return_value = {"cls_token": torch.zeros(1, 1, 768)}
            read_vit_state("/uploaded/google.npz", "vit_base_patch16_224_in21k")
            create.assert_called_once_with("vit_base_patch16_224_in21k", pretrained=False, num_classes=0)
            create.return_value.load_pretrained.assert_called_once_with(str(Path("/uploaded/google.npz")))

    def test_preflight_refuses_gpu_training_without_gpu(self):
        with patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "Kaggle GPU"):
            runner.preflight(build_configs(self.source()), require_gpu=True)

    def test_preparation_selection_and_preflight_bindings(self):
        def preflight_fixture(configs, require_gpu=False):
            validate_configs(configs)
            return {"config_sha256": {c["prefix"] + "_" + str(c["seed"][0]): runner.config_digest(c) for c in configs},
                    "weights_sha256": "fixture", "dataset_sha256": "fixture", "task_sizes": [10] * 10,
                    "class_orders": {str(s): list(range(100)) for s in SEEDS}, "versions": {"fixture": "unit-only"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "input/any-name/cifar-100-python"
            marker.mkdir(parents=True)
            for name in ("train", "test", "meta"):
                (marker / name).write_bytes(b"temporary test fixture")
            weights = root / "input/weights/weights.bin"
            weights.parent.mkdir()
            weights.write_bytes(b"temporary test fixture")
            args = SimpleNamespace(source=None, dataset="cifar224", input_root=root / "input", output_root=root / "working",
                                   data_root=None, pretrained_path=None, cifar_archive=None)
            with patch.object(runner, "preflight", side_effect=preflight_fixture), redirect_stdout(io.StringIO()):
                study = runner.prepare(args)
                self.assertEqual(len(list((study / "configs").glob("*.json"))), 9)
                with patch.object(runner, "execute_run") as execute:
                    runner.run_study(study, ["A1"], [2015])
                    self.assertEqual(execute.call_count, 1)
                    self.assertEqual(execute.call_args.args[0]["prefix"], "A1")
                    self.assertEqual(execute.call_args.args[0]["seed"], [2015])
                destination = study / "runs/A1_2015"
                destination.mkdir(parents=True)
                with self.assertRaises(FileExistsError):
                    runner.run_study(study, ["A1"], [2015])
                path = study / "configs/A0_1993.json"
                config = json.loads(path.read_text())
                config["init_epochs"] += 1
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "differ beyond"):
                    runner.run_study(study, ["A0"], [1993])
                with patch.object(runner, "source_fingerprints", return_value={"changed": "source"}), \
                     self.assertRaisesRegex(ValueError, "Source code changed"):
                    runner.run_study(study, ["A0"], [1993])

    def test_no_offline_download_route_for_unsupported_backbones(self):
        from utils.inc_net import get_convnet
        with self.assertRaisesRegex(ValueError, "adapter ViT"):
            get_convnet({"offline": True, "convnet_type": "vit_base_patch16_224"})
        with self.assertRaisesRegex(ValueError, "pretrained_path"):
            get_convnet({"offline": True, "convnet_type": "pretrained_vit_b16_224_in21k_adapter"})

    def test_metric_formulas_and_csv_json_are_consistent(self):
        self.assertIsNone(forgetting([[80]]))
        self.assertAlmostEqual(forgetting([[80], [75, 90], [70, 85, 95]]), 7.5)
        with tempfile.TemporaryDirectory() as directory:
            recorder = TaskRecorder(directory, "A0", 1993)
            config = {"init_epochs": 10, "inc_epochs": 30, "ca_epochs": 10, "ca": True}
            base = {"top5": 98., "grouped": {"total": 80., "old": 0., "new": 80.}, "task_accuracies": [80.]}
            first = recorder.add(0, config, base, None, {"adapter": 2.}, 3.)
            self.assertIsNone(first["pre_ca_old"])
            post = {"top5": 99., "grouped": {"total": 82.5, "old": 75., "new": 90.}, "task_accuracies": [75., 90.]}
            pre = {"grouped": {"total": 78., "old": 70., "new": 86.}, "task_accuracies": [70., 86.]}
            diagnostic = {"pre_ca": pre, "post_ca": post, "ca_applied": True}
            second = recorder.add(1, config, post, diagnostic, {"adapter": 3., "ca": 1.}, 5.)
            summary = recorder.summary(2, 10.)
            self.assertEqual(summary["average_incremental_accuracy"], 81.25)
            self.assertEqual(summary["final_top1"], 82.5)
            self.assertEqual(summary["final_top5"], 99.)
            self.assertEqual(summary["average_forgetting"], 5.)
            self.assertEqual(summary["training_seconds"], 6.)
            self.assertEqual(summary["total_task_seconds"], 8.)
            self.assertEqual(second["pre_ca_forgetting"], 10.)
            with (Path(directory) / "tasks.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            records = [json.loads(line) for line in (Path(directory) / "tasks.jsonl").read_text().splitlines()]
            self.assertEqual(float(rows[-1]["running_aia"]), summary["average_incremental_accuracy"])
            self.assertEqual(records[-1]["post_ca_task_accuracies"], [75., 90.])
            self.assertNotIn("Loss", task_line(second))
            with self.assertRaises(ValueError):
                recorder.summary(3, 10.)

    def test_runner_produces_only_final_checkpoint_and_short_task_logs(self):
        manager = make_manager()
        rng = np.random.default_rng(49)
        manager._train_data = rng.integers(0, 256, (120, 2, 2, 3), dtype=np.uint8)
        manager._train_targets = np.repeat(np.arange(15), 8)
        manager._test_data = rng.integers(0, 256, (60, 2, 2, 3), dtype=np.uint8)
        manager._test_targets = np.repeat(np.arange(15), 4)
        manager._class_order = list(range(15))
        manager._increments = [5, 5, 5]
        template = make_learner(val_ratio=0.).args
        template.update(device=[0], model_name="adapter", shuffle=True, init_cls=5, increment=5,
                        data_root="/uploaded", pretrained_path="/uploaded/weights.pth")
        configs = build_configs(template)
        expected = manager._class_order
        summaries = {}
        def factory(name, args):
            learner = make_learner(quantum=args["use_quantum_kernel_base"] or args["use_quantum_kernel_inc"], val_ratio=0.)
            learner.args = args
            learner.use_quantum_kernel_base = args["use_quantum_kernel_base"]
            learner.use_quantum_kernel_inc = args["use_quantum_kernel_inc"]
            learner.topk = 5
            return learner
        with tempfile.TemporaryDirectory() as directory, \
             patch("data.data_manager.DataManager", return_value=manager), \
             patch("utils.model_factory.get_model", side_effect=factory), \
             patch("trainer._set_device", side_effect=lambda args: args.update(device=[torch.device("cpu")])), \
             patch("models.RSIAT_adapter.AutoencoderSigmoid", small_autoencoder):
            for config in configs[:3]:
                destination = Path(directory) / config["prefix"]
                output = io.StringIO()
                with redirect_stdout(output):
                    summary = runner.execute_run(config, destination, expected, [5, 5, 5])
                summaries[config["prefix"]] = summary
                self.assertEqual(len(output.getvalue().splitlines()), 4)
                self.assertIn("FINAL Average Accuracy=", output.getvalue())
                self.assertIn("Final Top-5=", output.getvalue())
                self.assertNotIn("Loss", output.getvalue())
                self.assertNotIn("kernel=", output.getvalue())
                self.assertEqual([p.name for p in destination.glob("*.pkl")], ["final_task.pkl"])
                with (destination / "tasks.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 3)
                self.assertEqual(summary["status"], "complete")
                historical = {p.name: p.read_bytes() for p in destination.iterdir() if p.is_file()}
                with patch("models.RSIAT_adapter.Learner.incremental_train", side_effect=AssertionError("training forbidden")), \
                     redirect_stdout(io.StringIO()):
                    recovered = runner.evaluate_final(destination)
                self.assertEqual(recovered["final_top5"], summary["final_top5"])
                self.assertEqual(recovered["final_top1"], summary["final_top1"])
                for name, content in historical.items():
                    self.assertEqual((destination / name).read_bytes(), content)
                with self.assertRaises(FileExistsError):
                    runner.execute_run(config, destination, expected, [5, 5, 5])
            self.assertEqual(summaries["A1"]["base_state_sha256"], summaries["A2"]["base_state_sha256"])

    def test_failed_run_records_failure_without_final_metrics(self):
        config = build_configs(self.source())[0]
        config.update(data_root="/uploaded", pretrained_path="/uploaded/weights.pth")
        with tempfile.TemporaryDirectory() as root, patch("data.data_manager.DataManager", side_effect=ValueError("bad data")), \
             patch("trainer._set_device", side_effect=lambda args: args.update(device=[torch.device("cpu")])):
            destination = Path(root) / "failed"
            with self.assertRaisesRegex(ValueError, "bad data"):
                runner.execute_run(config, destination, [], [])
            self.assertTrue((destination / "FAILED.json").is_file())
            self.assertFalse((destination / "run_summary.json").exists())
            self.assertFalse(list(destination.glob("*.pkl")))

    def test_trainer_final_only_policy_preserves_existing_intermediate_file(self):
        import trainer
        manager = make_manager()
        evaluations = iter([
            {"top1": 80., "top5": 100., "grouped": {"total": 80.}, "task_accuracies": [80.]},
            {"top1": 75., "top5": 100., "grouped": {"total": 75.}, "task_accuracies": [70., 80.]},
            {"top1": 70., "top5": 100., "grouped": {"total": 70.}, "task_accuracies": [60., 70., 80.]},
        ])
        saved = []
        def save(path):
            saved.append(path)
            Path(path).write_bytes(b"final test checkpoint")
        model = SimpleNamespace(_network=nn.Linear(1, 1), incremental_train=lambda data: None,
                                eval_task=lambda: next(evaluations), after_task=lambda: None, save_checkpoint=save)
        with tempfile.TemporaryDirectory() as root:
            config = build_configs(self.source())[0]
            config.update(seed=1993, output_root=root, dataset="synthetic", init_cls=2, increment=2,
                          data_root="/uploaded", pretrained_path="/uploaded/weights.pth")
            checkpoint_dir = Path(root) / "ckpt/A0/synthetic/2_2/seed_1993"
            checkpoint_dir.mkdir(parents=True)
            historical = checkpoint_dir / "task_1.pkl"
            historical.write_bytes(b"historical fixture")
            with patch("trainer.DataManager", return_value=manager) as constructor, \
                 patch("trainer.model_factory.get_model", return_value=model), \
                 patch("trainer._set_device", side_effect=lambda args: args.update(device=[torch.device("cpu")])), \
                 patch("trainer.logging.basicConfig"), patch("trainer.logging.FileHandler"), redirect_stdout(io.StringIO()):
                trainer._train(config)
            self.assertEqual(len(saved), 1)
            self.assertEqual(Path(saved[0]).name, "task_2.pkl")
            self.assertEqual(historical.read_bytes(), b"historical fixture")
            self.assertFalse((checkpoint_dir / "task_0.pkl").exists())
            self.assertEqual(constructor.call_args.kwargs, {"data_root": "/uploaded", "offline": True})

    def test_collection_is_explicitly_partial_and_excludes_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            study = Path(root) / "study"
            (study / "configs").mkdir(parents=True)
            run = study / "runs/A0_1993"
            run.mkdir(parents=True)
            write_json(study / "study_manifest.json", {"runs": [{"run_id": "A0_1993"}, {"run_id": "A1_1993"}]})
            for path in (study / "preflight.json", study / "source_config.json", run / "run_summary.json"):
                write_json(path, {})
            (run / "final_task.pkl").write_bytes(b"not for review")
            output = Path(root) / "results.zip"
            with redirect_stdout(io.StringIO()):
                collect(study, output)
            with zipfile.ZipFile(output) as archive:
                self.assertFalse(any(name.endswith(".pkl") for name in archive.namelist()))
                manifest = json.loads(archive.read("collection_manifest.json"))
                self.assertEqual(manifest["status"], "partial")
                self.assertEqual(manifest["not_started"], ["A1_1993"])
            with self.assertRaises(FileExistsError):
                collect(study, output)

    def test_notebook_cells_compile_and_source_bundle_excludes_old_artifacts(self):
        notebook = runner.REPO_ROOT / "notebooks/QKSR_Failure_Localization_Kaggle_Offline.ipynb"
        content = json.loads(notebook.read_text(encoding="utf-8"))
        for cell in content["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "offline_notebook", "exec")
                self.assertEqual(cell["outputs"], [])
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = package(Path(directory) / "source.zip")
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertIn("qg_rsa/scripts/run_failure_localization.py", names)
                self.assertFalse(any("/logs/" in name or "/ckpt/" in name or "__pycache__" in name for name in names))
                manifest = json.loads(archive.read("qg_rsa/BUILD_MANIFEST.json"))
                self.assertTrue(all("\\" not in name for name in manifest["source_sha256"]))
                for relative, expected in manifest["source_sha256"].items():
                    import hashlib
                    self.assertEqual(hashlib.sha256(archive.read("qg_rsa/" + relative)).hexdigest(), expected)

    def test_notebook_bootstrap_accepts_zip_and_kaggle_unpacked_source(self):
        notebook = json.loads((runner.REPO_ROOT / "notebooks/QKSR_Failure_Localization_Kaggle_Offline.ipynb").read_text(encoding="utf-8"))
        bootstrap = "".join(next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), patch.object(sys, "path", sys.path.copy()):
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            output = package(inputs / "source.zip")
            unpacked = root / "unpacked_inputs"
            with zipfile.ZipFile(output) as archive:
                archive.extractall(unpacked)
            for index, input_root in enumerate((inputs, unpacked)):
                code = bootstrap.replace('Path("/kaggle/input")', "Path(" + repr(str(input_root)) + ")")
                code = code.replace('Path("/kaggle/working")', "Path(" + repr(str(root / ("working" + str(index)))) + ")")
                namespace = {}
                exec(compile(code, "offline_bootstrap_fixture", "exec"), namespace)
                self.assertTrue((namespace["REPO_ROOT"] / "scripts/run_failure_localization.py").is_file())


if __name__ == "__main__":
    unittest.main()
