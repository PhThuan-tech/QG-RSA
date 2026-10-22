import json
import tempfile
import unittest
from pathlib import Path

from data.data import CUB, prepare_cub_split
from utils.qsd import qsd_settings

ROOT = Path(__file__).resolve().parents[1]


def fake_cub(root, classes=3, per_class=4):
    """Raw CUB_200_2011 layout: images/<class>/<file>, images.txt, train_test_split.txt."""
    lines, split, image_id = [], [], 0
    for c in range(classes):
        folder = root / "images" / "{:03d}.Bird_{}".format(c + 1, c)
        folder.mkdir(parents=True)
        for k in range(per_class):
            image_id += 1
            name = "{}/img_{}.jpg".format(folder.name, k)
            (root / "images" / name).write_bytes(b"jpg")
            lines.append("{} {}".format(image_id, name))
            split.append("{} {}".format(image_id, 1 if k % 2 == 0 else 0))
    (root / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "train_test_split.txt").write_text("\n".join(split) + "\n", encoding="utf-8")


class CubTest(unittest.TestCase):
    def test_official_split_and_loader_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, dest = Path(tmp) / "CUB_200_2011", Path(tmp) / "split"
            fake_cub(raw)
            self.assertEqual(prepare_cub_split(str(raw), str(dest)), {"train": 6, "test": 6})
            self.assertEqual(sorted(p.name for p in (dest / "train").iterdir()),
                             sorted(p.name for p in (dest / "test").iterdir()))
            self.assertEqual(len(list((dest / "train" / "001.Bird_0").iterdir())), 2)
            # Idempotent: a second call keeps the same files.
            self.assertEqual(prepare_cub_split(str(raw), str(dest)), {"train": 6, "test": 6})
            loader = CUB()
            loader.data_root = str(dest)
            with self.assertRaises(ValueError):  # only 3 classes: the 200-class protocol is enforced
                loader.download_data()


class OverlayTest(unittest.TestCase):
    def test_scf_overlay_changes_evaluation_only(self):
        scf = json.loads((ROOT / "exps/scf/scf.json").read_text(encoding="utf-8"))
        a2 = json.loads((ROOT / "exps/qsd/qsd2_rep0.json").read_text(encoding="utf-8"))
        changed = {k for k in set(scf) | set(a2) if scf.get(k) != a2.get(k)}
        self.assertEqual({k for k in changed if not k.startswith("qsd")}, {"prefix", "offline"})
        cfg = qsd_settings(scf)
        for head in ("fuse_ridge_stats_b0p25", "fuse_pgm_stats_b0p5", "fuse_pgm_stats_hier_b1"):
            self.assertIn(head, cfg["heads"])  # the frozen A2 weights are evaluated
        self.assertNotIn("pgm_stats_oracle", cfg["heads"])


class SummaryTest(unittest.TestCase):
    def write_run(self, root, dataset, prefix, seed, gain, tasks=3):
        folder = root / "adapter" / dataset / "0" / "10"
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        base_rows = [[90.0 - 5 * (t - j) for j in range(t + 1)] for t in range(tasks)]
        heads = {"linear": 0.0, "fuse_ridge_stats_b0p25": gain, "fuse_pgm_stats_b0p5": gain / 2,
                 "fuse_pgm_stats_hier_b1": gain / 2, "ridge_stats": -1.0, "pgm_stats": -1.0}
        for t in range(tasks):
            for head, g in heads.items():
                row = [a + g for a in base_rows[t]]
                lines.append("x => QSD tasks head={}: {}".format(head, json.dumps(row)))
        for head, g in heads.items():
            curve = [sum(r) / len(r) + g for r in base_rows]
            lines.append("x => QSD curve head={}: {}".format(head, json.dumps(curve)))
        lines.append("x => CNN top1 curve: {}".format(json.dumps([sum(r) / len(r) for r in base_rows])))
        (folder / "{}_{}_pretrained_vit_b16_224_in21k_adapter.log".format(prefix, seed)).write_text("\n".join(lines))

    def test_unified_three_seed_table_and_rules(self):
        from scripts.summarize_scf import collect, continual, render
        self.assertEqual(continual([[90.0], [85.0, 92.0], [80.0, 88.0, 95.0]]), (7.0, -7.0))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, seed in enumerate((1993, 2015, 2026)):
                self.write_run(root, "cifar224", "scf", seed, 0.4 + 0.1 * i)
                self.write_run(root, "cub", "scf", seed, 0.8 + 0.1 * i)
                self.write_run(root, "imagenetr", "qsd2", seed, 0.3 + 0.1 * i)
            self.write_run(root, "imagenetr", "qsd2_rep1", 1993, 9.0)  # replicate 1: not in the 3-seed table
            self.write_run(root, "imagenetr", "qsd2", 2042, 9.0)       # extra seed: not in the table
            runs = collect([tmp])
            self.assertEqual(len(runs), 9)
            text, decision = render(runs)
        self.assertTrue(decision["transfer"]["fuse_ridge_stats_b0p25"]["holds"])
        self.assertEqual(decision["transfer"]["fuse_ridge_stats_b0p25"]["pooled"]["n"], 6)
        self.assertFalse(decision["quantum"]["holds"])  # PGM gain is half the ridge gain here
        self.assertIn("ImageNet-A 1993", text)          # reported as missing
        self.assertNotIn("≠", text)


if __name__ == "__main__":
    unittest.main()
