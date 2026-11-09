"""Run-level statistics on synthetic logs; not benchmark results."""
import math
from datetime import datetime, timedelta
import tempfile
import unittest
from pathlib import Path

from scripts import run_level_stats as rls


def write_log(root, dataset, prefix, seed, curve, forgetting=5.0, minutes=30):
    folder = Path(root) / "adapter" / dataset / "0" / "20"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "{}_{}_vit.log".format(prefix, seed)
    end = (datetime(2026, 10, 20, 10) + timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(
        "2026-10-20 10:00:00,000 [trainer.py] => prefix: {}\n".format(prefix)
        + "2026-10-20 10:05:00,000 [trainer.py] => Continual metrics after task 1: BWT=-1.0, forgetting={}\n".format(forgetting)
        + "{},000 [trainer.py] => CNN top1 curve: {}\n".format(end, list(curve)), encoding="utf-8")
    return path


class FamilyAndParsingTests(unittest.TestCase):
    def test_split_family(self):
        self.assertEqual(rls.split_family("noise"), ("noise", 0))
        self.assertEqual(rls.split_family("noise_rep3"), ("noise", 3))
        self.assertEqual(rls.split_family("diag_legacy_rep1"), ("diag_legacy", 1))
        self.assertEqual(rls.split_family("r3_candidate"), ("r3_candidate", 0))

    def test_parse_run_reads_metrics_dataset_and_duration(self):
        with tempfile.TemporaryDirectory() as root:
            run = rls.parse_run(write_log(root, "imagenetr", "noise_rep2", 1993, [90, 80], forgetting=7.5, minutes=42))
            self.assertEqual((run["dataset"], run["family"], run["replicate"], run["seed"]), ("imagenetr", "noise", 2, 1993))
            self.assertEqual((run["aia"], run["final"], run["forgetting"]), (85.0, 80, 7.5))
            self.assertAlmostEqual(run["minutes"], 42.0)

    def test_incomplete_runs_are_excluded_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            write_log(root, "imagenetr", "noise", 1, [90, 80, 70])
            write_log(root, "imagenetr", "noise_rep1", 1, [90, 80])   # stopped early
            self.assertEqual([r["family"] + str(r["replicate"]) for r in rls.collect([root])], ["noise0"])
            with tempfile.TemporaryDirectory() as other:
                write_log(other, "imagenetr", "noise", 1, [90, 80, 70])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    rls.collect([root, other])


class NoiseTests(unittest.TestCase):
    def test_noise_table_sd_pairs_and_selection(self):
        with tempfile.TemporaryDirectory() as root:
            for k, final in enumerate((70.0, 71.0, 72.0)):
                write_log(root, "imageneta", "noise" if k == 0 else "noise_rep%d" % k, 1, [80.0, final], minutes=15)
                write_log(root, "imagenetr", "noise" if k == 0 else "noise_rep%d" % k, 1, [90.0, 85.0 + 0.1 * k],
                          minutes=60)
            rows = rls.noise_table(rls.collect([root]))
            a = next(r for r in rows if r["dataset"] == "imageneta")
            self.assertAlmostEqual(a["aia_sd"], 0.5)          # AIA = 75, 75.5, 76
            sd_diff = math.sqrt(2) * 0.5
            self.assertEqual(a["pairs_needed"]["0.5"], math.ceil(((1.959964 + 0.841621) * sd_diff / 0.5) ** 2))
            self.assertEqual(a["cost_minutes"], a["pairs_needed"]["0.5"] * 2 * 15)
            text = rls.render_noise(rows)
            # ImageNet-R is slower but far less noisy, so it is cheaper to screen on.
            self.assertIn("**Benchmark sàng lọc theo quy tắc đặt trước:** `imagenetr`", text)

    def test_pairs_needed_has_floor(self):
        self.assertEqual(rls.pairs_needed(0.0, 0.5), 0)
        with tempfile.TemporaryDirectory() as root:
            for k in range(2):
                write_log(root, "imageneta", "noise" if k == 0 else "noise_rep1", 1, [80.0, 70.0])
            row = rls.noise_table(rls.collect([root]))[0]
            self.assertEqual(row["pairs_needed"]["1.0"], rls.MIN_PAIRS)


class CompareTests(unittest.TestCase):
    def build(self, root, deltas):
        for k, delta in enumerate(deltas):
            suffix = "" if k == 0 else "_rep%d" % k
            base = 70.0 + k
            write_log(root, "imagenetr", "noise" + suffix, 1, [80.0, base])
            write_log(root, "imagenetr", "cand" + suffix, 1, [80.0 + delta, base + delta])

    def test_paired_ci_and_verdicts(self):
        with tempfile.TemporaryDirectory() as root:
            self.build(root, [1.0, 1.2, 0.8])
            result = rls.compare(rls.collect([root]), "noise", "cand")
            self.assertEqual(result["n"], 3)
            self.assertAlmostEqual(result["aia"]["mean"], 1.0)
            half = 4.303 * 0.2 / math.sqrt(3)
            self.assertAlmostEqual(result["aia"]["ci95"][0], 1.0 - half)
            self.assertTrue(result["verdict"].startswith("ĐẠT"))
        with tempfile.TemporaryDirectory() as root:
            self.build(root, [0.5, -0.4, 0.1])
            self.assertTrue(rls.compare(rls.collect([root]), "noise", "cand")["verdict"].startswith("không kết luận"))
        with tempfile.TemporaryDirectory() as root:
            self.build(root, [1.0, 1.0])
            self.assertTrue(rls.compare(rls.collect([root]), "noise", "cand")["verdict"].startswith("chưa đủ"))

    def test_t_quantiles(self):
        self.assertAlmostEqual(rls.t975(2), 4.303)
        self.assertAlmostEqual(rls.t975(500), rls.Z_ALPHA)


if __name__ == "__main__":
    unittest.main()
