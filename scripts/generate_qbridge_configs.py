"""Generate locked R0-R10 overlays, optionally merge exact base-run configuration."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.qbridge import config_from_args, write_json
from utils.qbridge_experiment import merge_dict


def generate(output_dir, source=None, base_path="<REQUIRED_EXISTING_BASE_CHECKPOINT>",
             artifact_root="<REQUIRED_ARTIFACT_DIRECTORY>", seeds=(1993, 2015, 2026)):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/QBRIDGE_R0-R10_ablation_manifest_v1.1.json").read_text(encoding="utf-8"))
    full = json.loads((root / "docs/QBRIDGE_R6_full_override_v1.1.json").read_text(encoding="utf-8"))
    # v1.0-style randomization is split into the v1.1 R9 and R10 named selectors.
    overlays = {}
    for variant, entry in manifest["variants"].items():
        cfg = merge_dict(manifest["common_incremental_overrides"], entry["overrides"])
        if variant == "R6":
            cfg = merge_dict(cfg, full)
        cfg["qbridge"]["variant"] = variant
        cfg["qbridge"]["base_checkpoint_path"] = base_path
        cfg["qbridge_experiment"] = True
        cfg["q_inc_train_mode"] = "frozen"
        cfg["prefix"] = "qbridge_" + variant
        cfg["seed"] = list(seeds)
        cfg["save_checkpoints"] = True
        cfg["resume"] = False
        cfg["compact_diagonal_checkpoint"] = False
        if variant == "R1":
            cfg["qbridge"]["risk_mode"] = "uniform"
        if variant == "R6":
            # CLI runner resolves this seed along with per-seed artifact paths.
            cfg["qbridge"]["calibration"]["split_seed"] = "{seed}"
            cfg["qbridge"]["calibrator_artifact_path"] = artifact_root + "/{dataset}/{seed}/calibrator.pt"
        if variant in ("R7", "R8"):
            cfg["qbridge"]["selector_artifact_path"] = artifact_root + "/{dataset}/{seed}/" + ("rbf.pt" if variant == "R7" else "mlp.pt")
        cfg = merge_dict(source, cfg) if source is not None else cfg
        config_from_args(cfg)
        write_json(Path(output_dir) / (variant + ".json"), cfg)
        overlays[variant] = entry["description"]
    write_json(Path(output_dir) / "manifest.json", {
        "schema": "qbridge-v1.1-generated", "variants": overlays, "seeds": list(seeds),
        "status": "requires compatible actual base/artifacts; no benchmark executed",
        "full_raw": "R5", "full_calibrated": "R6", "base_path_template": base_path,
        "classical_training": {"R7": "scripts/qbridge_prepare.py rbf: frozen shared projection; base-only median gamma",
                               "R8": "scripts/qbridge_prepare.py mlp: fixed base features; steering-only SGD; report budget"}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Exact saved base-run JSON; no dataset defaults invented")
    parser.add_argument("--output-dir", default="exps/qbridge/overlays")
    parser.add_argument("--base", default="<REQUIRED_EXISTING_BASE_CHECKPOINT>")
    parser.add_argument("--artifact-root", default="<REQUIRED_ARTIFACT_DIRECTORY>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1993, 2015, 2026])
    cli = parser.parse_args()
    source = json.loads(Path(cli.source).read_text(encoding="utf-8")) if cli.source else None
    generate(cli.output_dir, source, cli.base, cli.artifact_root, cli.seeds)


if __name__ == "__main__":
    main()
