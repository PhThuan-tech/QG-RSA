"""Generate the minimum QKSR v3.1 ablation configs from one tuned base JSON."""

import argparse
import copy
import json
from pathlib import Path


VARIANTS = {
    "A_rsiat": {
        "use_quantum_kernel_base": False,
        "use_quantum_kernel_inc": False,
    },
    "B_qksr": {},
    "C_base_only": {"use_quantum_kernel_inc": False},
    "D1_inc_random_frozen": {
        "use_quantum_kernel_base": False,
        "use_quantum_kernel_inc": True,
        "q_inc_train_mode": "frozen",
    },
    "D2_inc_trainable": {
        "use_quantum_kernel_base": False,
        "use_quantum_kernel_inc": True,
        "q_inc_train_mode": "trainable",
        "q_gamma_mode": "bounded_learned",
        "inc_loss_mode": "margin",
    },
    "E1_rbf_proj": {"q_kernel_type": "rbf_proj"},
    "E2a_mlp_small": {"q_kernel_type": "mlp_small"},
    "E2b_mlp_cap": {"q_kernel_type": "mlp_cap"},
    "E3_no_cnot": {"q_kernel_type": "pqk_no_cnot"},
    "E4_random_frozen": {"q_kernel_type": "pqk_random_frozen"},
    "E5_order2": {"q_kernel_order": 2},
    "E6_reupload": {"q_reupload": True},
}


def generate(source: Path, output_dir: Path) -> None:
    base = json.loads(source.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = copy.deepcopy(base)
        config.update(overrides)
        config["prefix"] = "qksr_{}".format(name)
        destination = output_dir / "{}.json".format(name)
        destination.write_text(
            json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="exps/adapter_cub.json")
    parser.add_argument("--output-dir", default="exps/qksr_ablation")
    args = parser.parse_args()
    generate(Path(args.source), Path(args.output_dir))


if __name__ == "__main__":
    main()

