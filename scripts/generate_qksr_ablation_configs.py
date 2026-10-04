"""Generate the minimum QKSR v3.1 ablation configs from one tuned base JSON."""

import argparse
import copy
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.qksr_profiles import CURRENT_MARGIN_OPTIONS, RETENTION_OPTIONS, RETENTION_SHARED_OPTIONS


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
    # Change one factor at a time along F1/F3/F4/F5. F6/F7 isolate
    # the inherited SSCA integration issue from the quantum metric.
    "F1_inc_current": {"q_inc_pair": "current"},
    "F2_inc_margin": {"inc_loss_mode": "margin"},
    "F3_current_margin": {"q_inc_pair": "current", "inc_loss_mode": "margin"},
    "F4_current_margin_weak": {
        "q_inc_pair": "current", "inc_loss_mode": "margin", "q_inc_weight": 0.25,
    },
    "F5_current_margin_warmup": {
        "q_inc_pair": "current", "inc_loss_mode": "margin",
        "q_inc_weight": 0.25, "q_inc_warmup_epochs": 3,
    },
    "F6_rsiat_paired_ssca": {
        "use_quantum_kernel_base": False, "use_quantum_kernel_inc": False,
        "ssca_feature_mode": "paired_eval",
    },
    "F7_qksr_paired_ssca": {"ssca_feature_mode": "paired_eval"},
    "F8_rbf_current_margin_warmup": {
        "q_kernel_type": "rbf_proj", "q_inc_pair": "current", "inc_loss_mode": "margin",
        "q_inc_weight": 0.25, "q_inc_warmup_epochs": 3,
    },
}

# Controlled chain: each adjacent config changes ONE retention mechanism.
_retention_step = {**CURRENT_MARGIN_OPTIONS, "ssca_feature_mode": "paired_eval", "record_stage_metrics": True}
VARIANTS["G0_paired_current_margin"] = dict(_retention_step)
for _name, _change in (
    ("G1_signed_projector", {"ae_type": "signed_residual"}),
    ("G2_reset_projector", {"ae_reset_each_task": True}),
    ("G3_detached_prototypes", {"q_detach_prototypes": True}),
    ("G4_relation_retention", {"relation_distill_weight": 1.0, "relation_temperature": 0.2}),
    ("G5_mean_cov_transport", {"statistics_transport": "guarded_ridge"}),
    ("G6_cov_shrinkage", {"stats_cov_shrinkage": 0.05}),
):
    _retention_step.update(_change)
    VARIANTS[_name] = dict(_retention_step)
VARIANTS["G7_retention_rbf"] = {**RETENTION_OPTIONS, "q_kernel_type": "rbf_proj"}
VARIANTS["G8_rsiat_shared_retention"] = {
    **RETENTION_SHARED_OPTIONS, "use_quantum_kernel_base": False, "use_quantum_kernel_inc": False,
}


def generate(source: Path, output_dir: Path) -> None:
    source_config = json.loads(source.read_text(encoding="utf-8"))
    # Explicit legacy reference, including when source is a RSIAT-only config.
    base = {
        "use_quantum_kernel_base": True, "use_quantum_kernel_inc": True,
        "q_kernel_type": "pqk", "q_kernel_order": 1, "q_order2_weight": 1.0,
        "q_num_qubits": 8, "q_num_layers": 2, "q_reupload": False,
        "q_dtype": "float32", "q_gamma_mode": "bounded_learned",
        "q_calib_samples": 512, "q_init_seed": 1234, "q_metric_lr_mult": 0.1,
        "q_inc_train_mode": "frozen", "q_inc_pair": "old_proj", "inc_loss_mode": "mean",
        "q_inc_weight": 1.0, "q_inc_warmup_epochs": 0,
        "ssca_feature_mode": "legacy", "rs_margin_q": 0.5,
        "rs_margin_q_mode": "fixed", "rs_margin_quantile": 0.9, "rs_margin_inc": 0.3,
        **source_config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in VARIANTS.items():
        config = copy.deepcopy(base)
        config.update(overrides)
        # New exploratory runs retain all generated checkpoints; do not delete
        # prior task artifacts while diagnosing forgetting/stage effects.
        if name.startswith("G"):
            config["keep_last_checkpoint"] = False
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
