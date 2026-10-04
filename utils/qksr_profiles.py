"""Explicit experimental profiles; none are claimed to improve ImageNet yet."""

CURRENT_MARGIN_OPTIONS = {
    "q_inc_pair": "current", "inc_loss_mode": "margin",
    "q_inc_weight": 0.25, "q_inc_warmup_epochs": 3,
}

RETENTION_SHARED_OPTIONS = {
    "ae_type": "signed_residual", "ae_reset_each_task": True,
    "ae_init_seed": 1234, "ssca_feature_mode": "paired_eval",
    "relation_distill_weight": 1.0, "relation_temperature": 0.2,
    "statistics_transport": "guarded_ridge", "transport_rank": 32,
    "transport_ridge": 0.01, "transport_max_change": 0.25,
    "transport_support_scale": 1.0, "transport_support_floor": 0.05,
    "stats_cov_shrinkage": 0.05, "record_stage_metrics": True,
    "keep_last_checkpoint": False,
}

RETENTION_OPTIONS = {
    **CURRENT_MARGIN_OPTIONS, **RETENTION_SHARED_OPTIONS,
    "q_detach_prototypes": True,
}
