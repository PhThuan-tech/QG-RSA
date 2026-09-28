# RSIAT-BiAlign controlled experiment

## Scientific status

- **Research question:** Does replacing RSIAT's one-way incremental alignment with a BiCyc-inspired bidirectional alignment reduce accumulated forgetting over CIFAR100 B0I10?
- **[HYPOTHESIS]:** If bidirectional alignment reduces accumulated drift, gains should be most visible in later tasks and old-class retention.
- **[IMPLEMENTATION ADAPTATION]:** This is not a full BiCyc integration and is not claimed as novel. The only intended method switch is `L_align -> L_fwd + L_back`.
- **[PAPER CLAIM]:** The experiment specification reports RSIAT ViT-B/16-IN21K references `A_bar=95.15%` and `A_B=92.20%`; these are context, not results from this branch.

## Pre-edit RSIAT trace and P_t audit

Path: `main.py -> trainer.RSIAT_train/_train -> models.RSIAT_adapter.Learner.incremental_train -> _train -> _init_train -> _compute_rt_loss -> _inc_loss_components`.

1. `P_t` is `utils.toolkit.AutoencoderSigmoid`, held as `Learner.old_ae`.
2. Mapping: `P_t(z) = z + decoder(encoder(z))` with `768 -> 64 -> ae_code_dims -> 64 -> 768`.
3. In the locked config `ae_residual_mode="sigmoid"`, the last decoder activation is `Sigmoid`; its residual is constrained to `[0,1]`, so it is one-sided rather than a natural inverse family.
4. PyTorch default linear initialization is used. `P_t` is first created when `_cur_task == 1`.
5. `P_t` is not recreated for later tasks; it is warm-started across incremental transitions.
6. Its SGD group uses `ae_init_lr=0.0005611742230603744` and `ae_weight_decay=0.0029339295871570244`.
7. `old_ae_state_dict` is saved at task boundaries and strictly restored by the learner checkpoint path. Optimizer/scheduler state is recreated per task, as in the existing repository.

The frozen teacher is made by `Learner.after_task`: `self._network.copy().freeze()`. `same_input_feature_pair` evaluates current and old extractors on the exact same augmented input, with the old extractor under `torch.no_grad()`.

Official downstream flow is preserved for `official` and `bialign`: the two official extraction passes, official SSCA displacement, class-mean/covariance lifecycle, and `BaseLearner._stage2_compact_classifier` CA path.

## Paper/mechanism to code map

| Concept | Module/class | File | Function / variables |
|---|---|---|---|
| Current feature `z_new=f_t(x)` | `Learner` / `SimpleVitNet` | `models/RSIAT_adapter.py` | `_compute_rt_loss`; `features` |
| Frozen previous feature `z_old=f_{t-1}(x)` | frozen `old_network_module_ptr` | `models/RSIAT_adapter.py`, `utils/bicyc_transport.py` | `same_input_feature_pair`; `features_old` |
| RSIAT forward projector `P_t` | `AutoencoderSigmoid`, `old_ae` | `utils/toolkit.py`, `models/RSIAT_adapter.py` | `old_ae(features_old)` |
| Reverse projector `D_t` | `SignedResidualProjector`, `reverse_projector` | `utils/bialign.py`, `models/RSIAT_adapter.py` | `D_t(z)=z+up(GELU(down(z)))` |
| `L_fwd` | `bialign_loss_terms` | `utils/bialign.py` | `MSE(P_t(z_old.detach()), z_new.detach())` |
| `L_back` | `bialign_loss_terms` | `utils/bialign.py` | `MSE(D_t(z_new), z_old.detach())` |
| `L_bialign` | `bialign_loss_terms` | `utils/bialign.py` | `loss_fwd + loss_back` (no `/2`, no `lambda_bi`) |
| unchanged `L_orth` | `Learner` | `models/RSIAT_adapter.py` | `_bialign_loss_components`; same P-mapped prototypes, normalization, similarity mean |
| total incremental loss | `Learner` | `models/RSIAT_adapter.py` | `L_cos + beta*L_bialign + gamma*L_orth` |
| official SSCA | `Learner` | `models/RSIAT_adapter.py` | `displacement(..., sigma=4.0)` in official-statistics branch |
| official CA | `BaseLearner` | `models/base.py` | `_stage2_compact_classifier`; no BiCyc statistics transport |

## D_t architecture and lifecycle

`D_t` is a lightweight signed residual bottleneck:

```text
z -> Linear(768,64) -> GELU -> Linear(64,768) -> residual + z
```

- Parameters: 99,136; there is no 768x768 projection.
- The final linear weight and bias are zero initialized, making `D_t(z)==z` exactly at transition start.
- Final residual output is signed/unconstrained (no sigmoid).
- Initialization uses a deterministic task seed while preserving global RNG state, so introducing D_t does not perturb matched data-order RNG.
- **[IMPLEMENTATION ADAPTATION]:** D_t is reset to identity for every new transition. This follows its transition-specific inverse role and the pre-registered identity preference. It is not silently warm-started. The completed-transition D_t is still checkpointed and strictly restorable for provenance.

## Gradient-routing contract

| Isolated term | Current adapter | P_t | D_t | frozen old model |
|---|---:|---:|---:|---:|
| `L_fwd` | no | yes | no | no |
| `L_back` | yes | no | yes | no |
| `L_orth` | no | yes | no | no |
| `L_cos` | yes | no | no | no |

The smoke configuration records finite-aware L2 gradient norms after the first backward pass for the current adapter, P_t, D_t, and old model. It fails closed if an expected BiAlign group has no gradient, any gradient is nonfinite, or the frozen old model has a gradient.

## Optimizer groups

For incremental tasks:

| Group | LR | weight decay |
|---|---:|---:|
| current ViT adapter / convnet | `0.018461273665041644` | `0.0001714081878951529` |
| cosine classifier | `0.018461273665041644` | `0.0001714081878951529` |
| P_t | `0.0005611742230603744` | `0.0029339295871570244` |
| D_t | `0.0005611742230603744` | `0.0029339295871570244` |

**[IMPLEMENTATION ADAPTATION]:** D_t reuses P_t's optimizer scale, avoiding a new tuned hyperparameter. All groups use the existing SGD/momentum and cosine scheduler. `beta=1.0` applies directly to `L_fwd+L_back`; BiCyc `lambda_bi=5` is not applied and the sum is not divided by two.

The inherited Track-B record schema still carries legacy transport-default metadata for backward-compatible output parsing. In `bialign`, those values are inactive: configs contain no `lambda_bi`/`lambda_cycle`, `transport==0`, cycle terms are zero, `stage2_epochs` is empty, and `statistics_transport` is null.

## Matched full-run protocol

Both configs use the same immutable task0 checkpoint and compatibility report:

- checkpoint: `/mnt/d/RSIAT_overnight/BiCyc_style/common_task0/task_0.pkl`
- SHA-256: `d13bd496019dc5739eefd709c6bf60e28506577dbbb41b50ee81a5e109e55ea0`
- task0 Top1 (`A_0`): `99.1`
- seed/class order: `1993`, identical checkpoint class order
- training batch size: `32`
- class-statistics extraction batch size: `32` in both arms
- incremental epochs: `30`; CA epochs: `10`
- official SSCA and official CA; covariance fallback disabled in both arms

Prepared commands (do not launch until all pre-run gates pass):

```bash
cd /home/khanh/research/rsiat-bialign
PYTHON_BIN=/home/khanh/research/khanh-research/RQ7_transition_7_to_8_local/.venv/bin/python \
  ./scripts/run_rsiat_bialign_full.sh official
PYTHON_BIN=/home/khanh/research/khanh-research/RQ7_transition_7_to_8_local/.venv/bin/python \
  ./scripts/run_rsiat_bialign_full.sh bialign
```

`tools/verify_bialign_configs.py` fails closed unless the controlled fields and task0 hashes match. The only method switch is `bicyc_mode: official -> bialign`; arm labels and output paths necessarily differ.

**[IMPLEMENTATION DEVIATION — runtime only]:** The repository's official class-statistics loader hard-codes batch 64. A completed one-epoch Stage-I smoke then OOMed in that official extraction path on the 8 GB GPU, before pre-CA/CA. `stats_batch_size=32` is therefore set identically in both full arms and the smoke. The code default remains 64 when this key is absent; sample order, transforms, statistic formulas, SSCA, CA, and lifecycle are unchanged. The launch script also enables PyTorch's expandable CUDA allocator for both arms. This correction was selected from the traceback/memory limit, not task accuracy.

## Metrics and decision rule

For post-CA all-seen-class accuracies `A_0...A_9`:

- `A_bar = mean(A_0...A_9)`
- `A_B = A_9`

Also record per-task pre/post-CA Top1, old/new Top1, CA gain, finite status, runtime, and per-class Top1. Do not interpret a smoke result as method evidence. Full-curve outcomes should be labeled as evidence from one seed, not as a general conclusion.

## Pre-registered interpretations

1. Better `A_bar`/`A_B`, especially late-task old accuracy: supports the hypothesis for this seed.
2. Better old but worse new accuracy: evidence of a stability/plasticity tradeoff and possible over-regularization.
3. Better pre-CA but no post-CA gain: official SSCA/CA may mask representation benefit.
4. Matching curves: little additional signal in this pretrained low-drift setting.
5. Immediate collapse: diagnose implementation/optimization failure before treating it as evidence against the hypothesis.

## One-epoch smoke result

**[EVIDENCE — implementation sanity only]:** The corrected task0→task1 smoke completed with return code 0, exactly one Stage-I epoch, official SSCA, one official CA epoch, and strict checkpoint save/resume.

- Stage-I: `L_cos=1.22950975`, `L_fwd=0.38670576`, `L_back=0.13714505`, `L_bialign=0.52385081`, `L_orth=0.26496074`, total `1.85934485`.
- First-backward gradient L2: current adapter `2.52255393`, P_t `0.01332358`, D_t `0.00602429`; frozen old model had no gradient. All were finite.
- All Track-B-only terms were zero; no cycle, Stage-II, pair-preserve, or statistics transport ran.
- Pre-CA Top1 `97.25`; post-CA `A_1=97.35`; old/new Top1 `97.9/96.8`; CA gain `+0.10 pp`.
- Saved task1 checkpoint SHA-256: `8ae4eccc8f72833611e522a9c6e688e4bad95a2ebd120c34912245af0e90f1e5`.
- A fresh learner strictly restored network, P_t, D_t, class statistics, and task state with no missing/unexpected trainable state.

The first attempt completed the same Stage-I epoch but OOMed afterward in the official hard-coded batch-64 statistics loader. The corrected retry's full Stage-I record was exactly equal to the failed attempt's record, supporting that `stats_batch_size=32` changed only the downstream runtime envelope.

**[INFERENCE]:** The implementation and required gradient routes are healthy enough to prepare the matched full-run commands. This smoke provides no evidence that BiAlign improves continual-learning quality and was not used to tune beta or D_t.
