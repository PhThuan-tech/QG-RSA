# Q-BRIDGE v1.1 — Research & Implementation Contract

**Working title:** Calibrated Quantum Risk-Guided Cross-Task Margin Learning with Old-Anchor Protection  
**Setting:** Exemplar-Free Class-Incremental Learning (EFCIL)  
**Base system:** RSIAT + base-task QKSR, repository `QG-RSA`, branch `Quantum_research`  
**Status:** **Design specification — NOT an experimentally verified contribution**  
**Date:** 14 October 2026  
**Primary recipient:** Codex / research implementation agent  
**Version policy:** v1.1 supersedes `QBRIDGE_Implementation_Spec_v1.md` when instructions conflict.

> **One-line method:** Use the frozen PQK learned at base as an *optional risk-score teacher* for old–new cosine-margin violations during incremental adaptation, with a separate old-anchor guard; do not modify the RSIAT SSCA/CA stages or the inference graph.

## 0. Executive contract & non-negotiable invariants

**R5 = full raw pilot**, **R6 = full calibrated v1**. Both include: new-class classification + RSIAT alignment + risk-weighted cross-task margin + old-anchor guard. R6 differs **only by a monotone scalar mapping from PQK similarity to weight intensity**; R6 cannot alter PQK ranking. The newly proposed objectives **replace** original incremental quantum separation; they do not add to it.

**Scope locks**

1. **Base:** Reuse the *same* existing QKSR base checkpoint across paired incremental runs; do **not** redesign base learning in v1.1.
2. **Incremental:** Only introduce specified losses, selector, and optional calibrator. Do **not** change ViT, adapter architecture, SSCA, covariance buffers, Gaussian CA, task splits, or inference.
3. **Scientific claim:** Study whether PQK adds useful *weighting information beyond uniform and classical selectors*. No pre-verified claim of quantum advantage, robustness, fewer forgetting events, or stable old representations.
4. **Baseline switch:** `qbridge.enabled=false` means the **exact legacy code path/config applies**, including original separation when previously enabled. `enabled=true` means **legacy incremental separation is off irrespective of its old flag**.
5. **Reproducibility:** Use the same base checkpoint and class order for all incremental variants on a given seed. No test-set hyperparameter tuning.
6. **Full configuration:** `R6` must explicitly set `lambda_cross>0`, `lambda_guard>0`, `risk_gain>0`, `risk_mode=calibrated` and reference a fitted calibrator. No silent fallback to raw risk under an `R6` label.
7. **No unsupported assumptions:** All tensor layouts, head APIs, checkpoint contents and optimizer behaviors must be verified in *actual source* before modification. This document defines intended behavior, not an assertion that keys/APIs already exist.

## 1. Why this method / falsifiable hypotheses

**Observed starting point (three paired seeds, per provided experiment summary):** CIFAR100 PQK variants show nearly identical AIA to RSIAT; ImageNet-A PQK base improves initial accuracy but that gain does not persist in AIA/final accuracy. This supports testing the mismatch between base similarity learning and incremental classifier decisions; it **does not identify the cause**.

**Limitations targeted**

| ID | Original QKSR issue | v1.1 intervention | Unresolved risk |
|---|---|---|---|
| L1 | Quantum similarity is not cosine-head decision margin | Risk *weights* multiply a loss on actual classifier cosine margins | Risk scores may not predict decision conflicts |
| L2 | `old_proj` PQK separation has no direct gradient to current adapter | New margin loss depends on `z=f_t(x)` | New-image gradients can still damage old classes |
| L3 | Uniform repulsion of every old prototype | Pair-dependent PQK weights preserve a nonzero uniform baseline | Quantum weights may be redundant versus margin violations |
| L4 | No explicit old→new decision safeguard in new-image-only margin | Detached transported old-anchor guard trains the **new head** | Anchor may be wrong; guard may be erased by subsequent CA |
| L5 | Frozen base metric shifts domain as encoder adapts | Evaluate risk and old prototypes in same **previous snapshot space** | This does not stop long-horizon metric domain shift |

**Hypotheses (do not write them as conclusions):**

- **H-Margin:** Direct old–new margin training improves the new/old balance relative to the original separation loss.
- **H-Guard:** Old-anchor guard improves old-class retention relative to margin alone, without unacceptable plasticity loss.
- **H-Selector:** PQK risk provides useful **pairwise weighting** beyond constant, shuffled, cosine, projected RBF and MLP controls.
- **H-Cal:** Monotone calibration improves *weight shaping / loss dynamics* over raw PQK **without improving within-sample or global ranking of the same PQK scores**.

## 2. Task lifecycle, tensors, and identities

At task `t=0`, run **unchanged** base QKSR. For every task `t>0`, the input is exclusively images `x` of new classes `C_t`, with global labels `y`. No exemplar images from old classes enter adaptation.

| Symbol | Shape | Contract |
|---|---|---|
| `z = f_t(x)` | `[B,d]`, `d=768` | Current features; gradient reaches current adapters |
| `z_minus = f_(t−1)(x)` | `[B,d]` | Frozen snapshot features of the **same** new images |
| `mu_old` | `[C_old,d]` | Stored **estimated** old means tied by metadata to the previous encoder snapshot; not ground truth current means |
| `T_omega` | `[*,d]→[*,d]` | Existing RSIAT `old_ae`, trainable from alignment |
| `W_old`, `W_new` | `[...,d]` | Cosine head weight rows selected by global class IDs; ordering must not be assumed contiguous |
| `s(z)` | `[B,C_seen]` | Raw **normalized cosine** scores, before CosFace scale or label margin |
| `a=K_Q(z_minus,mu_old)` | `[B,C_old]` | Frozen base PQK similarity; no gradient |
| `r` | `[B,C_old]` | Raw or monotonically calibrated bounded risk **score**, not guaranteed probability |
| `v` | `[B,C_old]` | Nonnegative cosine-margin violations |
| `anchor=stopgrad(T(mu_old))` | `[C_old,d]` | Surrogate old-class features for guard *during adaptation only* |

**Key metadata:** Save `task_index`, `class_order`, `old_global_ids`, `new_global_ids`, `snapshot_encoder_id`, `means_encoder_id`, `metric_base_checkpoint_hash`, and `calibrator_fit_id` with diagnostics. Runtime must assert `means_encoder_id == snapshot_encoder_id` as a *version contract*. This alignment **does not** establish accurate old-class means.

### 2.1 Base (unchanged)

\[
 z_i=f_{\phi_0}(x_i),\quad
 a_i=\pi\tanh(W\operatorname{LN}(z_i)+b),\quad
 K_Q(z_i,z_j)=e^{-\gamma_Q\,D_R(z_i,z_j)}.
\]

`P_eta` projects to Ry angles, the Ry/CNOT statevector simulator is followed by local 1-/2-qubit RDM features, then squared Frobenius distance and an RBF. Keep base CosFace+steering and PQK parameter learning as-is, and **freeze** the resulting PQK parameters, including its bandwidth, for all subsequent tasks.

**After base:** persist the usable base checkpoint, frozen metric configuration/bandwidth, class moments and class-order metadata. R6 additionally fits a calibrator using **base training data only**, as specified in §3. The calibrated mode does not change base PQK weights or adapter weights.

## 3. Quantum risk score and calibration

### 3.1 Previous-space risk teacher (R5, R6)

For new image `i` and old class `c`:

\[
\boxed{a_{ic}=\operatorname{stopgrad}\big(K_{Q,\eta_0}(z_i^-,\mu_c^{t-1})\big).}
\]

Both features are *intended* to live in the feature space associated with `f_(t−1)`. Base-trained metric `eta_0` is frozen and **must not recalibrate its RBF bandwidth** at an incremental task. Cache old-prototype RDMs per task (invalidate when means change at task boundary); encode each new sample once per batch. `a` must be finite and within numerical tolerance of `[0,1]`.

- **R5 raw:** `r = clamp(a, 0, 1)`.
- **R6 calibrated:** `r = sigmoid((softplus(u)+1e-6)*a + b)`; `u,b` are learned **only at base-data calibration fitting**, then frozen.
- **Rank theorem:** For strictly increasing scalar `g`, `a_ic>a_ik` iff `g(a_ic)>g(a_ik)`. Hence **AUROC, ranking AUPRC, top-k ordering, and Spearman for the same targets cannot improve under this calibration**, except finite precision and ties. Evaluate R6 for proper-score fit, score distribution, gradient weighting and downstream accuracy, **not better ranking**.

### 3.2 Precisely defined calibration protocol

**Target is a base-task proxy, not observed future forgetting.** For sample `i` with true label `y_i` and every wrong base class `c`:

\[
 u_{ic}=\mathbf1\big[m_d+s_c(z_i)-s_{y_i}(z_i)>0\big].
\]

Here `s` is the same normalized cosine score API used at incremental adaptation. `m_d` is a separately logged *decision-proxy margin*; do not confuse it with CosFace margin, `m_x`, `m_o`, or PQK bandwidth.

**Data protocol (fixed before test evaluation):**

1. Select only original base **training** images, stratify by class, deterministically partition sample IDs into `P` prototype-fit (60%), `F` calibrator-fit (20%), `V` calibrator-evaluation (20%); record IDs/hash/seed. `P`, `F`, `V` must be pairwise disjoint.
2. Compute per-class calibration prototypes solely from `P`, using deterministic evaluation transforms. Get `z` and base-head cosine scores for `F`/`V` with the frozen base encoder. For each `F`/`V` query, exclude its true class from the negative-pair candidate set.
3. Fit the scalar monotone `g` on `F` negative pairs by **unweighted BCE** with all observed pairs by default (or deterministic uniform subsampling with tracked inverse-probability correction). **Do not use class-balanced BCE weights** and then interpret the sigmoid directly as an uncorrected event probability. If sampling/weights change prevalence, adjust the model/validation target or call it only a risk score.
4. Evaluate once on `V` without fitting or selecting parameters from `V`: NLL, Brier score, ECE with fixed bins, positive rate, active-risk distributions, AUROC/AUPRC, and relation to constant-prevalence calibration. Record paired raw-versus-calibrated ranking invariance.
5. Validate finite `u,b`, both classes present in fit/evaluation pairs, no total collapse and compatibility with the saved base metric. If not, label **R6 unavailable / calibration gate failed**. Never silently relabel an R5 run as R6 or use incremental test to fit.

**Honesty about reuse of checkpoints:** A pre-existing base checkpoint may already have trained on images in `F` and `V`. The splits above are independent only for **prototype/calibrator fitting**; they are *not* held out from representation training. Report results explicitly as **in-sample encoder, calibrator-held-out**. A claim of genuine out-of-base-training generalization requires a new base checkpoint trained with a disjoint held-out split. Calibration positives may be very rare if the base classifier is highly accurate; report this failure mode instead of fabricating targets.

**Calibration interpretation:** Even under unweighted BCE, output quality for new-task conflicts can shift with dataset/task prevalence. Call `r` a **calibrated base-proxy confusion score**, not the probability of future interference, unless additionally validated.

## 4. Incremental loss: fully specified mathematics

### 4.1 Original classification and alignment (unchanged)

\[
L_{\mathrm{cls,new}}=\operatorname{CosFaceCE}(s(z)\text{ restricted to new classes},y),
\qquad
L_{\mathrm{align}}=\frac{1}{Bd}\sum_{i=1}^B\lVert z_i-T_\omega(z_i^-)\rVert_2^2.
\]

Use current RSIAT scaling, CosFace margin, optimizer groups and schedule exactly as legacy. `T_omega` receives gradients from alignment, old encoder is frozen. Avoid reapplying CosFace margins/scales when computing the new cross-task loss below.

### 4.2 Risk-weighted cross-task margin (R2–R10)

All scores are raw cosine in `[-1,1]`:

\[
v_{ic}=[m_x+s_c(z_i)-s_{y_i}(z_i)]_+,
\qquad
w_{ic}=1+\lambda_r\operatorname{stopgrad}(r_{ic}),\quad0\le r_{ic}\le1.
\]

**Default stable normalization (v1.1):**

\[
\boxed{L_{\mathrm{cross}}=\frac{1}{B\,C_{\mathrm{old}}}
\sum_{i=1}^{B}\sum_{c\in C_{\mathrm{old}}}w_{ic}\,v_{ic}^{2}.}
\]

If `C_old=0`, return a differentiable zero/skip the branch. This denominator is **fixed for the entire task**, independent of which pairs are active; `v²` already provides zero contribution for non-violating pairs. Do **not** use `sum(active)>0` or divide by the current active count in v1.1. The mapping is continuous at `v=0` and the gradient is zero there. Report active fraction and loss scale as `C_old` grows; do not change denominators between runs. A later v2 may examine a predeclared, task-fixed scaling rule, but that is a distinct ablation.

Risk score choices:

- **Uniform R2/R3:** `w=1`, independent of `risk_gain`.
- **R4 cosine:** `r_ic = clamp((1+cos(z_i^-, mu_c))/2,0,1)`; avoid using current classifier logits in this teacher score.
- **R5 PQK:** `r=a`.
- **R6 calibrated PQK:** `r=g(a)`.
- **R7 projected RBF:** `r=exp(-gamma_R ||P_eta(z_i^-)-P_eta(mu_c)||²)` with base PQK's **frozen learned classical projector**, `gamma_R` calibrated on base training, no circuit.
- **R8 MLP:** base-feature-trained classical similarity selector, with train/fit objective and budget logged (§7.2).
- **R9 shuffled PQK:** permutation of the detached risk-score values **among active pairs within a batch**, seeded deterministically; preserves empirical active-pair risk histogram but disrupts pair identity. Handle zero/one active pair explicitly.
- **R10 constant PQK:** within each batch, replace risks on active pairs by their **detached mean over active pairs**; preserves the active-pair mean (hence average weight, not variance/histogram or exact weighted loss), but destroys pair ranking. Inactive pairs are immaterial because `v=0`.

**Important:** Merely setting `risk_gain=0` in R5/R6 reproduces the **uniform margin term**, not original QKSR separation. Additional `risk_gain>0` changes mean loss strength as well as selective weighting; R9/R10 are necessary controls. Record active-weight means/quantiles and `||∇_adapter L_cross||` to diagnose this confound.

### 4.3 Old-anchor guard — adaptation-only (R3–R10)

Transport the stored old prototypes with current residual projector but detach them:

\[
\bar\mu_c=\operatorname{stopgrad}\big(T_\omega(\mu_c^{t-1})\big).
\]

For **global** old class `c`, let `s_c(anchor_c)` be its cosine score and `max_{k∈C_new} s_k(anchor_c)` the largest new-class score. Define:

\[
\boxed{L_{\mathrm{guard}}=\frac{1}{C_{\mathrm{old}}}\sum_{c\in C_{\mathrm{old}}}
\left[m_o+\max_{k\in C_{\mathrm{new}}}s_k(\bar\mu_c)-s_c(\bar\mu_c)\right]_+^2.}
\]

The guard is **not** applied to old images and does not guarantee retention of old feature distributions. Implemented in **adaptation**, it does not remain an enforced constraint when the original Gaussian CA later refits classifier weights. Record *both pre-CA and post-CA* guard violation rates; never describe guard as preserved through CA by construction.

**Guard gradient policy:** `anchor.detach()`, `W_old` detached **within guard**, no gradient to adapter or `T_omega`; only new-head rows receive guard gradients. Anchors must be refreshed from updated `T_omega` **each training step** (or an explicitly documented stale-anchor ablation), never cached across a changing `T_omega`.

### 4.4 Final objective and mode switching

For `enabled=true`, `t>0`:

\[
\boxed{L_{\mathrm{inc}}=L_{\mathrm{cls,new}}+\beta L_{\mathrm{align}}+
\lambda_x L_{\mathrm{cross}}+\lambda_o L_{\mathrm{guard}}.}
\]

The legacy `L_sep` **MUST NOT be evaluated or added**. For `enabled=false`, dispatch to the **original implementation unchanged**, honoring its own `use_quantum_kernel_inc`, `q_inc_pair`, `gamma`, etc.

| Control state | Exact semantics |
|---|---|
| `enabled=false` | Legacy QKSR/RSIAT losses, **including** original separation if old flags enable it; no qbridge side effects |
| `enabled=true; lambda_cross=0; lambda_guard=0` | **New-class classification + alignment only** (explicit ablation), **not** legacy baseline |
| `enabled=true; risk_gain=0; lambda_cross>0` | Uniform cross-task margin (plus guard only if `lambda_guard>0`) |
| `enabled=true; lambda_guard>0` | Old-anchor guard active, independent of risk selector |

**At task 0**, Q-BRIDGE is bypassed regardless of enabled flag: base runs legacy QKSR only.

## 5. Gradient/optimizer contract

| Path / objective | Current adapter | New-class head rows | Old-class head rows | `T_omega` | Frozen old encoder | PQK/calibrator |
|---|---:|---:|---:|---:|---:|---:|
| `L_cls,new` | Yes | Yes | No **from this loss** | No | No | No |
| `L_align` | Yes | No | No | Yes | No | No |
| `L_cross` | Yes | Yes | **No from new loss** | No | No | No |
| `L_guard` | No | Yes | **No from new loss** | No | No | No |
| Post-task CA | No | Existing CA policy | Existing CA policy | No | No | No |

**Do not `torch.no_grad()` the whole classifier forward in `L_cross`:** that would cut gradients to `z`. Compute raw scores using normalized `z`, with **only old weight rows detached** in additional loss branches. For guard, detach both `anchor` and old weight rows. Global `y` labels must be mapped by class IDs, **not** by positional assumptions like `[:, :C_old]` unless explicitly asserted valid.

**Optimizer subtlety:** Detaching old rows stops autograd contributions from the new losses; it does **not** guarantee old weights stay numerically fixed across `optimizer.step()` if the head is one Parameter with weight decay/momentum. Keep the original baseline optimizer policy; if the experiment claims literal freezing of old weights, implement/verify row-wise update masking and decay handling explicitly. Add separate *gradient* tests and *post-step numerical* tests.

## 6. Full R6 pilot configuration (new JSON schema to implement)

The repository uses JSON experiment configs; the following is a **drop-in override fragment**, **not** a claim of a runnable full experiment until Codex merges it with an existing task/dataset/model config, supplies a compatible existing base checkpoint and fitted calibrator artifact, and adds parser support. All positive values are **pre-registered pilot starting points**, not optimized results.

```json
{
  "use_quantum_kernel_base": true,
  "use_quantum_kernel_inc": false,
  "qbridge": {
    "enabled": true,
    "variant": "R6",
    "base_checkpoint_path": "<REQUIRED_EXISTING_BASE_CHECKPOINT>",
    "calibrator_artifact_path": "<REQUIRED_BASE_CALIBRATOR_ARTIFACT>",
    "risk_mode": "calibrated_pqk",
    "kernel_source": "base_frozen",
    "calibration": {
      "enabled": true,
      "fit_source": "base_train_only",
      "prototype_fit_fraction": 0.60,
      "calibrator_fit_fraction": 0.20,
      "calibrator_eval_fraction": 0.20,
      "split_seed": 1993,
      "proxy_margin": 0.10,
      "fit_loss": "unweighted_bce",
      "monotonic": "softplus_slope_sigmoid",
      "strict_require_fitted": true,
      "recalibrate_per_task": false
    },
    "risk_gain": 0.50,
    "margin_x": 0.10,
    "lambda_cross": 0.10,
    "cross_normalization": "mean_all_old_pairs",
    "margin_old": 0.10,
    "lambda_guard": 0.10,
    "guard_refresh": "each_step",
    "old_head_grad_in_added_losses": false,
    "detach_risk": true,
    "detach_guard_anchors": true,
    "prototype_source": "previous_task_means",
    "prototype_cache": "per_task",
    "legacy_incremental_separation": "replace",
    "preserve_original_ssca_ca": true
  }
}
```

**R6 is valid only when:** `enabled=true`, `variant=R6`, both required artifact paths resolve to compatible real files, `risk_gain>0`, `lambda_cross>0`, `lambda_guard>0`, fitted compatible calibrator exists, PQK+calibrator frozen and legacy incremental separation not executed. **Fail loudly** on missing checkpoint/calibrator/metadata or a contradictory legacy `use_quantum_kernel_inc=true` setting, rather than running an accidental R5/R3 or silently executing two separation losses.

**R5 override:** `variant=R5`, `risk_mode=raw_pqk`, `calibration.enabled=false`, retaining the same positive cross/guard hyperparameters. **R3 override:** `variant=R3`, `risk_mode=uniform`, `risk_gain=0`. Every selector uses the same `m_x`, `m_o`, `lambda_cross`, `lambda_guard` and normalization. If hyperparameters are tuned, tune on disjoint validation only and report the complete search budget.

## 7. Locked ablation matrix and classical-control training

All R1–R10 variants branch from a paired **identical base QKSR checkpoint**. **R0** retains original QKSR incremental separation, from that checkpoint. A standalone original RSIAT (classical base) benchmark is an additional historical/reference baseline, **not** equivalent to R0.

| Run | Objective during incremental adaptation | Selector / distinct hypothesis |
|---|---|---|
| **R0** | Exact original QKSR classification + alignment + PQK `old_proj` separation | Baseline recovery |
| **R1** | **Replace** new-only CosFace with **all-seen-class CosFace**, keep `β L_align`, remove legacy separation; no cross margin or guard | Test old–new classification competition. Use global class IDs, same CosFace margin/scale; old-head rows detached for this replacement loss, with optimizer caveat in §5 |
| **R2** | `L_cls,new + βL_align + λ_x L_cross` | Uniform margin (`w=1`), guard off |
| **R3** | R2 `+ λ_o L_guard` | Classical margin + guard (common substrate) |
| **R4** | R3 with cosine risk `r=(1+cos)/2` | Simple classical selector |
| **R5** | R3 with **raw PQK** selector | **Q-BRIDGE Raw / Full MVP** |
| **R6** | R3 with **calibrated PQK** selector | **Q-BRIDGE Calibrated / Full v1** |
| **R7** | R3 with `rbf_proj` on base-learned angles (and base-only gamma fit) | Remove circuit, reuse same learned projection |
| **R8** | R3 with classical MLP selector trained on base features | Nonlinear classical capacity control |
| **R9** | R3 with **active-pair-shuffled raw PQK** selector | Retain active-pair risk histogram, destroy pair assignment |
| **R10** | R3 with **active-pair-constant raw PQK** selector | Match active-pair mean risk only; not histogram |

### 7.1 R1 implementation lock

- `R1` **replaces**, never adds, `L_cls,new` with all-seen CosFace CE on *new-task images only*.
- `L_align` persists with exactly baseline weight and reduction; no legacy `L_sep`, no cross loss, no guard.
- `W_old` should receive **no gradient from R1 CE** (use detached old weight rows; verify actual optimizer-step changes separately), while current adapters and new head do receive gradient.
- Classifier logits use original normalized cosine, CosFace margin and scale applied once to true-class global label. R1 is **not** called equivalent to a replay-based all-class CE.

### 7.2 R7/R8 training and fairness lock

- **R7 `rbf_proj`:** reuse the classical projector learned inside the *same base PQK checkpoint*; no quantum circuit, freeze projector; estimate RBF bandwidth on base-training features only using a documented deterministic method; do not fit from incremental eval/test. This is a **circuit ablation conditional on shared projection**, not a full matched-from-scratch classical baseline.
- **R8 MLP:** select a concrete architecture using existing `mlp_small` or `mlp_cap` implementations where possible; count parameters. Train the selector **offline from base train feature caches only**, while keeping base encoder/checkpoint fixed, under the **same label-aware pair/prototype steering objective** used for the PQK metric where architecturally applicable. Fix optimizer, batches, epochs, seed, and compute budget in manifest; freeze selector before increments. Clarify when objective differs. Capacity/compute mismatch disallows a quantum-specific advantage claim.
- Classical risk scores must map to `[0,1]` with a documented transform; run **matched calibration** in an extended control if interpreting R6's gain as quantum-specific. Primary R4/R7/R8 compare raw selectors to R5; compare calibration as an orthogonal factor (e.g. R4-cal, R7-cal, R8-cal), not by claiming R6 beats only uncalibrated controls.
- Truly identifying a *circuit-induced* gain requires a second protocol in which **PQK and classical metrics are both learned from base under matched training and resources**. R0–R10 alone, despite shared base checkpoint, mainly test downstream selector utility.

### 7.3 Interpretability of contrasts

| Contrast | What it isolates | What it does **not** prove |
|---|---|---|
| R3 − R2 | Add-on old-anchor guard | Real old-data margin preservation after CA |
| R5 − R3 | Raw PQK weighting **plus possible increase in mean margin strength** | Information-specific quantum benefit without R9/R10 |
| R5 vs R4/R7/R8/R9/R10 | Value of PQK assignment relative to controls, when scale/protocol matched | Quantum advantage or hardware speedup |
| R6 − R5 | Effect of monotone *weight shaping* | Improved pair ranking |
| R1 vs R2/R3 | All-class CosFace vs margin/guard | Apples-to-apples loss families without gradient checks |

## 8. Reference implementation pseudocode (explicit masks and detach)

The pseudocode uses **global-class-ID mapping** rather than positional slicing. An implementation may optimize these operations but cannot change semantics.

```python
# Task t>0. base_qkernel and cal are frozen, eval(); old_encoder is frozen.
# Meta: old_ids, new_ids are ordered GLOBAL class IDs; map labels to head row positions.
# mu_old[c] corresponds to old_ids[c] and previous_snapshot_id.

def scores_with_old_rows_detached(z, head, old_row_ids, seen_row_ids):
    z_unit = F.normalize(z, dim=-1)  # KEEP gradient into z
    W = head.weight_for_rows(seen_row_ids)
    # Build a row-wise mask based on GLOBAL -> HEAD ROW mapping.
    old_mask = torch.isin(seen_row_ids, old_row_ids)
    W_safe = torch.where(old_mask[:, None], W.detach(), W)
    return z_unit @ F.normalize(W_safe, dim=-1).T  # unscaled raw cosine

# Once per incremental task (no update to mu_old during adaptation):
assert mu_old_snapshot_id == old_encoder_snapshot_id
qproto_repr = frozen_pqk.encode_local_rdm(mu_old).detach()  # cache per task

for x, y_global in new_train_loader:
    z = current_encoder(x)  # [B,d], adapter trainable
    with torch.no_grad():
        z_minus = old_encoder(x)  # same images, same view
        qz_repr = frozen_pqk.encode_local_rdm(z_minus)
        a = frozen_pqk.similarity(qz_repr, qproto_repr)  # [B,C_old]
        risk = selector(a, z_minus, mu_old).clamp(0, 1).detach()
        # For R9/R10 only: substitute appropriately permuted/constant risk on active pairs later.

    projected = old_ae(z_minus)  # gradient to T_omega
    L_align = F.mse_loss(z, projected, reduction="mean")  # 1/(B*d)
    L_cls_new = original_new_class_cosface(z, y_global)

    scores = scores_with_old_rows_detached(z, head, old_row_ids, seen_row_ids)
    correct_pos = global_id_to_seen_position(y_global)
    old_pos = global_ids_to_seen_positions(old_ids)
    s_true = scores.gather(1, correct_pos[:, None])
    s_old = scores[:, old_pos]
    violation = (margin_x + s_old - s_true).relu()  # [B,C_old]
    active = violation > 0
    if variant == "R9":
        risk = shuffle_values_only_where_active(risk, active, seeded_rng)
    elif variant == "R10":
        risk = replace_active_values_with_active_mean(risk, active)
    weights = 1.0 + risk_gain * risk  # uniform mode: explicitly set weights=1
    C_old = len(old_ids)
    L_cross = (weights * violation.square()).sum() / (z.shape[0] * C_old)

    # Recompute anchors EACH step because old_ae changes during optimization.
    with torch.no_grad():
        anchor = old_ae(mu_old).detach()
    guard_scores = scores_with_old_rows_detached(anchor, head, old_row_ids, seen_row_ids)
    row_pos = global_ids_to_seen_positions(old_ids)
    new_pos = global_ids_to_seen_positions(new_ids)
    s_old_anchor = guard_scores.gather(1, row_pos[:, None]).squeeze(1)
    s_new_max = guard_scores[:, new_pos].max(dim=1).values
    L_guard = (margin_old + s_new_max - s_old_anchor).relu().square().mean()

    # enabled=true: NEVER add old L_sep; R1 has a separate all-class-CosFace branch.
    loss = L_cls_new + beta * L_align + lambda_cross * L_cross + lambda_guard * L_guard
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Continue unchanged legacy post-task steps: SSCA -> new stats -> Gaussian CA -> eval -> snapshot.
# After CA, recompute old-anchor guard metrics for diagnostics; no guard optimization in CA.
```

Implementation notes: `torch.isin` inputs must be integer tensors on the same device; `global_id_to_seen_position` and `global_ids_to_seen_positions` are required interfaces, not assumed existing APIs. The pseudocode computes PQK even for uniform mode for clarity; production code **must skip unneeded selector computation**. For R1 use a separate explicitly specified forward with all-class CosFace. Empty old/new subsets must be guarded before indexing. Adapt integration to source head shape and optimizer groups without changing science.

### 8.1 Performance & lifecycle specifics

- Cache RDM of old means once per task, because `mu_old` does not change during adaptation; do not cache `T(mu_old)` across steps while T is trainable.
- Preserve base kernel bandwidth and any numerical flooring exactly; old-prototype cache must be invalidated after SSCA at the task boundary.
- The guard operates during adaptation only; record pre/post-CA metrics and do not silently enforce it during Gaussian CA.
- `risk_gain=0` or `lambda_cross=0` should bypass unnecessary frozen circuit calls at runtime.
- Base is bypassed under `qbridge.enabled` at task 0; use frozen base `eta_0` at increments and never train it by mistake.

## 9. Verification: unit + integration acceptance tests

**A. Baseline recovery (must pass before experiments)**

1. `qbridge.enabled=false`, legacy `use_quantum_kernel_inc=true`: compare forward loss components, head scores and gradients against original QKSR code on deterministic fixture — original separation must be present.
2. `qbridge.enabled=false`, legacy separation off/classical: legacy behavior unchanged.
3. `qbridge.enabled=true`, `lambda_cross=lambda_guard=0`: check exact `L_cls,new+βL_align`, and **assert no separation call**, not baseline equivalence.
4. At task 0: model/loss/effective parameters match original QKSR base run irrespective of `enabled`.

**B. Numerics and mathematics**

5. Cross-score shape `[B,C_old]`; risk scores finite in `[0,1]`, tensor rows align global class IDs in permuted class order. Test `B=1`, `C_old=1`, and task transitions.
6. **Fixed-normalization continuity regression:** set `B=1`, `C_old=2`, `w=(1,1)`, `v=(0.2,eps)`; verify `L_cross → 0.02` as `eps→0+` **and** at `eps=0` (no jump). This directly catches `count_active` regression.
7. `risk_gain=0`: PQK-weighted margin equals uniform margin on same scores. `lambda_cross=0`: no cross-loss gradient.
8. R6 monotone calibration: strictly preserves ranks except ties/floating-point; verify AUROC/rank invariance on synthetic data and finite Brier/NLL.
9. R9: multiset of active risk scores unchanged but pair assignments permuted (for at least two active pairs); R10: active mean unchanged, variance/histogram not required to match. Seed reproducibility.
10. No active violations: loss and gradients finite and zero; `C_old=0`: branch bypassed; no division by zero.

**C. Gradient ownership and model updates**

11. From `L_cross` alone: nonzero gradient can reach current adapter `z`, new-head rows; no grad to PQK, calibrator, teacher, `T_omega`, old-head **from the new branch**. Do not accidentally detach old logits from the feature path.
12. From `L_guard` alone: gradient to new-head rows, no gradient to adapter/T/anchors/old-head. Test head-row optimizer updates versus original baseline decay/momentum behavior separately.
13. From `L_align`: gradients reach adapter and `T_omega`; teacher frozen.
14. CA retains the baseline optimizer/head policy and does not include new QBRIDGE losses.

**D. Data/protocol integrity**

15. Base calibration splits P/F/V disjoint; no incremental eval/test data in fitting or bandwidth calibration; metadata stored.
16. R6 fails on missing/incompatible calibrator; no silent raw fallback; PQK RBF gamma unchanged at incremental task 1, 5, 9.
17. Previous means/snapshot IDs match by task; record whether means are estimated (not true current means).
18. Inspect dataset accuracy at adaptation-end **and** post-CA; guard may regress after CA and must be reported.
19. Functional test: inference graph contains encoder+adapter+head only, not PQK, risk selector, old encoder, T or Gaussian sampler.
20. Compute test: log per-task additional runtime, GPU peak memory and PQK encoding count.

**Definition of done:** All A–D tests pass; the agent outputs code diffs, per-run JSON config overlays, deterministic fixture tests, calibration diagnostics and a truthful experiment manifest. Test passage does **not** mean benchmark improvement.

## 10. Experimental protocol and reports

- **Paired runs:** seeds `1993, 2015, 2026`, exactly matched base checkpoint per seed and dataset, matched class order, train/validation/test splits, optimizer settings, epochs and CA budget.
- **Prioritize ImageNet-A** for pilot then CIFAR100 confirmation; report task 1 and final tasks, not only final pooled accuracy.
- **Primary:** AIA, final Top-1, per-task Top-1, old/new accuracy and forgetting using an explicitly recorded convention; report per-seed **paired deltas** and uncertainty rather than solely mean±std.
- **Mechanistic:** `L_cross` active rate, guard violations pre/post CA, loss scale, risk histogram on active pairs, conditional conflict recall/top-k, raw vs calibrated Brier/ECE/NLL, gradient norm per module, agreement with cosine/RBF/MLP.
- **Fairness:** Control for boosted mean margin intensity; report R9/R10, and for strong quantum-specific claims compare similarly calibrated classical selectors in a separate factorial set.
- **Configuration integrity:** Each output stores immutable base checkpoint hash, selector hash, calibrated params/fit split IDs, qbridge config and legacy settings. No tests performed on altered dataset protocols without new labels.
- **Operational:** Kaggle offline, `/kaggle/input/` assets manually provided, outputs in `/kaggle/working/`; console only `[task, epoch, AIA, old/new acc, elapsed time, status]`, detailed diagnostics CSV/JSON; save full model checkpoint only at last task (reuse existing saved base checkpoint for branching; do not repeatedly rewrite it).

### Pre-registered go/no-go

- **Engineering GO:** baseline regression, gradients, metadata and calibration tests pass.
- **Method GO:** R5 or R6 improves a meaningful metric/tradeoff over R3, *without unacceptable drops* in final accuracy; examine paired seeds before concluding.
- **PQK-specific GO:** PQK scores/weights improve outcomes or conflict selection relative to matched cosine/RBF/MLP/constant/shuffled controls. A gain attributable only to R3 (margin+guard) is **not** quantum-specific.
- **R6-specific GO:** R6 improves downstream behavior over R5 via **weight shaping**; never claim improved similarity ranking solely from monotone calibration.
- **STOP/RENAME:** No reliable gain beyond classical controls: publish/report it as a negative ablation or a classical decision-aware method, not a quantum advantage. This conclusion applies to **v1.1 frozen base PQK**, not to every possible PQK/meta-learned future metric.

## 11. Risk register and explicitly out-of-scope work

| Risk | Diagnostic | Mitigation in v1.1 / status |
|---|---|---|
| Base PQK metric is separability-oriented, not future-risk-trained | Risk/conflict correlations from pseudo splits and real subsequent evaluation | Diagnostic; **not solved** |
| Calibrator mono-map cannot change ranking | Verify identical rank metrics | Correct claims; compare R6−R5 as intensity shaping |
| Late-task metric domain shift | Risk stability across task index | Observe; **no PQK retraining** |
| Old means differ from true class distributions | Per-class guard failure cases, if available | Surrogate caveat; no old exemplar access |
| Cross margin favors new samples | Old/new accuracy, old→new confusion | Guard and monitoring; no retention guarantee |
| CA can invalidate guard | Measure pre/post-CA guard and accuracy | Leave CA unchanged in v1.1 |
| Classifier old-head decay through optimizer | Compare old rows before/after optimizer step | Respect baseline; implement row freeze only if required for intended ablation |
| Metric compute cost and memory | Wall time, VRAM, qubit counts | Cached prototype RDM, skip unused branches |
| External novelty | Related-work comparison | Avoid priority/novelty claim before literature assessment |

**Not in v1.1:** new base decision-aware metric training, meta-learning/bilevel optimization, learned quantum observable memory, covariance transport, SSCA fixes, CA loss changes, post-CA guard optimization, hardware quantum advantage claims. Those are distinct follow-on experiments requiring separate specs and controls.

## 12. Required Codex deliverables

1. **Source audit** mapping actual classes/functions/config keys, including any discrepancies with the documented RSIAT/QKSR pipeline.
2. **Code patch** behind `qbridge.enabled` with backward-compatible legacy behavior.
3. **Locked config overlays** R0–R10 (plus R6 validation-ready JSON overlay) and calibrated/classical metric training manifests.
4. **Automated tests** A–D and concise command to run tests offline.
5. **Experiment instructions** for the exact Kaggle input/output layout, seed mapping and final-checkpoint policy.
6. **Structured reports**: no fabricated metrics; distinguish implementation verified, untested hypotheses and measured results.

## Appendix A — Precise differences from v1.0

- Corrected baseline recovery: `enabled=false` only; zero new weights with `enabled=true` is an explicit **classification+alignment ablation**.
- Corrected calibration claims: scalar increasing sigmoid **cannot improve ranking**, only scoring/weight intensity.
- Switched cross-task normalization from **active-count** to **fixed mean over all old-class pairs**; provided continuity regression test.
- Explicit R6 **enabled** full config with strictly positive guard/cross/risk weights and strict fitted-calibrator checks.
- Clarified guard is **adaptation-only** and can be undone by unchanged CA.
- Locked R1 semantics, cosine risk mapping, R7/R8 fit/freeze provenance, selector scope and two distinct randomization controls R9/R10.
- Added global-ID index mapping, row-wise gradient policy, optimizer decay caveat, provenance/version assertions, no-leakage calibration fitting and calibrated-classical control requirements.
- Replaced unconditional quantum-benefit rhetoric with exact, falsifiable v1.1 hypotheses and go/no-go criteria.
