# RSIAT BiAlign + Cycle controlled experiment

## Scope

- Base commit: `cc90d3bc5248193c38b4546e806495085edeb5eb`
- Branch: `exp/rsiat-bialign-cycle`
- New mode: `bialign_cycle`
- Research sequence: official RSIAT → BiAlign → BiAlign + Cycle

**[IMPLEMENTATION ADAPTATION]:** This is a minimal cycle-consistency extension of the existing BiAlign mode, not the Track-B/BiCyc transport pipeline.

## Losses

Existing BiAlign is unchanged:

```text
L_fwd     = MSE(P_t(z_old.detach()), z_new.detach())
L_back    = MSE(D_t(z_new), z_old.detach())
L_bialign = L_fwd + L_back
```

The new mode adds:

```text
L_cycle_new = MSE(P_t(D_t(z_new)), z_new.detach())
L_cycle_old = MSE(D_t(P_t(z_old.detach())), z_old.detach())
L_cycle     = L_cycle_new + L_cycle_old

L_repr = beta * L_bialign + gamma * L_orth + lambda_cycle * L_cycle
```

`lambda_cycle=1.0` in `exps/RSIAT_BiAlign_Cycle.json`.

## Gradient-routing contract

| Isolated term | P_t | D_t | current adapter / z_new | frozen old model |
|---|---:|---:|---:|---:|
| `L_fwd` | yes | no | no | no |
| `L_back` | no | yes | yes | no |
| `L_cycle_new` | yes | yes | yes | no |
| `L_cycle_old` | yes | yes | no | no |

Focused tests verify all four routes independently and verify a finite combined optimizer step.

## Preserved controls

`bialign_cycle` follows the existing BiAlign lifecycle:

- same RSIAT `old_ae` as P_t;
- same signed 768→64→768 identity-initialized `reverse_projector` as D_t;
- D_t resets per incremental transition with the same deterministic seed rule;
- D_t uses P_t's learning rate and weight decay;
- D_t checkpoint save and strict load are unchanged;
- no Track-B `forward_transport` or `backward_transport` is instantiated;
- no Stage-II transport fine-tuning runs;
- official class-statistics, SSCA, and CA paths remain active.

Plain `bialign` retains its prior loss exactly; cycle metrics are zero and cycle is disabled. A fixed-seed cross-worktree probe against the base commit verifies bitwise-identical plain-BiAlign losses and gradients.

## Logging

BiAlign-mode epoch records include:

- `loss_fwd`, `loss_back`, `bialign`;
- `cycle_new`, `cycle_old`, `cycle`;
- `lambda_cycle`.

For plain `bialign`, `cycle_new=cycle_old=cycle=lambda_cycle=0` in epoch-level reporting. Legacy official and Track-B mode schemas remain unchanged.

## Config and validation

```bash
PYTHON_BIN=/path/to/python
"$PYTHON_BIN" tools/verify_bialign_cycle_config.py
"$PYTHON_BIN" -u main.py --config exps/RSIAT_BiAlign_Cycle.json
```

The verifier checks the immutable shared task0 checkpoint/report hashes and fails unless the scientific config matches BiAlign except for `bicyc_mode=bialign_cycle` and `lambda_cycle=1.0` (plus output identity fields).

No full training run is part of this implementation change unless separately authorized.

## Validation result

- Full repository discovery: 42/42 tests passed.
- The tests cover all four isolated gradient routes, frozen-old-model behavior, matched D_t initialization, D_t transition reset, P_t/D_t optimizer scale, `lambda_cycle=0` equivalence, finite combined optimizer step, strict D_t checkpoint round-trip, official statistics routing, and absence of Track-B Stage II.
- Fixed-seed cross-worktree probes showed all five pre-existing modes numerically unchanged. Official/forward/bidirectional/cycle outputs had identical schemas; plain BiAlign had identical losses and gradients with only the requested added zero-valued `cycle` reporting field.
- The shared official task0 checkpoint strict-loaded under `bialign_cycle` before training.

## One-transition smoke

**[EVIDENCE — implementation only]:** The one-epoch task0→task1 smoke from the same immutable common task0 checkpoint completed with return code 0.

- Pre/post-CA Top1: `97.15 / 97.30`.
- Post-CA old/new Top1: `97.8 / 96.8`.
- `L_fwd=0.38797043`, `L_back=0.13636567`, `L_bialign=0.52433610`.
- `L_cycle_new=0.23914817`, `L_cycle_old=0.23720898`, `L_cycle=0.47635715`, `lambda_cycle=1.0`.
- First-backward gradient L2: current adapter `2.51933628`, P_t `0.03658778`, D_t `0.23255162`; all finite. The frozen old model had no gradient.
- Track-B A/D maps were absent; Stage II and statistics transport did not run. Official SSCA and official CA ran.
- Task1 checkpoint SHA-256: `f5d3ba24a6b937bf01c619373840f1cb03ac2578c6acfcdcc6f6ce92c8ef487a`.
- A fresh learner strictly restored P_t, D_t, network/task state, and official statistics with no missing transport state.

**[INFERENCE]:** The implementation is internally consistent enough for later controlled full-run evaluation. This smoke is not evidence that cycle consistency improves continual-learning performance.

## Fail-closed controls

The dedicated config verifier uses explicit runtime checks rather than Python `assert`, so validation remains active under `python -O`. A negative optimized-mode test with `lambda_cycle=999` was rejected as required. `bialign_cycle` checkpoint loading also rejects unexpected Track-B A/D state before constructing any transport map.
