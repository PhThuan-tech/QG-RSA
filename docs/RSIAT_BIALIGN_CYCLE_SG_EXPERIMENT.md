# RSIAT BiAlign + Cycle-SG

## Scientific status

**[IMPLEMENTATION ABLATION]** BiAlign + Cycle with stop-gradient cycle inputs; cycle regularizes P_t/D_t only and does not directly update current representation.

Research question: does cycle consistency help through projector stabilization, or through the additional cycle gradient into the current representation in the validated BiAlign+Cycle arm?

This arm is based exactly on commit `492f3f0a74904163bb9e71ce99187fd501982a56`. It is an ablation of the local BiAlign+Cycle implementation, not an exact reproduction of BiCyc.

## Exact mechanism

Let `z_old` be the frozen old representation, `z_new` the current representation, `P_t` the existing RSIAT `old_ae`, and `D_t` the existing BiAlign `reverse_projector`.

```text
L_fwd       = MSE(P_t(stopgrad(z_old)), stopgrad(z_new))
L_back      = MSE(D_t(z_new), stopgrad(z_old))
L_bialign   = L_fwd + L_back

L_cycle_new = MSE(P_t(D_t(stopgrad(z_new))), stopgrad(z_new))
L_cycle_old = MSE(D_t(P_t(stopgrad(z_old))), stopgrad(z_old))
L_cycle     = L_cycle_new + L_cycle_old

L_repr = beta * L_bialign + gamma * L_orth + lambda_cycle * L_cycle
```

The only mechanism delta from `bialign_cycle` is the detach at the **input of D_t** in `L_cycle_new`:

```text
bialign_cycle:    P_t(D_t(z_new))
bialign_cycle_sg: P_t(D_t(stopgrad(z_new)))
```

The target remains detached in both modes. `L_back`, `L_cycle_old`, `L_orth`, weights, initialization, optimizer, lifecycle, statistics, SSCA, and CA are unchanged.

## Gradient routing

| Isolated loss | P_t | D_t | current representation | old model |
|---|---:|---:|---:|---:|
| `L_fwd` | yes | no | no | no |
| `L_back` | no | yes | yes | no |
| `L_cycle_new` | yes | yes | no | no |
| `L_cycle_old` | yes | yes | no | no |
| `L_cycle` | yes | yes | no | no |

The full objective still updates the current adapter through classification and `L_back`; Cycle-SG itself cannot update it. A cycle-only gradient diagnostic is logged in smoke mode separately from full-objective gradients.

## Controlled configuration

`exps/RSIAT_BiAlign_Cycle_SG.json` is copied from the validated `exps/RSIAT_BiAlign_Cycle.json`. Only experiment identity/output fields and `bicyc_mode="bialign_cycle_sg"` differ. `lambda_cycle=1.0` and all scientific settings remain identical.

The common task0 checkpoint must have SHA-256:

```text
d13bd496019dc5739eefd709c6bf60e28506577dbbb41b50ee81a5e109e55ea0
```

## Path isolation

Cycle-SG reuses BiAlign P_t/D_t initialization, D_t identity reset, optimizer groups, and strict checkpoint state. It uses official RSIAT statistics, SSCA, and CA. It does not instantiate Track-B `forward_transport`/`backward_transport`, run Stage II, or transport statistics.

## Interpretation boundary

The task0→task1 one-epoch smoke is implementation evidence only. It checks finite optimization, exact gradient routing, official statistics/CA execution, and strict checkpoint reload. It is not evidence that Cycle-SG improves accuracy. A full B0I10 run requires a separate decision after implementation validation.
