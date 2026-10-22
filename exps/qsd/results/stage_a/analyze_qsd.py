"""Deeper analysis of QSD stage A predictions (argmax only). Usage: python -I exps/qsd/results/stage_a/analyze_qsd.py exps/qsd/results/stage_a"""
import glob
import os
import sys

import numpy as np
sys.stdout.reconfigure(encoding="utf-8")

ROOT = sys.argv[1]
HEADS = ("ridge_stats", "pgm_channel", "pgm_channel_hier", "pgm_fuse", "pgm_nonlinear")
SIZE = 20


def runs(ds):
    return sorted(glob.glob(os.path.join(ROOT, "*", "logs", "adapter", ds, "0", "20", "*_stats")))


def load(run, t):
    z = np.load(os.path.join(run, "qsd_predictions_task_%d.npz" % t))
    return {k: z[k] for k in z.files}


def decomp(pred, y):
    tp, tt = pred // SIZE, y // SIZE
    wrong = pred != y
    return (100 * (wrong & (tp == tt)).mean(), 100 * (wrong & (tp > tt)).mean(),
            100 * (wrong & (tp < tt)).mean())


def mean_sd(rows):
    a = np.array(rows)
    return a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1:])


for ds in ("imagenetr", "imageneta"):
    rs = runs(ds)
    print("=" * 30, ds, len(rs), "runs")
    T = 10
    # 1. delta top1 vs linear per stage t
    print("\n[1] Δ top1 vs linear at each stage t (mean over runs)")
    print("head".ljust(18) + "".join("t%-6d" % t for t in range(T)))
    for h in HEADS:
        rows = []
        for r in rs:
            row = []
            for t in range(T):
                z = load(r, t)
                row.append(100 * ((z[h] == z["target"]).mean() - (z["linear"] == z["target"]).mean()))
            rows.append(row)
        m, _ = mean_sd(rows)
        print(h.ljust(18) + "".join("%+6.2f " % v for v in m))
    # 2. final-stage per-task delta
    print("\n[2] Final stage: Δ accuracy per true task (mean over runs)")
    print("head".ljust(18) + "".join("T%-6d" % t for t in range(T)))
    lin_rows = []
    for r in rs:
        z = load(r, T - 1)
        y = z["target"]
        lin_rows.append([100 * (z["linear"][y // SIZE == k] == y[y // SIZE == k]).mean() for k in range(T)])
    print("linear(abs)".ljust(18) + "".join("%6.2f " % v for v in mean_sd(lin_rows)[0]))
    for h in HEADS:
        rows = []
        for r in rs:
            z = load(r, T - 1)
            y = z["target"]
            rows.append([100 * ((z[h][y // SIZE == k] == y[y // SIZE == k]).mean()
                                - (z["linear"][y // SIZE == k] == y[y // SIZE == k]).mean()) for k in range(T)])
        print(h.ljust(18) + "".join("%+6.2f " % v for v in mean_sd(rows)[0]))
    # 3. error decomposition final stage
    print("\n[3] Final-stage error decomposition (% of test set): within | inter->newer task | inter->older task")
    for h in ("linear",) + HEADS:
        rows = [decomp(load(r, T - 1)[h], load(r, T - 1)["target"]) for r in rs]
        m, s = mean_sd(rows)
        print("  %-17s within %5.2f | ->newer %5.2f | ->older %5.2f | total %5.2f" % (h, m[0], m[1], m[2], m.sum()))
    # 4. t=0: pure within-task classifier quality (no drift, no inter-task)
    print("\n[4] Stage t=0 (one task only, no drift): top1")
    for h in ("linear",) + HEADS:
        v = [100 * (load(r, 0)[h] == load(r, 0)["target"]).mean() for r in rs]
        print("  %-17s %.2f ± %.2f" % (h, np.mean(v), np.std(v, ddof=1)))
    # 5. within-task accuracy conditional on correct task (final stage)
    print("\n[5] Final stage: task-prediction acc | within-task acc given correct task")
    for h in ("linear",) + HEADS:
        tp_rows, wt_rows = [], []
        for r in rs:
            z = load(r, T - 1)
            y, p = z["target"], z[h]
            ok = p // SIZE == y // SIZE
            tp_rows.append(100 * ok.mean())
            wt_rows.append(100 * (p[ok] == y[ok]).mean())
        print("  %-17s task %5.2f | within|task %5.2f" % (h, np.mean(tp_rows), np.mean(wt_rows)))
    # 6. linear vs pgm_channel disagreements and what fuse does
    print("\n[6] Final stage: linear vs pgm_channel disagreement; fuse choice")
    for r in rs:
        z = load(r, T - 1)
        y, L, P, F = z["target"], z["linear"], z["pgm_channel"], z["pgm_fuse"]
        dis = L != P
        lr, pr = (L == y) & dis, (P == y) & dis
        both = dis & (L != y) & (P != y)
        f_l, f_p = (F == L) & dis, (F == P) & dis
        f_other = dis & ~f_l & ~f_p
        print("  %-14s dis %5.2f%% | lin right %5.2f pgm right %5.2f neither %5.2f | fuse=lin %5.2f fuse=pgm %5.2f other %4.2f"
              " | fuse acc on dis %5.2f | agree acc lin %5.2f" % (
                  os.path.basename(r)[:14], 100 * dis.mean(), 100 * lr.mean(), 100 * pr.mean(), 100 * both.mean(),
                  100 * f_l.mean(), 100 * f_p.mean(), 100 * f_other.mean(),
                  100 * (F[dis] == y[dis]).mean(), 100 * (L[~dis] == y[~dis]).mean()))
    # 7. where fuse fixes/breaks: by error type of linear
    print("\n[7] Final stage: fuse fixes (lin wrong->fuse right) / breaks, split by linear's error type")
    rows = []
    for r in rs:
        z = load(r, T - 1)
        y, L, F = z["target"], z["linear"], z["pgm_fuse"]
        inter = L // SIZE != y // SIZE
        fix, brk = (L != y) & (F == y), (L == y) & (F != y)
        brk_inter = brk & (F // SIZE != y // SIZE)
        rows.append([100 * (fix & inter).mean(), 100 * (fix & ~inter).mean(), 100 * brk_inter.mean(),
                     100 * (brk & ~brk_inter).mean()])
    m, _ = mean_sd(rows)
    print("  fix inter %.2f | fix within %.2f | break->inter %.2f | break->within %.2f" % tuple(m))
    # 8. old classes vs new (by stage) for fuse/hier
    print("\n[8] Δ on old vs new classes, averaged over stages t>=1")
    for h in HEADS:
        old, new = [], []
        for r in rs:
            for t in range(1, T):
                z = load(r, t)
                y = z["target"]
                o, n = y < t * SIZE, y >= t * SIZE
                old.append(100 * ((z[h][o] == y[o]).mean() - (z["linear"][o] == y[o]).mean()))
                new.append(100 * ((z[h][n] == y[n]).mean() - (z["linear"][n] == y[n]).mean()))
        print("  %-17s old %+5.2f | new %+5.2f" % (h, np.mean(old), np.mean(new)))
