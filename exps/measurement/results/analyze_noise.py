import sys, glob, numpy as np
from pathlib import Path
def runs(pattern):
    out = {}
    for d in sorted(glob.glob(pattern)):
        files = sorted(Path(d).glob("predictions_task_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))
        out[Path(d).name.split("_1993")[0]] = [np.load(f) for f in files]
    return out
for name, pat in [("ImageNet-R", "exps/measurement/results/logs/adapter/imagenetr/0/20/noise*_stats"),
                  ("ImageNet-A", "exps/diagnostics/results/round2_imageneta_s1993/logs/**/diag_legacy*_stats")]:
    R = runs(pat) if "**" not in pat else {Path(d).name.split("_1993")[0]: [np.load(f) for f in sorted(Path(d).glob("predictions_task_*.npz"), key=lambda p: int(p.stem.split("_")[-1]))] for d in sorted(glob.glob(pat, recursive=True))}
    print("=====", name, list(R))
    agg = []
    for k, tasks in R.items():
        starts = [int(t["known_classes"]) for t in tasks]
        f = tasks[-1]; p, y = f["pred"].astype(int), f["target"].astype(int)
        to = lambda l: np.searchsorted(starts, l, side="right") - 1
        w = p != y; tp, ty = to(p), to(y)
        per_task = [float((p[ty == j] == y[ty == j]).mean()*100) for j in range(len(starts))]
        # task-ID-known accuracy proxy: correct OR wrong only within task -> errors within task remain
        # recency: share of all inter-task errors that point to the newest task
        inter = w & (tp != ty)
        newest = float((inter & (tp == len(starts)-1)).sum() / max(inter.sum(), 1) * 100)
        # stage-wise new vs old accuracy
        nv = []
        for t, tk in enumerate(tasks[1:], 1):
            pp, yy, kc = tk["pred"], tk["target"], int(tk["known_classes"])
            old = yy < kc
            nv.append((round(float((pp[old]==yy[old]).mean()*100),1), round(float((pp[~old]==yy[~old]).mean()*100),1)))
        agg.append(dict(err=w.mean()*100, within=(w&(tp==ty)).mean()*100, older=(w&(tp<ty)).mean()*100, newer=(w&(tp>ty)).mean()*100, newest=newest, per_task=per_task, nv=nv))
        print(k, "err %.2f within %.2f to_older %.2f to_newer %.2f | inter->newest task %.0f%%" % (agg[-1]["err"], agg[-1]["within"], agg[-1]["older"], agg[-1]["newer"], newest))
    pt = np.mean([a["per_task"] for a in agg], 0)
    print("final acc by task (mean reps):", np.round(pt, 1))
    print("old/new acc per stage (rep0):", agg[0]["nv"])
    # per-image stability across reps at final task
    C = np.stack([t[-1]["pred"] == t[-1]["target"] for t in R.values()])
    allc, allw = C.all(0).mean()*100, (~C).all(0).mean()*100
    print("images always right %.1f%% | always wrong %.1f%% | flip %.1f%%" % (allc, allw, 100-allc-allw))
    # pairwise discordance
    n = len(C); disc = [ (C[i]!=C[j]).mean()*100 for i in range(n) for j in range(i+1,n)]
    print("pairwise discordant images %.2f%% (n=%d)" % (np.mean(disc), C.shape[1]))
