"""AIA, final accuracy, forgetting and BWT for every QSD head (round A2).

Forgetting and BWT use the convention of trainer.py: mean over old tasks of
(best accuracy before the current stage - current accuracy), and of
(current accuracy - accuracy right after learning the task). The linear head
is checked against the forgetting that RSIAT logs itself.

  python -I exps/qsd/results/a2/metrics_table.py exps/qsd/results/a2
"""
import glob
import json
import os
import re
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]
HEADS = ("linear", "fuse_ridge_stats_b0p25", "fuse_pgm_stats_b0p5", "fuse_pgm_stats_hier_b1", "ridge_stats",
         "pgm_stats", "pgm_stats_oracle")
LABEL = {"linear": "RSIAT (linear)", "fuse_ridge_stats_b0p25": "SCF ridge", "fuse_pgm_stats_b0p5": "SCF PGM",
         "fuse_pgm_stats_hier_b1": "SCF PGM hier", "ridge_stats": "ridge alone", "pgm_stats": "PGM alone",
         "pgm_stats_oracle": "PGM oracle (control)"}
TASKS = re.compile(r"QSD tasks head=(\w+): (\[.*\])")
LOGGED = re.compile(r"Continual metrics after task 9: BWT=([-\d.]+), forgetting=([-\d.]+)")


def metrics(matrix):
    final = matrix[-1]
    old = range(len(final) - 1)
    forgetting = statistics.mean(max(row[j] for row in matrix[j:-1]) - final[j] for j in old)
    bwt = statistics.mean(final[j] - matrix[j][j] for j in old)
    return forgetting, bwt


runs = []
for log in sorted(glob.glob(os.path.join(ROOT, "*", "logs", "adapter", "*", "0", "20", "qsd2*.log"))):
    text = open(log, encoding="utf-8").read()
    rows = {}
    for head, values in TASKS.findall(text):
        rows.setdefault(head, []).append(json.loads(values))
    curves = {h: [float(x) for x in json.loads(m)] for h, m in re.findall(r"QSD curve head=(\w+): (\[.*\])", text)}
    logged = LOGGED.search(text)
    name = os.path.basename(log)
    run = {"dataset": log.split(os.sep)[-4], "heldout": not (log.split(os.sep)[-4] == "imagenetr" and "_1993_" in name),
           "logged_forgetting": float(logged.group(2)), "heads": {}}
    for head in HEADS:
        f, b = metrics(rows[head])
        run["heads"][head] = {"aia": statistics.mean(curves[head]), "final": curves[head][-1], "forgetting": f, "bwt": b}
    runs.append(run)

mismatch = [r for r in runs if abs(r["heads"]["linear"]["forgetting"] - r["logged_forgetting"]) > 0.01]
print("Kiểm tra: forgetting của head linear trùng log RSIAT ở %d/%d run" % (len(runs) - len(mismatch), len(runs)))


def ci(values):
    m = statistics.mean(values)
    if len(values) < 2:
        return m, None
    half = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}[len(values)] \
        * statistics.stdev(values) / len(values) ** 0.5
    return m, (m - half, m + half)


for ds, title in (("imagenetr", "ImageNet-R (6 run)"), ("imageneta", "ImageNet-A (5 run)")):
    group = [r for r in runs if r["dataset"] == ds]
    print("\n## " + title + "\n")
    print("| Head | AIA | Final | Forgetting ↓ | BWT ↑ |")
    print("|---|---:|---:|---:|---:|")
    for head in HEADS:
        cells = []
        for key in ("aia", "final", "forgetting", "bwt"):
            v = [r["heads"][head][key] for r in group]
            cells.append("%.2f ± %.2f" % (statistics.mean(v), statistics.stdev(v)))
        print("| %s | %s |" % (LABEL[head], " | ".join(cells)))

print("\n## Held-out (8 run): chênh lệch so với RSIAT, CI 95%\n")
print("| Head | Δ AIA | Δ Final | Δ Forgetting (âm = quên ít hơn) | Δ BWT |")
print("|---|---|---|---|---|")
held = [r for r in runs if r["heldout"]]
for head in HEADS[1:]:
    cells = []
    for key in ("aia", "final", "forgetting", "bwt"):
        m, c = ci([r["heads"][head][key] - r["heads"]["linear"][key] for r in held])
        cells.append("%+.2f [%+.2f, %+.2f]" % (m, c[0], c[1]))
    print("| %s | %s |" % (LABEL[head], " | ".join(cells)))
