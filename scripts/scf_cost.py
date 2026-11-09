"""Inference cost of the SCF score sources (linear head, ridge, PGM), for the ICISN paper.

Uses the same functions as the experiments (utils/qsd.py) on random class statistics of the real
sizes (d = 768 features, D = 769 encoded dimensions). The cost of the ViT forward pass is the same
for every head and is not included. Reports, per number of classes C:
  - setup time after a task (ridge solve; PGM states + whitening), once per task;
  - scoring time per test image (batch of 1000);
  - memory of what each head keeps (weights or states).

  python scripts/scf_cost.py [--classes 100 200] [--device cuda] [--output paper_icisn/results/scf_cost.md]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.qsd import Encoder, _inv_sqrt, dense_scores, ridge_scores, stats_states  # noqa: E402

D_FEAT, BATCH, REPEAT = 768, 1000, 5


def timed(fn, device):
    """Median wall time (s) of REPEAT calls, after one warm-up."""
    fn()
    times = []
    for _ in range(REPEAT):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    return sorted(times)[len(times) // 2]


def measure(classes, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    means = torch.randn(classes, D_FEAT, generator=g).to(device)
    a = torch.randn(classes, D_FEAT, 64, generator=g).to(device) / 8
    covs = a @ a.transpose(1, 2) + 1e-3 * torch.eye(D_FEAT, device=device)
    z = torch.randn(BATCH, D_FEAT, generator=g).to(device)
    fc = torch.randn(classes, D_FEAT, generator=g).to(device)
    encoder = Encoder.from_base(z, 32, torch.Generator(device=device).manual_seed(0))
    psi = encoder.linear(z)

    rhos = stats_states(encoder, means, covs)
    whitening, _ = _inv_sqrt(rhos.mean(0), 0.1 / 1024)
    second = covs.double().sum(0) + means.double().T @ means.double()
    weight = torch.linalg.solve(second + torch.eye(D_FEAT, dtype=torch.float64, device=device), means.double().T)

    rows = {
        "Linear head": (0.0, timed(lambda: z @ fc.T, device), fc.numel() * 4),
        "Ridge (SCF classical)": (
            timed(lambda: torch.linalg.solve(second + torch.eye(D_FEAT, dtype=torch.float64, device=device),
                                             means.double().T), device),
            timed(lambda: z.double() @ weight, device), weight.numel() * 8),
        "PGM (SCF quantum)": (
            timed(lambda: _inv_sqrt(stats_states(encoder, means, covs).mean(0), 0.1 / 1024), device),
            timed(lambda: dense_scores(psi, rhos, whitening), device),
            rhos.numel() * 4 + whitening.numel() * 8),
    }
    assert ridge_scores(means, covs, z, 0.1).shape == (BATCH, classes)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--classes", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    lines = ["# SCF inference cost ({}, torch {})".format(name, torch.__version__), "",
             "Không tính ViT forward (giống nhau cho mọi head). Batch {} ảnh, trung vị của {} lần.".format(BATCH, REPEAT),
             "", "| C | Head | Setup mỗi task (ms) | Chấm điểm (ms/ảnh) | Bộ nhớ (MB) |", "|---:|---|---:|---:|---:|"]
    for classes in args.classes:
        for head, (setup, score, nbytes) in measure(classes, device).items():
            lines.append("| {} | {} | {:.1f} | {:.4f} | {:.1f} |".format(classes, head, setup * 1e3,
                                                                       score * 1e3 / BATCH, nbytes / 2 ** 20))
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
