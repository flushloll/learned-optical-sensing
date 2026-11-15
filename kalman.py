"""Classical baseline: find the ball in each frame, then run a Kalman filter.

Detection: brightness-weighted centroid of pixels well above the frame's median
(the background). Filter: constant-acceleration model in x and y. Prediction:
roll the model forward, then clamp to the box. It knows nothing about bounces,
so its errors near walls are expected; that is the honest version of
"plain classical tracking".

    python kalman.py      -> results/kalman.json
"""
import itertools
import json
import os

import torch

import sim
from train import TEST_CHUNK, TEST_SEQS


def centroids(frames):
    b, t = frames.shape[:2]
    f = frames.flatten(2)
    med = f.median(dim=2, keepdim=True).values
    mx = f.max(dim=2, keepdim=True).values
    w = (f - (med + 0.5 * (mx - med))).clamp(min=0)
    w = w / w.sum(2, keepdim=True).clamp(min=1e-8)
    ys = (torch.arange(sim.H).float() + 0.5).repeat_interleave(sim.W)
    xs = (torch.arange(sim.W).float() + 0.5).repeat(sim.H)
    return torch.stack([(w * xs).sum(2), (w * ys).sum(2)], -1)       # (B, T, 2)


def kalman_predict(z, q, r):
    """z: (B, T, 2) measurements. Returns predictions (B, len(HORIZONS), 2)."""
    dt = sim.DT
    f1 = torch.tensor([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    hm = torch.tensor([[1.0, 0, 0]])
    g = torch.tensor([[dt ** 3 / 6], [dt ** 2 / 2], [dt]])
    qm = q * (g @ g.t())
    b = z.shape[0]
    preds = []
    for d in range(2):                                   # x and y are independent
        x = torch.zeros(b, 3, 1)
        x[:, 0, 0] = z[:, 0, d]
        p = torch.diag(torch.tensor([r, 1e4, 1e6])).expand(b, 3, 3).clone()
        for t in range(sim.CONTEXT):
            if t > 0:
                x = f1 @ x
                p = f1 @ p @ f1.t() + qm
            s = hm @ p @ hm.t() + r
            k = p @ hm.t() / s
            x = x + k * (z[:, t, d].view(b, 1, 1) - hm @ x)
            p = (torch.eye(3) - k @ hm) @ p
        out = []
        for h in sim.HORIZONS:
            fh = torch.linalg.matrix_power(f1, h)
            out.append((fh @ x)[:, 0, 0])
        preds.append(torch.stack(out, 1))
    return torch.stack(preds, -1).clamp(3.0, sim.W - 3.0)


def errors(cond_index, cond, q, r, seqs=TEST_SEQS, seed_base=10_000):
    errs = []
    for chunk in range(seqs // TEST_CHUNK):
        gen = torch.Generator().manual_seed(seed_base + 100 * cond_index + chunk)
        frames, pos = sim.make_batch(TEST_CHUNK, cond, gen, "cpu")
        pred = kalman_predict(centroids(frames[:, :sim.CONTEXT]), q, r)
        errs.append((pred - sim.targets(pos) * sim.W).norm(dim=-1))
    return torch.cat(errs)


def main():
    # tune q, r on separate clean validation sequences (different seeds from test)
    best = None
    for q, r in itertools.product([1e6, 1e7, 1e8, 1e9, 1e10, 1e11], [0.01, 0.03, 0.1, 0.3, 1.0]):
        e = errors(0, "clean", q, r, seqs=500, seed_base=90_000).mean().item()
        if best is None or e < best[0]:
            best = (e, q, r)
    _, q, r = best
    res = {}
    for ci, cond in enumerate(sim.CONDITIONS):
        e = errors(ci, cond, q, r)
        res[cond] = {f"{h * 10}ms": e[:, i].mean().item() for i, h in enumerate(sim.HORIZONS)}
        res[cond]["mean"] = e.mean().item()
    out = {"name": "kalman", "method": "kalman", "q": q, "r": r,
           "measurements_per_frame": sim.H * sim.W, "test": res}
    os.makedirs("results", exist_ok=True)
    with open("results/kalman.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
