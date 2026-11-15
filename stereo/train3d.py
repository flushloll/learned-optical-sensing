"""Train one setup on the 3D stereo task, evaluate it, and probe its hidden state.

    python train3d.py --setup stereo_frames --size random
    python train3d.py --setup bino_mask --readings 32 --size random
    python train3d.py --setup ssl_stereo --size random      # self-supervised

Setups: mono_frames, stereo_frames, mono_mask, sep_mask, bino_mask, ssl_mono, ssl_stereo
Supervised setups predict future 3D positions. ssl_* setups only predict their own
future frames (no positions, no depth labels); afterwards a linear probe checks
whether depth can be read out of what they learned. The probe is also run on an
untrained copy of the same network as a control.
"""
import argparse
import json
import math
import os
import time

import torch

import sim3d as S
from models3d import build

TEST_SEQS = 2000
CHUNK = 250
PROBE_FIT = 4000


def device_auto():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ssl_target(frames):
    idx = [S.CONTEXT - 1 + 5, S.CONTEXT - 1 + 10]
    t = frames[:, idx]                                  # (B, 2, E, H, W)
    return t.transpose(1, 2).flatten(1, 2)              # (B, E*2, H, W)


def loss_fn(model, frames, pos):
    out = model(frames[:, :S.CONTEXT])
    if model.ssl:
        tgt = ssl_target(frames)
        w = 1 + 20 * (tgt > S.BG + 0.15).float()        # the ball is a few pixels; up-weight it
        return (w * (out - tgt) ** 2).mean()
    return torch.nn.functional.mse_loss(out, S.targets(pos))


def batches(n, size, seed_base, dev, eyes, baseline):
    for c in range(n // CHUNK):
        gen = torch.Generator(device=dev).manual_seed(seed_base + c)
        yield S.make_batch(CHUNK, size, gen, dev, eyes, baseline)


@torch.no_grad()
def evaluate(model, dev, eyes, baseline):
    """Lateral (x, y) and depth (z) error in cm, per horizon, on both size regimes."""
    model.eval()
    res = {}
    for ri, size in enumerate(("fixed", "random")):
        lat, dep, zt = [], [], []
        for frames, pos, _ in batches(TEST_SEQS, size, 20_000 + 1000 * ri, dev, eyes, baseline):
            pred = S.denormalize(model(frames[:, :S.CONTEXT]))
            true = pos[:, [S.CONTEXT - 1 + h for h in S.HORIZONS]]
            d = (pred - true) * 100
            lat.append(d[..., :2].norm(dim=-1))
            dep.append(d[..., 2].abs())
            zt.append(true[..., 2])
        lat, dep, zt = torch.cat(lat), torch.cat(dep), torch.cat(zt)
        r = {}
        for i, h in enumerate(S.HORIZONS):
            r[f"lateral_{h * 10}ms"] = lat[:, i].mean().item()
            r[f"depth_{h * 10}ms"] = dep[:, i].mean().item()
        z = zt[:, -1]
        for lo, hi in ((1.0, 1.67), (1.67, 2.33), (2.33, 3.0)):
            m = (z >= lo) & (z < hi)
            r[f"depth_100ms_z{lo:.1f}-{hi:.1f}"] = dep[m, -1].mean().item()
        res[size] = r
    return res


@torch.no_grad()
def probe(model, dev, eyes, baseline, size):
    """Ridge regression from the GRU's final hidden state to depth and lateral position."""
    model.eval()

    def collect(n, seed):
        hs, ys = [], []
        for frames, pos, r in batches(n, size, seed, dev, eyes, baseline):
            hs.append(model.hidden(frames[:, :S.CONTEXT]))
            now, fut = pos[:, S.CONTEXT - 1], pos[:, -1]
            ys.append(torch.stack([now[:, 2], fut[:, 2], now[:, 0], r], 1))
        return torch.cat(hs).double().cpu(), torch.cat(ys).double().cpu()

    xf, yf = collect(PROBE_FIT, 30_000)
    xt, yt = collect(TEST_SEQS, 40_000)
    mu, sd = xf.mean(0), xf.std(0).clamp(min=1e-6)
    xf, xt = (xf - mu) / sd, (xt - mu) / sd
    xf = torch.cat([xf, torch.ones(len(xf), 1, dtype=xf.dtype)], 1)
    xt = torch.cat([xt, torch.ones(len(xt), 1, dtype=xt.dtype)], 1)
    w = torch.linalg.solve(xf.t() @ xf + 1.0 * torch.eye(xf.shape[1], dtype=xf.dtype), xf.t() @ yf)
    err = xt @ w - yt
    names = ["depth_now", "depth_+100ms", "x_now", "ball_radius"]
    out = {}
    for i, n in enumerate(names):
        rmse = err[:, i].pow(2).mean().sqrt().item() * 100
        chance = (yt[:, i] - yf[:, i].mean()).pow(2).mean().sqrt().item() * 100
        r2 = 1 - (rmse / chance) ** 2 if chance > 1e-9 else None   # constant target (fixed size)
        out[n] = {"rmse_cm": rmse, "chance_rmse_cm": chance, "r2": r2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--readings", type=int, default=None)
    ap.add_argument("--size", default="random", choices=["fixed", "random"])
    ap.add_argument("--baseline", type=float, default=S.BASELINE)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = device_auto()
    torch.manual_seed(a.seed)
    name = f"{a.setup}" + (f"_R{a.readings}" if a.readings else "") + \
        f"_{a.size}_b{a.baseline:g}_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)

    model, eyes = build(a.setup, a.readings)
    model.to(dev)
    torch.manual_seed(a.seed + 999)
    untrained, _ = build(a.setup, a.readings)
    untrained.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        frames, pos, _ = S.make_batch(a.batch, a.size, gen, dev, eyes, a.baseline)
        loss = loss_fn(model, frames, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            print(f"[{name}] step {step:5d} loss {loss.item():.5f} {time.time() - t0:6.0f}s", flush=True)

    out = {"name": name, "setup": a.setup, "readings": a.readings, "size": a.size,
           "baseline": a.baseline, "seed": a.seed, "steps": a.steps, "eyes": eyes,
           "readings_per_frame": a.readings or eyes * S.H * S.W,
           "train_seconds": time.time() - t0,
           "probe": probe(model, dev, eyes, a.baseline, a.size),
           "probe_untrained": probe(untrained, dev, eyes, a.baseline, a.size)}
    if not model.ssl:
        out["test"] = evaluate(model, dev, eyes, a.baseline)
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    if hasattr(model.encoder, "masks"):
        torch.save(model.encoder.masks().detach().cpu(), os.path.join(a.out, name + "_masks.pt"))
    print(json.dumps(out.get("test", out["probe"]))[:400], flush=True)


if __name__ == "__main__":
    main()
