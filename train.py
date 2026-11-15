"""Train one sensing method and evaluate it on every test condition.

    python train.py --method mask --k 16
    python train.py --method frames
Methods: frames, diff, events, mask, random_mask, oracle
Writes results/<name>.json (and the learned masks for mask runs).
"""
import argparse
import json
import math
import os
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

import sim
from models import build

TEST_SEQS = 2000
TEST_CHUNK = 250


def device_auto():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sensor_input(method, frames, pos):
    """Convert raw frames into what each method's predictor receives (context only)."""
    c = sim.CONTEXT
    if method == "frames":
        return frames[:, :c, None]
    if method == "diff":
        return sim.frame_differences(frames[:, :c])[:, :, None]
    if method == "events":
        return sim.events(frames[:, :c])
    if method in ("mask", "mask_norm", "random_mask"):
        return frames[:, :c]
    if method == "oracle":
        return pos[:, :c] / sim.W
    raise ValueError(method)


def measurements_per_frame(method, k):
    if method == "mask_norm":
        return k + 1                      # K masks + one total-light detector
    return {"frames": sim.H * sim.W, "diff": sim.H * sim.W, "events": 2 * sim.H * sim.W,
            "mask": k, "random_mask": k, "oracle": 2}[method]


def encoder_flops(model, method, dev):
    """Digital FLOPs per frame spent in the encoder. For mask methods the K dot
    products are done optically in hardware, so they are excluded."""
    gen = torch.Generator(device=dev).manual_seed(0)
    frames, pos = sim.make_batch(2, "clean", gen, dev)
    x = sensor_input(method, frames, pos)[:1, :1].flatten(0, 1)
    enc = model.encoder
    enc.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        if method in ("mask", "mask_norm", "random_mask"):
            y = enc.measure(x.flatten(1))
            optical = fc.get_total_flops()
            enc.mlp(enc.norm(y))
            return fc.get_total_flops() - optical
        enc(x)
    return fc.get_total_flops()


@torch.no_grad()
def evaluate(model, method, dev):
    model.eval()
    res = {}
    for ci, cond in enumerate(sim.CONDITIONS):
        errs = []
        for chunk in range(TEST_SEQS // TEST_CHUNK):
            gen = torch.Generator(device=dev).manual_seed(10_000 + 100 * ci + chunk)
            frames, pos = sim.make_batch(TEST_CHUNK, cond, gen, dev)
            pred = model(sensor_input(method, frames, pos)) * sim.W
            errs.append((pred - sim.targets(pos) * sim.W).norm(dim=-1))   # px
        e = torch.cat(errs)                                               # (N, H)
        res[cond] = {f"{h * 10}ms": e[:, i].mean().item() for i, h in enumerate(sim.HORIZONS)}
        res[cond]["mean"] = e.mean().item()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--train-cond", default="clean", choices=["clean", "mixed"],
                    help="mixed = batches cycle through clean, flicker and texture")
    a = ap.parse_args()

    if a.threads:
        torch.set_num_threads(a.threads)
    dev = device_auto()
    torch.manual_seed(a.seed)
    name = a.method + (f"_k{a.k}" if a.k else "") + \
        ("_mixed" if a.train_cond == "mixed" else "") + f"_s{a.seed}"
    os.makedirs(a.out, exist_ok=True)

    model = build(a.method, a.k).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        model.train()
        cond = "clean"
        if a.train_cond == "mixed":
            cond = ("clean", "flicker", "texture")[step % 3]
        frames, pos = sim.make_batch(a.batch, cond, gen, dev)
        loss = torch.nn.functional.mse_loss(model(sensor_input(a.method, frames, pos)),
                                            sim.targets(pos))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            rmse_px = math.sqrt(loss.item()) * sim.W
            print(f"[{name}] step {step:5d}  train rmse {rmse_px:6.2f}px  "
                  f"{time.time() - t0:6.0f}s", flush=True)

    out = {
        "name": name, "method": a.method, "k": a.k, "seed": a.seed, "steps": a.steps,
        "train_cond": a.train_cond,
        "measurements_per_frame": measurements_per_frame(a.method, a.k),
        "encoder_flops_per_frame": encoder_flops(model, a.method, dev),
        "trainable_params": sum(p.numel() for p in params),
        "train_seconds": time.time() - t0,
        "test": evaluate(model, a.method, dev),
    }
    with open(os.path.join(a.out, name + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    if a.method in ("mask", "mask_norm", "random_mask"):
        torch.save(model.encoder.masks().detach().cpu(), os.path.join(a.out, name + "_masks.pt"))
    print(json.dumps(out["test"]["clean"]), flush=True)


if __name__ == "__main__":
    main()
