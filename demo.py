"""Make demo/demo.gif: two 16-mask models predicting a ball 100 ms ahead while the
lighting starts to flicker halfway through.

  D  = 16 learned masks
  D' = 16 learned masks + one total-light detector

Both are trained on clean scenes only (CPU is fine, a few minutes each):
    python demo.py            # trains, then renders demo/demo.gif
    python demo.py --steps 500  # quick, less accurate
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

import sim
from models import build
from train import sensor_input

OUT = "demo"
K = 16
T = 70                  # frames in the demo sequence (0.7 s)
FLICKER_FROM = 35       # flicker turns on halfway
H_IDX = 2               # horizon index -> 100 ms ahead
AHEAD = sim.HORIZONS[H_IDX]


def train(method, steps, seed=0):
    path = os.path.join(OUT, f"{method}_k{K}.pt")
    model = build(method, K)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.eval()
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    gen = torch.Generator().manual_seed(1000 + seed)
    for step in range(1, steps + 1):
        frames, pos = sim.make_batch(64, "clean", gen, "cpu")
        loss = torch.nn.functional.mse_loss(
            model(sensor_input(method, frames, pos)), sim.targets(pos))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"[{method}] step {step}/{steps}  rmse {math.sqrt(loss.item()) * sim.W:.2f}px",
                  flush=True)
    torch.save(model.state_dict(), path)
    return model.eval()


def demo_sequence(seed=7):
    gen = torch.Generator().manual_seed(seed)
    pos, r = sim.simulate(1, T, gen, "cpu", speed=(40.0, 60.0))
    frames = torch.maximum(torch.full((1, T, sim.H, sim.W), sim.BG),
                           sim.render(pos, r, sim.TARGET_BRIGHTNESS, "cpu"))
    t = torch.arange(T).float() * sim.DT
    light = torch.where(torch.arange(T) >= FLICKER_FROM,
                        1 + 0.3 * torch.sin(2 * math.pi * 10.0 * t), torch.ones(T))
    frames = frames * light.view(1, T, 1, 1)
    frames = frames + sim.NOISE_STD * torch.randn(frames.shape, generator=gen)
    return frames.clamp(min=0)[0], pos[0]


@torch.no_grad()
def rolling_predictions(model, method, frames):
    ts = list(range(sim.CONTEXT - 1, T - AHEAD))
    windows = torch.stack([frames[t - sim.CONTEXT + 1:t + 1] for t in ts])
    out = model(sensor_input(method, windows, None))[:, H_IDX] * sim.W
    return ts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    d = train("mask", a.steps)
    dn = train("mask_norm", a.steps)
    # Not cherry-picked: of 40 random sequences, show the one with the median steady-light error.
    def clean_err(seed):
        f, p = demo_sequence(seed)
        ts, pr = rolling_predictions(dn, "mask_norm", f)
        tru = torch.stack([p[t + AHEAD] for t in ts])
        return float((pr - tru).norm(dim=1)[:FLICKER_FROM - ts[0]].mean())
    errs = sorted((clean_err(sd), sd) for sd in range(40))
    seed = errs[len(errs) // 2][1]
    print(f"steady-light error over 40 sequences: median {errs[20][0]:.2f}px, using seed {seed}")
    frames, pos = demo_sequence(seed)
    ts, pd = rolling_predictions(d, "mask", frames)
    _, pn = rolling_predictions(dn, "mask_norm", frames)
    truth = torch.stack([pos[t + AHEAD] for t in ts])
    err_d = (pd - truth).norm(dim=1)
    err_n = (pn - truth).norm(dim=1)
    print(f"mean error before flicker: D {err_d[:FLICKER_FROM - ts[0]].mean():.2f}px, "
          f"D' {err_n[:FLICKER_FROM - ts[0]].mean():.2f}px")
    print(f"mean error during flicker: D {err_d[FLICKER_FROM - ts[0]:].mean():.2f}px, "
          f"D' {err_n[FLICKER_FROM - ts[0]:].mean():.2f}px")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.86, wspace=0.18)
    img = ax.imshow(frames[ts[0]], cmap="gray", vmin=0, vmax=1.3)
    ax.set_xticks([]); ax.set_yticks([])
    tr, = ax.plot([], [], "o", ms=14, mfc="none", mec="#3ddc84", mew=2, label="true (100 ms ahead)")
    m1, = ax.plot([], [], "x", ms=11, mew=2.5, color="#ff5a5a", label="D: 16 masks")
    m2, = ax.plot([], [], "+", ms=14, mew=2.5, color="#4aa3ff", label="D': 16 masks + light sensor")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), fontsize=7.5, ncol=1, frameon=False)
    title = ax.set_title("", fontsize=10)

    secs = [t * sim.DT for t in ts]
    ax2.axvspan(FLICKER_FROM * sim.DT, secs[-1], color="#ffd54f", alpha=0.25, lw=0)
    ax2.text((FLICKER_FROM + 2) * sim.DT, 0.95, "lighting flicker on", transform=ax2.get_xaxis_transform(),
             fontsize=8, va="top")
    l1, = ax2.plot([], [], color="#ff5a5a", lw=2, label="D")
    l2, = ax2.plot([], [], color="#4aa3ff", lw=2, label="D'")
    ax2.set_xlim(secs[0], secs[-1])
    ax2.set_ylim(0, max(float(err_d.max()), 5) * 1.1)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("prediction error (px, 64-px frame)")
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("Predicting a ball 100 ms ahead from 16 optical readings per frame (not 4,096 pixels)",
                 fontsize=10)

    def step(i):
        t = ts[i]
        img.set_data(frames[t])
        tr.set_data([truth[i, 0]], [truth[i, 1]])
        m1.set_data([pd[i, 0]], [pd[i, 1]])
        m2.set_data([pn[i, 0]], [pn[i, 1]])
        title.set_text("flicker ON" if t >= FLICKER_FROM else "steady light")
        l1.set_data(secs[:i + 1], err_d[:i + 1])
        l2.set_data(secs[:i + 1], err_n[:i + 1])
        return img, tr, m1, m2, title, l1, l2

    FuncAnimation(fig, step, frames=len(ts), blit=False).save(
        os.path.join(OUT, "demo.gif"), writer=PillowWriter(fps=12), dpi=80)
    print("wrote", os.path.join(OUT, "demo.gif"))


if __name__ == "__main__":
    main()
