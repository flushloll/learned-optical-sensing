"""Aggregate stereo results into plots and results3d.md.

Each model is reported on the size regime it was trained on.
"""
import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sim3d as S

LABEL = {"classical_mono": "classical, 1 eye (depth from size)",
         "classical_stereo": "classical, 2 eyes (triangulation)",
         "mono_frames": "CNN, 1 eye", "stereo_frames": "CNN, 2 eyes",
         "mono_mask": "masks, 1 eye", "sep_mask": "masks, 2 separate eyes",
         "bino_mask": "masks, binocular (each sees both eyes)",
         "ssl_mono": "self-supervised CNN, 1 eye", "ssl_stereo": "self-supervised CNN, 2 eyes"}
BINS = ["depth_100ms_z1.0-1.7", "depth_100ms_z1.7-2.3", "depth_100ms_z2.3-3.0"]


def load():
    g = defaultdict(list)
    for f in glob.glob("results/*.json"):
        r = json.load(open(f))
        g[(r["setup"], r.get("readings"), r["size"], r["baseline"])].append(r)
    return g


def val(rs, metric):
    v = np.array([r["test"][r["size"]][metric] for r in rs])
    return v.mean(), v.std(), len(v)


def name(setup, readings):
    return LABEL[setup] + (f", R={readings}" if readings else "")


def fmt(rs, m):
    mu, sd, n = val(rs, m)
    return f"{mu:.1f}" + (f" ± {sd:.1f}" if n > 1 else "")


def main_rows(g):
    rows = [("classical_mono", None), ("classical_stereo", None),
            ("mono_frames", None), ("stereo_frames", None)]
    for r in (32, 128):
        rows += [("mono_mask", r), ("sep_mask", r), ("bino_mask", r)]
    return rows


def table(g, lines):
    b = S.BASELINE
    lines += ["## Supervised: error 100 ms ahead (cm), eyes 10 cm apart", "",
              "| setup | readings/frame | fixed size: lateral | fixed: depth | random size: lateral | random: depth | random: depth near / mid / far |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for s, r in main_rows(g):
        f, rnd = g.get((s, r, "fixed", b)), g.get((s, r, "random", b))
        if not (f and rnd):
            continue
        bins = " / ".join(f"{val(rnd, m)[0]:.1f}" for m in BINS)
        lines.append(f"| {name(s, r)} | {rnd[0]['readings_per_frame']} | {fmt(f, 'lateral_100ms')} | "
                     f"{fmt(f, 'depth_100ms')} | {fmt(rnd, 'lateral_100ms')} | {fmt(rnd, 'depth_100ms')} | {bins} |")
    lines += ["", "Near / mid / far = true depth 1-1.67 / 1.67-2.33 / 2.33-3 m. "
              "Lateral = error in x, y combined. ± = std over 2 seeds.", ""]


def baseline_table(g, lines):
    lines += ["## Eye separation (CNN, 2 eyes, random size, error 100 ms ahead, cm)", "",
              "| eye separation | lateral | depth | depth near / mid / far |", "|---:|---:|---:|---|"]
    mono = g.get(("mono_frames", None, "random", S.BASELINE))
    if mono:
        lines.append(f"| one eye | {fmt(mono, 'lateral_100ms')} | {fmt(mono, 'depth_100ms')} | "
                     + " / ".join(f"{val(mono, m)[0]:.1f}" for m in BINS) + " |")
    for b in (0.03, 0.1, 0.3):
        rs = g.get(("stereo_frames", None, "random", b))
        if rs:
            lines.append(f"| {b * 100:.0f} cm | {fmt(rs, 'lateral_100ms')} | {fmt(rs, 'depth_100ms')} | "
                         + " / ".join(f"{val(rs, m)[0]:.1f}" for m in BINS) + " |")
    lines.append("")


def probe_table(g, lines):
    lines += ["## Does depth emerge without being taught? (linear probe on the hidden state)", "",
              "R² of a linear readout from the model's final hidden state (1 = perfect, 0 = no better "
              "than guessing the average). Self-supervised models were trained only to predict their "
              "own future frames; they never saw a depth label. 'untrained' = same network with random "
              "weights, as a control.", "",
              "| model | depth now | depth +100 ms | ball radius | x now | depth now, RMSE cm |",
              "|---|---:|---:|---:|---:|---:|"]
    keys = [("ssl_mono", None), ("ssl_stereo", None), ("mono_frames", None), ("stereo_frames", None)]
    for s, r in keys:
        rs = g.get((s, r, "random", S.BASELINE))
        if not rs:
            continue
        for tag, field in (("", "probe"), (" (untrained)", "probe_untrained")):
            p = rs[0][field]
            lines.append(f"| {name(s, r)}{tag} | {p['depth_now']['r2']:.2f} | {p['depth_+100ms']['r2']:.2f} | "
                         f"{p['ball_radius']['r2']:.2f} | {p['x_now']['r2']:.2f} | {p['depth_now']['rmse_cm']:.1f} |")
    lines += ["", "Supervised CNN rows are trained on 3D positions, so their depth R² is expected to be high; "
              "they are included as a ceiling.", ""]


def plot_main(g):
    b = S.BASELINE
    rows = [x for x in main_rows(g) if (x[0], x[1], "random", b) in g and (x[0], x[1], "fixed", b) in g]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rows))
    for i, (size, c) in enumerate((("fixed", "C0"), ("random", "C3"))):
        mus = [val(g[(s, r, size, b)], "depth_100ms")[0] for s, r in rows]
        sds = [val(g[(s, r, size, b)], "depth_100ms")[1] for s, r in rows]
        ax.bar(x + (i - 0.5) * 0.4, mus, 0.4, yerr=sds, color=c, capsize=2,
               label=f"ball size {size}")
    ax.set_xticks(x)
    ax.set_xticklabels([name(s, r) for s, r in rows], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("depth error 100 ms ahead (cm)")
    ax.set_title("Does the second eye matter? Depth error, fixed vs. random ball size (eyes 10 cm apart)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("depth_error.png", dpi=130)


def plot_bins(g):
    b = S.BASELINE
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    centers = [1.33, 2.0, 2.67]
    for s, r, st in [("classical_mono", None, "k:"), ("classical_stereo", None, "k--"),
                     ("mono_frames", None, "C0-o"), ("stereo_frames", None, "C3-o"),
                     ("bino_mask", 128, "C1-s")]:
        rs = g.get((s, r, "random", b))
        if rs:
            ax.plot(centers, [val(rs, m)[0] for m in BINS], st, label=name(s, r))
    ax.set_xlabel("true depth of ball (m)")
    ax.set_ylabel("depth error 100 ms ahead (cm)")
    ax.set_title("Random ball size: depth error vs. distance")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("depth_vs_distance.png", dpi=130)


def plot_masks():
    path = "results/bino_mask_R32_random_b0.1_s0_masks.pt"
    try:
        m = torch.load(path)
    except FileNotFoundError:
        return
    n = 8
    fig, axs = plt.subplots(3, n, figsize=(13, 5))
    for i in range(n):
        left = m[i, :S.H * S.W].view(S.H, S.W)
        right = m[i, S.H * S.W:].view(S.H, S.W)
        axs[0, i].imshow(left, cmap="gray", vmin=0, vmax=1)
        axs[1, i].imshow(right, cmap="gray", vmin=0, vmax=1)
        d = left - right
        lim = d.abs().max().item()
        axs[2, i].imshow(d, cmap="bwr", vmin=-lim, vmax=lim)
        for a in axs[:, i]:
            a.set_axis_off()
    axs[0, 0].set_title("left-eye part", fontsize=8, loc="left")
    axs[1, 0].set_title("right-eye part", fontsize=8, loc="left")
    axs[2, 0].set_title("left minus right", fontsize=8, loc="left")
    fig.suptitle("Learned binocular masks (R=32, random size): each detector sums light from both eyes")
    fig.tight_layout()
    fig.savefig("binocular_masks.png", dpi=100)


if __name__ == "__main__":
    g = load()
    lines = ["# Stereo experiment results", ""]
    table(g, lines)
    baseline_table(g, lines)
    probe_table(g, lines)
    open("results3d.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_main(g)
    plot_bins(g)
    plot_masks()
