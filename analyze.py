"""Aggregate results/*.json into plots and a markdown table.

Outputs: error_vs_k.png, robustness.png, learned_masks.png, results.md
"""
import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sim

BASE = {"frames": "A  full frames + CNN", "diff": "B  frame differences + CNN",
        "events": "C  events + CNN", "kalman": "E  Kalman filter (classical)",
        "oracle": "oracle: true position", "mask": "D  learned masks",
        "mask_norm": "D' learned masks + total-light detector", "random_mask": "random masks"}
HORIZON = "100ms"


def key(r):
    m = r["method"] + ("+mixed" if r.get("train_cond") == "mixed" else "")
    return m, r.get("k")


def label(m, k):
    mixed = m.endswith("+mixed")
    s = BASE[m.replace("+mixed", "")]
    if k is not None:
        s += f" K={k}"
    return s + ("  [trained on mixed]" if mixed else "")


def load():
    groups = defaultdict(list)
    for f in glob.glob("results/*.json"):
        r = json.load(open(f))
        groups[key(r)].append(r)
    return groups


def stat(rs, cond, h):
    v = np.array([r["test"][cond][h] for r in rs])
    return v.mean(), v.std(), len(v)


def error_vs_k(groups):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"frames": "C0", "diff": "C2", "events": "C3", "kalman": "k", "oracle": "C7"}
    for m in ["frames", "diff", "events", "kalman", "oracle"]:
        if (m, None) in groups:
            mu = stat(groups[(m, None)], "clean", HORIZON)[0]
            ax.axhline(mu, ls="--", color=colors[m], lw=1.2, label=f"{label(m, None)}: {mu:.2f}px")
    for m, c in [("mask", "C1"), ("random_mask", "C4"), ("mask_norm", "C5")]:
        ks = sorted(k for (mm, k) in groups if mm == m)
        if not ks:
            continue
        mus, sds = zip(*[stat(groups[(m, k)], "clean", HORIZON)[:2] for k in ks])
        ax.errorbar(ks, mus, yerr=sds, marker="o", color=c, capsize=3, label=BASE[m])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("measurements per frame (K)   [full frame = 4096]")
    ax.set_ylabel(f"position error {HORIZON} ahead (px, log)")
    ax.set_title("Clean test set: error vs. number of optical measurements")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("error_vs_k.png", dpi=130)


def robustness(groups):
    sel = [("kalman", None), ("frames", None), ("frames+mixed", None), ("events", None),
           ("diff", None), ("mask", 16), ("mask_norm", 16), ("mask+mixed", 16),
           ("mask_norm+mixed", 16)]
    sel = [s for s in sel if s in groups]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    wdt = 0.85 / len(sel)
    x = np.arange(len(sim.CONDITIONS))
    for i, g in enumerate(sel):
        mus = [stat(groups[g], c, HORIZON)[0] for c in sim.CONDITIONS]
        ax.bar(x + i * wdt, mus, wdt, label=label(*g))
    ax.set_xticks(x + 0.425 - wdt / 2)
    ax.set_xticklabels(sim.CONDITIONS)
    ax.set_yscale("log")
    ax.set_ylabel(f"position error {HORIZON} ahead (px, log)")
    ax.set_title("Robustness. 'fast' and 'distractor' are never seen in training by any model")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig("robustness.png", dpi=130)


def masks_figure():
    try:
        m = torch.load("results/mask_k16_s0_masks.pt")
    except FileNotFoundError:
        return
    fig, axs = plt.subplots(2, 8, figsize=(12, 3.4))
    for i, ax in enumerate(axs.flat):
        ax.imshow(m[i].view(sim.H, sim.W), cmap="gray", vmin=0, vmax=1)
        ax.set_axis_off()
    fig.suptitle("The 16 learned mask patterns (white = light sent to detector)")
    fig.tight_layout()
    fig.savefig("learned_masks.png", dpi=110)


def table(groups):
    order = [(m, None) for m in ["frames", "frames+mixed", "diff", "events", "kalman", "oracle"]]
    for m in ["mask", "mask_norm", "mask+mixed", "mask_norm+mixed", "random_mask"]:
        order += sorted((g for g in groups if g[0] == m), key=lambda g: g[1])
    cols = [("clean", "10ms"), ("clean", "50ms"), ("clean", "100ms")] + \
           [(c, HORIZON) for c in sim.CONDITIONS[1:]]
    lines = ["| method | meas./frame | digital FLOPs/frame | seeds | "
             + " | ".join(f"{c} {h}" for c, h in cols) + " |",
             "|---|---:|---:|---:|" + "---:|" * len(cols)]
    for g in order:
        if g not in groups:
            continue
        rs = groups[g]
        fl = rs[0].get("encoder_flops_per_frame")
        cells = []
        for c, h in cols:
            mu, sd, n = stat(rs, c, h)
            cells.append(f"{mu:.2f}" + (f" ± {sd:.2f}" if n > 1 else ""))
        lines.append(f"| {label(*g)} | {rs[0]['measurements_per_frame']} | "
                     f"{'-' if fl is None else f'{fl:,}'} | {len(rs)} | " + " | ".join(cells) + " |")
    with open("results.md", "w") as f:
        f.write("Errors: mean Euclidean distance in pixels on a 64×64 sensor, 2000 test "
                "sequences per condition, ± = std over seeds. Mask methods: the K dot "
                "products are assumed done optically, so their FLOPs are not counted. "
                "Kalman: digital cost is a centroid over 4096 pixels plus a tiny filter.\n\n")
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    g = load()
    error_vs_k(g)
    robustness(g)
    masks_figure()
    table(g)
