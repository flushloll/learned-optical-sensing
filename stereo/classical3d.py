"""Hand-built baselines, no learning.

mono:   find the ball in one image; depth from apparent size, assuming the
        ball has the typical radius (R_FIXED; with random sizes this is the
        geometric mean of the range, so it is the best single guess).
stereo: find the ball in both images; depth by triangulation from disparity,
        z = F * baseline / (u_left - u_right). Needs no size assumption.
Both then run a constant-acceleration Kalman filter per axis, tuned on
validation data, and roll it forward.

    python classical3d.py      -> results/classical_*.json
"""
import itertools
import json
import math
import os

import torch

import sim3d as S
from train3d import CHUNK, TEST_SEQS


def detect(img):
    """(N, H, W) -> centre u, v and radius rho (px) of the bright blob."""
    f = img.flatten(1)
    med = f.median(1, keepdim=True).values
    mx = f.max(1, keepdim=True).values
    w = (f - (med + 0.5 * (mx - med))).clamp(min=0)
    w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
    ys = (torch.arange(S.H).float() + 0.5).repeat_interleave(S.W)
    xs = (torch.arange(S.W).float() + 0.5).repeat(S.H)
    u, v = (w * xs).sum(1), (w * ys).sum(1)
    # total excess brightness of an anti-aliased disc ~ (1 - BG) * pi * rho^2
    excess = (f - med).clamp(min=0) * (f > med + 0.1)
    rho = (excess.sum(1) / ((1 - S.BG) * math.pi)).sqrt().clamp(min=0.3)
    return u, v, rho


def measure(frames, eyes, baseline):
    """frames (B, T, E, H, W) -> 3D measurements (B, T, 3) in metres."""
    b, t = frames.shape[:2]
    cams = S.camera_xs(eyes, baseline)
    det = [detect(frames[:, :, e].flatten(0, 1)) for e in range(eyes)]
    if eyes == 1:
        u, v, rho = det[0]
        z = S.F * S.R_FIXED / rho
        x = (u - S.CX) * z / S.F + cams[0]
    else:
        (ul, vl, _), (ur, vr, _) = det
        disp = (ul - ur).clamp(min=0.5)
        z = S.F * baseline / disp
        x = ((ul + ur) / 2 - S.CX) * z / S.F            # midpoint of the two cameras is x = 0
        v = (vl + vr) / 2
    y = (v - S.CY) * z / S.F
    m = torch.stack([x, y, z], -1).view(b, t, 3)
    lo, hi = torch.tensor(S.BOX_LO), torch.tensor(S.BOX_HI)
    return torch.maximum(torch.minimum(m, hi), lo)


def kalman(z, q, r_axes):
    """Constant-acceleration Kalman filter per axis. z: (B, T, 3)."""
    dt = S.DT
    f1 = torch.tensor([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    hm = torch.tensor([[1.0, 0, 0]])
    g = torch.tensor([[dt ** 3 / 6], [dt ** 2 / 2], [dt]])
    qm = q * (g @ g.t())
    b = z.shape[0]
    preds = []
    for d in range(3):
        r = r_axes[d]
        x = torch.zeros(b, 3, 1)
        x[:, 0, 0] = z[:, 0, d]
        p = torch.diag(torch.tensor([r, 100.0, 1e4])).expand(b, 3, 3).clone()
        for t in range(S.CONTEXT):
            if t > 0:
                x = f1 @ x
                p = f1 @ p @ f1.t() + qm
            k = p @ hm.t() / (hm @ p @ hm.t() + r)
            x = x + k * (z[:, t, d].view(b, 1, 1) - hm @ x)
            p = (torch.eye(3) - k @ hm) @ p
        preds.append(torch.stack([(torch.linalg.matrix_power(f1, h) @ x)[:, 0, 0]
                                  for h in S.HORIZONS], 1))
    out = torch.stack(preds, -1)
    lo, hi = torch.tensor(S.BOX_LO), torch.tensor(S.BOX_HI)
    return torch.maximum(torch.minimum(out, hi), lo)


def run(size, eyes, baseline, q, r_axes, n, seed_base):
    lat, dep, zt = [], [], []
    for c in range(n // CHUNK):
        gen = torch.Generator().manual_seed(seed_base + c)
        frames, pos, _ = S.make_batch(CHUNK, size, gen, "cpu", eyes, baseline)
        pred = kalman(measure(frames[:, :S.CONTEXT], eyes, baseline), q, r_axes)
        true = pos[:, [S.CONTEXT - 1 + h for h in S.HORIZONS]]
        d = (pred - true) * 100
        lat.append(d[..., :2].norm(dim=-1))
        dep.append(d[..., 2].abs())
        zt.append(true[..., 2])
    return torch.cat(lat), torch.cat(dep), torch.cat(zt)


def main():
    os.makedirs("results", exist_ok=True)
    for eyes in (1, 2):
        for size in ("fixed", "random"):
            best = None
            for q, rxy, rz in itertools.product([1e2, 1e3, 1e4, 1e5], [1e-6, 1e-5, 1e-4],
                                                [1e-4, 1e-3, 1e-2, 1e-1]):
                lat, dep, _ = run(size, eyes, S.BASELINE, q, (rxy, rxy, rz), 500, 90_000)
                score = (lat[:, -1].mean() + dep[:, -1].mean()).item()
                if best is None or score < best[0]:
                    best = (score, q, rxy, rz)
            _, q, rxy, rz = best
            ri = 0 if size == "fixed" else 1
            lat, dep, zt = run(size, eyes, S.BASELINE, q, (rxy, rxy, rz), TEST_SEQS, 20_000 + 1000 * ri)
            r = {}
            for i, h in enumerate(S.HORIZONS):
                r[f"lateral_{h * 10}ms"] = lat[:, i].mean().item()
                r[f"depth_{h * 10}ms"] = dep[:, i].mean().item()
            z = zt[:, -1]
            for lo, hi in ((1.0, 1.67), (1.67, 2.33), (2.33, 3.0)):
                m = (z >= lo) & (z < hi)
                r[f"depth_100ms_z{lo:.1f}-{hi:.1f}"] = dep[m, -1].mean().item()
            setup = "classical_mono" if eyes == 1 else "classical_stereo"
            name = f"{setup}_{size}_b{S.BASELINE:g}"
            out = {"name": name, "setup": setup, "size": size, "baseline": S.BASELINE,
                   "eyes": eyes, "readings_per_frame": eyes * S.H * S.W,
                   "kalman": {"q": q, "r_xy": rxy, "r_z": rz}, "test": {size: r}}
            with open(f"results/{name}.json", "w") as f:
                json.dump(out, f, indent=2)
            print(name, {k: round(v, 2) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
