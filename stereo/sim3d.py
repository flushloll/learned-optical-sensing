"""3D bouncing ball seen by one or two pinhole "eyes".

World units are metres. Cameras sit at z = 0 looking along +z with parallel
optical axes, separated horizontally by `baseline`. +y points down, so gravity
is +y. The ball lives in a box x, y in [-0.5, 0.5], depth z in [1, 3].

Depth cues available:
  * size: projected radius = F * r / z  (only useful if r is known)
  * disparity (two eyes): u_left - u_right = F * baseline / z  (needs no size)
  * gravity: apparent vertical acceleration = F * g / z  (a weak monocular cue)
"""
import math

import torch

H = W = 64
F = 48.0                    # focal length in pixels
CX = CY = W / 2
DT = 0.01                   # 100 fps
SUBSTEPS = 10
G = 9.8
CONTEXT = 10
HORIZONS = (1, 5, 10)       # 10, 50, 100 ms ahead
SEQ_LEN = CONTEXT + max(HORIZONS)
NOISE_STD = 0.02
BG = 0.2
BOX_LO = (-0.5, -0.5, 1.0)
BOX_HI = (0.5, 0.5, 3.0)
CENTER = (0.0, 0.0, 2.0)
HALF = (0.5, 0.5, 1.0)
R_FIXED = 0.08              # ball radius (m) when size is fixed
R_RANGE = (0.04, 0.16)      # 4x range when random: wider than the 3x depth range,
                            # so image size alone cannot give depth
BASELINE = 0.10             # eye separation (m)


def _u(gen, shape, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=dev)


def simulate(batch, steps, gen, dev, size="fixed"):
    if size == "fixed":
        r = torch.full((batch,), R_FIXED, device=dev)
    else:
        r = _u(gen, (batch,), *R_RANGE, dev)
    lo = torch.tensor(BOX_LO, device=dev) + r[:, None]
    hi = torch.tensor(BOX_HI, device=dev) - r[:, None]
    pos = lo + torch.rand(batch, 3, generator=gen, device=dev) * (hi - lo)
    d = torch.randn(batch, 3, generator=gen, device=dev)
    vel = d / d.norm(dim=1, keepdim=True) * _u(gen, (batch, 1), 0.5, 2.0, dev)
    e = _u(gen, (batch, 1), 0.75, 0.95, dev)
    g = torch.tensor([0.0, G, 0.0], device=dev)
    h = DT / SUBSTEPS
    out = torch.empty(batch, steps, 3, device=dev)
    for t in range(steps):
        out[:, t] = pos
        for _ in range(SUBSTEPS):
            vel = vel + g * h
            pos = pos + vel * h
            under, over = pos < lo, pos > hi
            pos = torch.where(under, 2 * lo - pos, pos)
            pos = torch.where(over, 2 * hi - pos, pos)
            vel = torch.where(under | over, -e * vel, vel)
    return out, r


def camera_xs(eyes, baseline):
    """Left eye at -b/2 (the single eye in mono is the left one), right at +b/2."""
    return [-baseline / 2] if eyes == 1 else [-baseline / 2, baseline / 2]


def project(pos, r, cam_x):
    z = pos[..., 2]
    u = F * (pos[..., 0] - cam_x) / z + CX
    v = F * pos[..., 1] / z + CY
    rho = F * r.view(-1, *([1] * (pos.dim() - 2))) / z
    return u, v, rho


def render_eye(pos, r, cam_x, dev):
    u, v, rho = project(pos, r, cam_x)
    ys = torch.arange(H, device=dev).float() + 0.5
    xs = torch.arange(W, device=dev).float() + 0.5
    dx = xs.view(1, 1, 1, W) - u[..., None, None]
    dy = ys.view(1, 1, H, 1) - v[..., None, None]
    d = torch.sqrt(dx * dx + dy * dy)
    return (rho[..., None, None] - d + 0.5).clamp(0, 1)


def make_batch(batch, size, gen, dev, eyes=2, baseline=BASELINE):
    """Frames (B, T, eyes, H, W), positions (B, T, 3), radii (B,)."""
    pos, r = simulate(batch, SEQ_LEN, gen, dev, size)
    views = []
    for cx in camera_xs(eyes, baseline):
        ball = render_eye(pos, r, cx, dev)
        f = torch.maximum(torch.full_like(ball, BG), ball)
        f = f + NOISE_STD * torch.randn(f.shape, generator=gen, device=dev)
        views.append(f.clamp(min=0))
    return torch.stack(views, 2), pos, r


def normalize(pos):
    c = torch.tensor(CENTER, device=pos.device)
    s = torch.tensor(HALF, device=pos.device)
    return (pos - c) / s


def denormalize(p):
    c = torch.tensor(CENTER, device=p.device)
    s = torch.tensor(HALF, device=p.device)
    return p * s + c


def targets(pos):
    idx = [CONTEXT - 1 + h for h in HORIZONS]
    return normalize(pos[:, idx])            # (B, len(HORIZONS), 3)
