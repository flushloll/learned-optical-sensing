"""Bouncing-ball world + simulated sensors.

Everything is vectorised over a batch of independent sequences and generated
on the fly, so training never sees the same sequence twice.

Coordinates are in pixels, +y points down (gravity pulls toward larger y).
"""
import math

import torch
import torch.nn.functional as F

H = W = 64                 # sensor resolution -> 4096 pixels
DT = 0.01                  # one frame every 10 ms (100 fps)
SUBSTEPS = 10              # physics integrated at 1 ms for accurate bounces
GRAVITY = 400.0            # px / s^2
CONTEXT = 10               # frames the predictor sees (100 ms of history)
HORIZONS = (1, 5, 10)      # predict 10 ms, 50 ms, 100 ms past the last frame
SEQ_LEN = CONTEXT + max(HORIZONS)
NOISE_STD = 0.02           # per-pixel sensor noise
BG = 0.2                   # background brightness
TARGET_BRIGHTNESS = 1.0

CONDITIONS = ("clean", "fast", "flicker", "texture", "distractor")


def _uniform(gen, shape, lo, hi, device):
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=device)


def simulate(batch, steps, gen, device, speed=(20.0, 60.0)):
    """Return positions (B, T, 2) and radii (B,) of a ball bouncing in a box."""
    r = _uniform(gen, (batch,), 3.0, 5.0, device)
    lo, hi = r, W - r
    pos = torch.stack([_uniform(gen, (batch,), 0, 1, device) * (hi - lo) + lo,
                       _uniform(gen, (batch,), 0, 1, device) * (hi - lo) + lo], 1)
    s = _uniform(gen, (batch,), *speed, device)
    ang = _uniform(gen, (batch,), 0, 2 * math.pi, device)
    vel = torch.stack([s * torch.cos(ang), s * torch.sin(ang)], 1)
    e = _uniform(gen, (batch,), 0.75, 0.95, device)   # restitution
    h = DT / SUBSTEPS
    out = torch.empty(batch, steps, 2, device=device)
    for t in range(steps):
        out[:, t] = pos
        for _ in range(SUBSTEPS):
            vel = vel + torch.tensor([0.0, GRAVITY * h], device=device)
            pos = pos + vel * h
            px, py = pos[:, 0], pos[:, 1]
            vx, vy = vel[:, 0], vel[:, 1]
            cols = []
            for p, v in ((px, vx), (py, vy)):
                under, over = p < lo, p > hi
                p = torch.where(under, 2 * lo - p, p)
                v = torch.where(under, -e * v, v)
                p = torch.where(over, 2 * hi - p, p)
                v = torch.where(over, -e * v, v)
                cols.append((p, v))
            pos = torch.stack([cols[0][0], cols[1][0]], 1)
            vel = torch.stack([cols[0][1], cols[1][1]], 1)
    return out, r


def render(pos, r, brightness, device):
    """Anti-aliased disc for each (B, T) position -> (B, T, H, W) in [0, brightness]."""
    ys = torch.arange(H, device=device).float() + 0.5
    xs = torch.arange(W, device=device).float() + 0.5
    dx = xs.view(1, 1, 1, W) - pos[..., 0, None, None]
    dy = ys.view(1, 1, H, 1) - pos[..., 1, None, None]
    d = torch.sqrt(dx * dx + dy * dy)
    return brightness * (r.view(-1, 1, 1, 1) - d + 0.5).clamp(0, 1)


def make_batch(batch, condition, gen, device):
    """Frames (B, T, H, W) and target-ball positions (B, T, 2) for one condition."""
    speed = (60.0, 100.0) if condition == "fast" else (20.0, 60.0)
    pos, r = simulate(batch, SEQ_LEN, gen, device, speed)
    ball = render(pos, r, TARGET_BRIGHTNESS, device)

    bg = torch.full((batch, 1, H, W), BG, device=device)
    if condition == "texture":
        coarse = torch.rand(batch, 1, 8, 8, generator=gen, device=device) * 2 - 1
        bg = bg + 0.15 * F.interpolate(coarse, size=(H, W), mode="bilinear",
                                       align_corners=False)
    frames = torch.maximum(bg.expand(-1, SEQ_LEN, -1, -1), ball)

    if condition == "distractor":
        pos2, r2 = simulate(batch, SEQ_LEN, gen, device, speed)
        frames = torch.maximum(frames, render(pos2, r2 * 0.8, 0.55, device))

    if condition == "flicker":   # global illumination oscillating 5-15 Hz, +-30 %
        f = _uniform(gen, (batch, 1), 5.0, 15.0, device)
        ph = _uniform(gen, (batch, 1), 0, 2 * math.pi, device)
        t = torch.arange(SEQ_LEN, device=device).float().view(1, -1) * DT
        light = 1 + 0.3 * torch.sin(2 * math.pi * f * t + ph)
        frames = frames * light[..., None, None]

    frames = frames + NOISE_STD * torch.randn(frames.shape, generator=gen, device=device)
    return frames.clamp(min=0), pos


# ---------------------------------------------------------------- sensors

def frame_differences(frames):
    d = torch.zeros_like(frames)
    d[:, 1:] = frames[:, 1:] - frames[:, :-1]
    return d


def events(frames, threshold=0.3, eps=0.05):
    """Event-camera model: a pixel fires each time its log brightness moves by
    `threshold` from its last reference level. Returns (B, T, 2, H, W) counts of
    ON / OFF events between consecutive frames.

    Simplification: events are binned at the 100 fps frame rate, so this does
    not model the microsecond timing a real event camera provides.
    """
    logi = torch.log(frames + eps)
    ref = logi[:, 0].clone()
    out = torch.zeros(frames.shape[0], frames.shape[1], 2, H, W, device=frames.device)
    for t in range(1, frames.shape[1]):
        n = torch.trunc((logi[:, t] - ref) / threshold)
        out[:, t, 0] = n.clamp(min=0)
        out[:, t, 1] = (-n).clamp(min=0)
        ref = ref + n * threshold
    return out


def targets(pos):
    """Future positions at each horizon after the last context frame, scaled to [0, 1]."""
    idx = [CONTEXT - 1 + h for h in HORIZONS]
    return pos[:, idx] / W          # (B, len(HORIZONS), 2)
