"""Per-frame encoders (one per sensing method) feeding a shared GRU predictor."""
import torch
import torch.nn as nn

from sim import H, W, HORIZONS

FEAT = 64


class ConvEncoder(nn.Module):
    """Small CNN for full-image inputs (frames, differences, event counts)."""

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * (H // 8) * (W // 8), FEAT), nn.ReLU(),
        )

    def forward(self, x):            # (N, C, H, W)
        return self.net(x)


class MaskEncoder(nn.Module):
    """K optical masks, as a DMD + single-pixel detector would apply them.

    Each measurement is y_k = sum_j m_kj * x_j with mask values m in [0, 1]
    (a micromirror either sends a pixel's light to the detector or not; values in
    between correspond to time-multiplexed / grey-level patterns). Only these K
    numbers leave the "sensor" each frame. With learned=False the masks are
    fixed random patterns (classic compressive sensing).

    normalize=True adds one extra detector that sees all the light (an all-open
    mask) and divides every measurement by it. A global brightness change
    (flicker, clouds) scales numerator and denominator equally, so it cancels
    in the optics/readout instead of having to be learned. Costs K+1 readings.
    """

    def __init__(self, k, learned=True, normalize=False):
        super().__init__()
        self.normalize = normalize
        logits = torch.randn(k, H * W) * (0.5 if learned else 3.0)
        self.logits = nn.Parameter(logits, requires_grad=learned)
        self.norm = nn.BatchNorm1d(k)   # digital gain/offset after readout
        self.mlp = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, FEAT), nn.ReLU())

    def masks(self):
        return torch.sigmoid(self.logits)

    def measure(self, x):            # (N, H*W) -> (N, K)
        y = x @ self.masks().t() / (H * W) * 64
        if self.normalize:
            y = y / x.mean(1, keepdim=True).clamp(min=1e-3)
        return y

    def forward(self, x):
        return self.mlp(self.norm(self.measure(x.flatten(1))))


class StateEncoder(nn.Module):
    """Oracle: given the true (x, y) of the ball every frame. Upper bound."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, FEAT), nn.ReLU())

    def forward(self, x):
        return self.mlp(x)


class Predictor(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(FEAT, 128, batch_first=True)
        self.head = nn.Linear(128, 2 * len(HORIZONS))

    def forward(self, x):            # x: (B, T, ...) per-frame inputs
        b, t = x.shape[:2]
        f = self.encoder(x.flatten(0, 1)).view(b, t, -1)
        h, _ = self.gru(f)
        return self.head(h[:, -1]).view(b, len(HORIZONS), 2)


def build(method, k=None):
    if method == "frames":
        enc = ConvEncoder(1)
    elif method == "diff":
        enc = ConvEncoder(1)
    elif method == "events":
        enc = ConvEncoder(2)
    elif method == "mask":
        enc = MaskEncoder(k, learned=True)
    elif method == "mask_norm":
        enc = MaskEncoder(k, learned=True, normalize=True)
    elif method == "random_mask":
        enc = MaskEncoder(k, learned=False)
    elif method == "oracle":
        enc = StateEncoder()
    else:
        raise ValueError(method)
    return Predictor(enc)
