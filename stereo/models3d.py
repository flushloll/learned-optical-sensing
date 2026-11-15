"""Encoders for one or two eyes, a GRU predictor, and a frame-prediction decoder."""
import torch
import torch.nn as nn

from sim3d import H, W, HORIZONS

FEAT = 64
HID = 128


class ConvEncoder(nn.Module):
    """CNN over the eyes' images stacked as channels (early binocular fusion)."""

    def __init__(self, eyes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(eyes, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * (H // 8) * (W // 8), FEAT), nn.ReLU(),
        )

    def forward(self, x):                  # (N, eyes, H, W)
        return self.net(x)


class MaskEncoder(nn.Module):
    """R learned optical masks over one or two eyes' images.

    mode="mono": all R masks see the single (left) eye.
    mode="sep":  R/2 masks see only the left eye, R/2 only the right eye
                 (two separate single-pixel cameras).
    mode="bino": every mask sees both eyes' images at once and sums their light
                 on one detector (optically: a beam combiner in front of the DMD).
    Total readings per frame are R in every mode, so comparisons are fair.
    """

    def __init__(self, readings, mode):
        super().__init__()
        eyes = 1 if mode == "mono" else 2
        self.logits = nn.Parameter(torch.randn(readings, eyes * H * W) * 0.5)
        support = torch.ones(readings, eyes * H * W)
        if mode == "sep":
            half = readings // 2
            support[:half, H * W:] = 0      # first half: left eye only
            support[half:, :H * W] = 0      # second half: right eye only
        self.register_buffer("support", support)
        self.norm = nn.BatchNorm1d(readings)
        self.mlp = nn.Sequential(nn.Linear(readings, 128), nn.ReLU(), nn.Linear(128, FEAT), nn.ReLU())

    def masks(self):
        return torch.sigmoid(self.logits) * self.support

    def measure(self, x):
        return x.flatten(1) @ self.masks().t() / (H * W) * 64

    def forward(self, x):
        return self.mlp(self.norm(self.measure(x)))


class Predictor(nn.Module):
    """Encoder -> GRU -> future 3D positions (supervised) or future frames (self-supervised)."""

    def __init__(self, encoder, ssl_eyes=None):
        super().__init__()
        self.encoder = encoder
        self.gru = nn.GRU(FEAT, HID, batch_first=True)
        self.ssl = ssl_eyes is not None
        if self.ssl:
            out = ssl_eyes * 2              # each eye's frame at +50 ms and +100 ms
            self.head = nn.Sequential(
                nn.Linear(HID, 32 * 8 * 8), nn.ReLU(), nn.Unflatten(1, (32, 8, 8)),
                nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, out, 4, 2, 1))
        else:
            self.head = nn.Linear(HID, 3 * len(HORIZONS))

    def hidden(self, x):
        b, t = x.shape[:2]
        f = self.encoder(x.flatten(0, 1)).view(b, t, -1)
        h, _ = self.gru(f)
        return h[:, -1]

    def forward(self, x):
        out = self.head(self.hidden(x))
        return out if self.ssl else out.view(-1, len(HORIZONS), 3)


def build(setup, readings=None):
    """Returns (model, eyes)."""
    if setup == "mono_frames":
        return Predictor(ConvEncoder(1)), 1
    if setup == "stereo_frames":
        return Predictor(ConvEncoder(2)), 2
    if setup in ("mono_mask", "sep_mask", "bino_mask"):
        mode = setup.split("_")[0]
        return Predictor(MaskEncoder(readings, mode)), 1 if mode == "mono" else 2
    if setup == "ssl_mono":
        return Predictor(ConvEncoder(1), ssl_eyes=1), 1
    if setup == "ssl_stereo":
        return Predictor(ConvEncoder(2), ssl_eyes=2), 2
    raise ValueError(setup)
