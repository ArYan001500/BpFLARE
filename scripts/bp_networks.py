"""Network classes, the CWC image transform and the weighted Huber loss shared by the training and analysis
scripts (residual network, gated raw-signal and coherence-image fusion network)."""


from __future__ import annotations


import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.net(x) + self.skip(x))


class RawResNet1D(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (48, 96, 160, 256), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = [ConvBlock1D(2, channels[0], 9, 2)]
        for left, right in zip(channels, channels[1:]):
            layers += [ConvBlock1D(left, right, 7, 2), ConvBlock1D(right, right, 5)]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels[-1] * 2, 192), nn.SiLU(), nn.Dropout(dropout), nn.Linear(192, 3))
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_output_bias(self, values: np.ndarray) -> None:
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.as_tensor(values, dtype=self.head[-1].bias.dtype, device=self.head[-1].bias.device))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.encoder(raw)
        return self.head(torch.cat([x.mean(-1), x.amax(-1)], dim=1))


def weighted_huber_loss_custom(prediction: torch.Tensor, target: torch.Tensor, weights: tuple[float, float, float]) -> torch.Tensor:
    target_weights = prediction.new_tensor(weights).view(1, 3)
    per_target = F.huber_loss(prediction, target, reduction="none", delta=5.0)
    return (per_target * target_weights).mean() + 0.03 * F.relu(prediction[:, 1] + 5.0 - prediction[:, 0]).mean()


def _finite_zscore_1d(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        median = float(np.nanmedian(x[np.isfinite(x)])) if np.isfinite(x).any() else 0.0
        x = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / std).astype(np.float32)


def _resize_2d(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    if image.shape == size:
        return image.astype(np.float32)
    factors = (size[0] / image.shape[0], size[1] / image.shape[1])
    return zoom(image, factors, order=1).astype(np.float32)


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class MBConvLite(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expansion: int = 3, dropout: float = 0.1):
        super().__init__()
        hidden = in_ch * expansion
        self.use_skip = stride == 1 and in_ch == out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            SqueezeExcite2D(hidden),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_skip:
            y = self.dropout(y)
            return x + y
        return y


class EfficientNetLiteCWT(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            MBConvLite(32, 32, stride=1, expansion=2, dropout=dropout),
            MBConvLite(32, 64, stride=2, expansion=3, dropout=dropout),
            MBConvLite(64, 64, stride=1, expansion=3, dropout=dropout),
            MBConvLite(64, 96, stride=2, expansion=3, dropout=dropout),
            MBConvLite(96, 96, stride=1, expansion=3, dropout=dropout),
            MBConvLite(96, 128, stride=2, expansion=3, dropout=dropout),
            MBConvLite(128, 128, stride=1, expansion=3, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_output_bias(self, values: np.ndarray) -> None:
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.as_tensor(values, dtype=self.head[-1].bias.dtype, device=self.head[-1].bias.device))

    def forward(self, cwt: torch.Tensor) -> torch.Tensor:
        x = self.encoder(cwt)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def compute_cwc_phaseaware_image(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs: float,
    image_size: int,
    bandlimited: bool = False,
) -> np.ndarray:
    import pywt
    from scipy.ndimage import uniform_filter

    ecg_z = _finite_zscore_1d(ecg)
    ppg_z = _finite_zscore_1d(ppg)
    if bandlimited:
        low_freq, high_freq = 0.7, min(8.0, 0.45 * fs)
    else:
        low_freq, high_freq = 0.5, min(25.0, 0.45 * fs)
    target_freqs = np.geomspace(low_freq, high_freq, image_size)
    central = pywt.central_frequency("cmor1.5-1.0")
    scales = central * fs / target_freqs
    wx, _ = pywt.cwt(ecg_z, scales.astype(np.float32), "cmor1.5-1.0", sampling_period=1.0 / fs)
    wy, _ = pywt.cwt(ppg_z, scales.astype(np.float32), "cmor1.5-1.0", sampling_period=1.0 / fs)
    cross = wx * np.conj(wy)
    smooth_size = (3, 9)
    sxy_real = uniform_filter(cross.real.astype(np.float32), size=smooth_size, mode="nearest")
    sxy_imag = uniform_filter(cross.imag.astype(np.float32), size=smooth_size, mode="nearest")
    sxx = uniform_filter((np.abs(wx) ** 2).astype(np.float32), size=smooth_size, mode="nearest")
    syy = uniform_filter((np.abs(wy) ** 2).astype(np.float32), size=smooth_size, mode="nearest")
    mag = np.square(sxy_real) + np.square(sxy_imag)
    coherence = mag / np.maximum(sxx * syy, 1e-8)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    phase = np.arctan2(sxy_imag, sxy_real).astype(np.float32)
    coherence = _resize_2d(coherence, (image_size, image_size))
    sin_phase = _resize_2d(np.sin(phase).astype(np.float32), (image_size, image_size))
    cos_phase = _resize_2d(np.cos(phase).astype(np.float32), (image_size, image_size))
    return np.stack([coherence, sin_phase, cos_phase]).astype(np.float32)


class GatedRawCWCFusionNet(nn.Module):
    def __init__(self, dropout: float = 0.15):
        super().__init__()
        channels = (48, 96, 160, 256)
        raw_layers: list[nn.Module] = [ConvBlock1D(2, channels[0], 9, 2)]
        for left, right in zip(channels, channels[1:]):
            raw_layers += [ConvBlock1D(left, right, 7, 2), ConvBlock1D(right, right, 5)]
        self.raw_encoder = nn.Sequential(*raw_layers)
        self.raw_proj = nn.Sequential(nn.Linear(channels[-1] * 2, 128), nn.SiLU(), nn.Dropout(dropout))
        self.cwc_encoder = EfficientNetLiteCWT(dropout=dropout).encoder
        self.cwc_proj = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(256, 128), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        self._init_linear()

    def _init_linear(self) -> None:
        for module in (self.raw_proj, self.cwc_proj, self.gate, self.head):
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.kaiming_uniform_(sub.weight, a=math.sqrt(5))
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    def set_output_bias(self, values: np.ndarray) -> None:
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.as_tensor(values, dtype=self.head[-1].bias.dtype, device=self.head[-1].bias.device))

    def forward(self, raw: torch.Tensor, cwc: torch.Tensor) -> torch.Tensor:
        raw_encoded = self.raw_encoder(raw)
        raw_emb = self.raw_proj(torch.cat([raw_encoded.mean(-1), raw_encoded.amax(-1)], dim=1))
        cwc_emb = self.cwc_proj(self.cwc_encoder(cwc).mean(dim=(2, 3)))
        gate = self.gate(torch.cat([raw_emb, cwc_emb], dim=1))
        fused = gate * raw_emb + (1.0 - gate) * cwc_emb
        return self.head(fused)
