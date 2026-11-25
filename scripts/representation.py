"""Phase-aware cross-wavelet coherence with an added group-delay channel.

Channels 0 to 2 reproduce the manuscript's representation exactly: squared wavelet coherence and
the sine and cosine of the smoothed cross-wavelet phase. `verify_against_legacy` checks this
claim numerically rather than asserting it, and the cache builder runs that check before
committing to a long job.

Channel 3 is new. The manuscript stores phase, but phase is a poor unit for a physical delay: a
fixed 200 ms lag is a quarter cycle at 1.25 Hz and two full cycles at 10 Hz, so one physiological
quantity appears as a different number in every row of the image, and the network has to discover
that they mean the same thing. Group delay expresses the same information in seconds, where a
fixed lag is simply a constant across the map.

The naive conversion delta_t = phi / (2 pi f) does not work, because phase is wrapped into
[-pi, pi] and therefore only unambiguous while the delay stays below 1 / (2f): about 250 ms at
2 Hz and 50 ms at 10 Hz. Physiological pulse arrival times are 200 to 300 ms, so that formula
aliases across most of the band. Taking the slope of the *unwrapped* phase along frequency
avoids the problem:

    tau_g(f, t) = -(1 / 2 pi) * d phi_unwrapped / df

The derivative is evaluated on the true geometrically spaced frequency grid, not on row indices,
and it is computed at full precision before any resizing or quantisation, since differentiating
interpolated and 8-bit quantised phase would amplify exactly the noise this channel is meant to
remove.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS = 125.0


@dataclass(frozen=True)
class RepresentationConfig:
    """Frozen to match the manuscript, except for the added group-delay channel."""

    fs_hz: float = FS
    wavelet: str = "cmor1.5-1.0"
    frequency_min_hz: float = 0.5
    frequency_max_hz: float = 25.0
    frequency_count: int = 128
    image_size: int = 128
    smoothing_scale_bins: int = 3
    smoothing_time_samples: int = 9
    epsilon: float = 1e-8
    # Group delay is clipped to this range before scaling to 8 bits. One second is far outside
    # any physiological pulse arrival time, so the clip only removes estimator noise.
    group_delay_clip_s: float = 1.0


def _resize_2d(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize. Mirrors the manuscript byte for byte, including zoom's default edge
    mode, which differs from "nearest" at the borders and shifted the result when it was
    changed."""
    from scipy.ndimage import zoom

    if array.shape == shape:
        return array.astype(np.float32)
    factors = (shape[0] / array.shape[0], shape[1] / array.shape[1])
    return zoom(array, factors, order=1).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    """float32 throughout, and a median fill for non-finite values, matching the manuscript."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if not np.isfinite(x).all():
        median = float(np.nanmedian(x[np.isfinite(x)])) if np.isfinite(x).any() else 0.0
        x = np.nan_to_num(x, nan=median, posinf=median, neginf=median)
    std = float(np.std(x))
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - float(np.mean(x))) / std).astype(np.float32)


def frequencies(config: RepresentationConfig) -> np.ndarray:
    return np.geomspace(config.frequency_min_hz, config.frequency_max_hz, config.frequency_count)


def compute_image(
    ecg: np.ndarray,
    ppg: np.ndarray,
    config: RepresentationConfig = RepresentationConfig(),
    with_group_delay: bool = True,
) -> np.ndarray:
    """Return (4, image_size, image_size) float32: R^2, sin phi, cos phi, group delay.

    With `with_group_delay=False` returns the three manuscript channels only.
    """
    import pywt
    from scipy.ndimage import uniform_filter

    ecg_z, ppg_z = _zscore(ecg), _zscore(ppg)
    freqs = frequencies(config)
    central = pywt.central_frequency(config.wavelet)
    scales = central * config.fs_hz / freqs

    wx, _ = pywt.cwt(ecg_z, scales.astype(np.float32), config.wavelet,
                     sampling_period=1.0 / config.fs_hz)
    wy, _ = pywt.cwt(ppg_z, scales.astype(np.float32), config.wavelet,
                     sampling_period=1.0 / config.fs_hz)

    cross = wx * np.conj(wy)
    smooth = (config.smoothing_scale_bins, config.smoothing_time_samples)
    sxy_real = uniform_filter(cross.real.astype(np.float32), size=smooth, mode="nearest")
    sxy_imag = uniform_filter(cross.imag.astype(np.float32), size=smooth, mode="nearest")
    sxx = uniform_filter((np.abs(wx) ** 2).astype(np.float32), size=smooth, mode="nearest")
    syy = uniform_filter((np.abs(wy) ** 2).astype(np.float32), size=smooth, mode="nearest")

    mag = np.square(sxy_real) + np.square(sxy_imag)
    coherence = mag / np.maximum(sxx * syy, config.epsilon)
    coherence = np.clip(coherence, 0.0, 1.0).astype(np.float32)
    phase = np.arctan2(sxy_imag, sxy_real).astype(np.float32)

    channels = [
        _resize_2d(coherence, (config.image_size, config.image_size)),
        _resize_2d(np.sin(phase).astype(np.float32), (config.image_size, config.image_size)),
        _resize_2d(np.cos(phase).astype(np.float32), (config.image_size, config.image_size)),
    ]

    if with_group_delay:
        # Unwrap along frequency, then differentiate against the real frequency grid. Both steps
        # happen at full precision on the native (frequency, time) grid, before any resizing.
        unwrapped = np.unwrap(phase.astype(np.float64), axis=0)
        d_phi_df = np.gradient(unwrapped, freqs, axis=0)
        tau = -d_phi_df / (2.0 * np.pi)
        tau = np.clip(tau, -config.group_delay_clip_s, config.group_delay_clip_s)
        channels.append(_resize_2d(tau.astype(np.float32),
                                   (config.image_size, config.image_size)))

    return np.stack(channels).astype(np.float32)


def quantize(image: np.ndarray, config: RepresentationConfig = RepresentationConfig()) -> np.ndarray:
    """Pack to uint8. Channel 0 is in [0,1]; channels 1 and 2 in [-1,1]; channel 3 in +/- clip."""
    out = np.empty(image.shape, dtype=np.uint8)
    out[0] = np.clip(np.rint(image[0] * 255.0), 0, 255)
    out[1] = np.clip(np.rint((image[1] + 1.0) * 127.5), 0, 255)
    out[2] = np.clip(np.rint((image[2] + 1.0) * 127.5), 0, 255)
    if image.shape[0] > 3:
        scaled = image[3] / config.group_delay_clip_s
        out[3] = np.clip(np.rint((scaled + 1.0) * 127.5), 0, 255)
    return out


def dequantize(packed: np.ndarray,
               config: RepresentationConfig = RepresentationConfig()) -> np.ndarray:
    """Inverse of `quantize`, returning float32 in the original physical ranges."""
    packed = np.asarray(packed)
    out = np.empty(packed.shape, dtype=np.float32)
    out[..., 0, :, :] = packed[..., 0, :, :] / 255.0
    out[..., 1, :, :] = packed[..., 1, :, :] / 127.5 - 1.0
    out[..., 2, :, :] = packed[..., 2, :, :] / 127.5 - 1.0
    if packed.shape[-3] > 3:
        out[..., 3, :, :] = (packed[..., 3, :, :] / 127.5 - 1.0) * config.group_delay_clip_s
    return out


def verify_against_legacy(ecg: np.ndarray, ppg: np.ndarray, legacy_module,
                          tolerance: float = 1e-5) -> dict:
    """Confirm channels 0 to 2 reproduce the manuscript implementation on real data."""
    mine = compute_image(ecg, ppg, with_group_delay=False)
    theirs = legacy_module.compute_cwc_phaseaware_image(
        np.asarray(ecg, dtype=np.float64), np.asarray(ppg, dtype=np.float64),
        fs=FS, image_size=128, bandlimited=False)
    diffs = [float(np.max(np.abs(mine[c] - theirs[c]))) for c in range(3)]
    return {
        "max_abs_difference_per_channel": diffs,
        "tolerance": tolerance,
        "matches": bool(max(diffs) <= tolerance),
        "shape_mine": list(mine.shape),
        "shape_legacy": list(theirs.shape),
    }
