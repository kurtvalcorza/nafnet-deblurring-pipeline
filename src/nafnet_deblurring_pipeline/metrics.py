"""Full-reference restoration metrics, classical baselines and a no-reference sharpness proxy (numpy only).

PSNR follows the upstream NAFNet GoPro test configuration (``calculate_psnr``, RGB, crop_border 0, test_y_channel
False, peak 255). SSIM is the classical Wang et al. (2004) index: an 11-tap Gaussian window of sigma 1.5, computed per
RGB channel over the 'valid' region and averaged over the channels. The upstream repository's default
``calculate_ssim`` uses a 3-D Gaussian variant, so SSIM values here are **not** numerically comparable with the
upstream README table; this notebook never quotes that table as its own measurement.

Baselines (scored with the same functions on the same records):

* ``identity`` — the blurred input itself: the floor every method must beat;
* ``unsharp`` — blind unsharp masking; its two settings are chosen on the **training** split and then frozen;
* ``wiener`` — non-blind Wiener deconvolution given the **true** blur kernel (an oracle no real deblurrer has); its
  noise-to-signal ratio is chosen on the training split. It is unavailable when the kernel is unknown (BYOD).
"""
# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .samples import convolve

METRIC_DEFINITIONS = {
    "psnr": "peak signal-to-noise ratio in dB between the uint8 RGB output and the sharp reference (peak 255, all pixels, all channels); higher is better; +1 dB is about 21 % less mean squared error",
    "ssim": "structural similarity (Gaussian window 11, sigma 1.5, per RGB channel, valid region, channel mean); in -1..1, 1 means identical structure; complements PSNR because it weighs local structure rather than squared error",
}
UNSHARP_GRID = {"sigma": (1.0, 2.0, 3.0), "amount": (0.25, 0.5, 1.0, 2.0)}
WIENER_NSR_GRID = (3e-3, 1e-2, 3e-2, 1e-1)
_G = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
_G = _G / _G.sum()  # the 2-D window is the outer product of this 1-D Gaussian, so filtering is separable


def _pair(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred, ref = np.asarray(pred), np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: output {pred.shape} vs reference {ref.shape}")
    if pred.dtype != np.uint8 or ref.dtype != np.uint8:
        raise TypeError("metrics expect uint8 arrays (outputs are rounded to 8 bits before scoring)")
    return pred.astype(np.float64), ref.astype(np.float64)


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """PSNR in dB of two uint8 arrays (see METRIC_DEFINITIONS); +inf for identical arrays."""
    a, b = _pair(pred, ref)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _valid_filter(z: np.ndarray) -> np.ndarray:
    """Gaussian filtering over the 'valid' region (no padding): rows, then columns."""
    from numpy.lib.stride_tricks import sliding_window_view

    rows = sliding_window_view(z, 11, axis=0) @ _G
    return sliding_window_view(rows, 11, axis=1) @ _G


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean SSIM over the RGB channels of two uint8 H x W x 3 arrays (see METRIC_DEFINITIONS)."""
    a, b = _pair(pred, ref)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    if min(a.shape[:2]) < 11:
        raise ValueError("ssim needs images of at least 11 px on each side")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    values = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = _valid_filter(x), _valid_filter(y)
        sxx, syy, sxy = _valid_filter(x * x) - mx**2, _valid_filter(y * y) - my**2, _valid_filter(x * y) - mx * my
        values.append(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2)))
    return float(np.mean([v.mean() for v in values]))


def score(outputs: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    """Per-record and mean PSNR/SSIM of one uint8 output per record against ``record['sharp']``."""
    if len(outputs) != len(records) or not records:
        raise ValueError("outputs and records must be non-empty and the same length")
    rows = [{"id": r["id"], "source": r["source"], "psnr": psnr(o, r["sharp"]), "ssim": ssim(o, r["sharp"])} for o, r in zip(outputs, records, strict=True)]
    finite = [row["psnr"] for row in rows if np.isfinite(row["psnr"])]
    return {"method": method, "n": len(rows), "psnr": float(np.mean(finite)) if finite else float("inf"), "ssim": float(np.mean([row["ssim"] for row in rows])), "per_record": rows}


def paired_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Per-record PSNR/SSIM differences ``b - a`` on the same records: mean, min, max and how many records improved."""
    ra = {r["id"]: r for r in a["per_record"]}
    diffs = [(r["psnr"] - ra[r["id"]]["psnr"], r["ssim"] - ra[r["id"]]["ssim"]) for r in b["per_record"]]
    dp, ds = np.array([d[0] for d in diffs]), np.array([d[1] for d in diffs])
    return {
        "from": a["method"],
        "to": b["method"],
        "n": len(diffs),
        "psnr_gain_mean": float(dp.mean()),
        "psnr_gain_min": float(dp.min()),
        "psnr_gain_max": float(dp.max()),
        "ssim_gain_mean": float(ds.mean()),
        "records_improved_psnr": int((dp > 0).sum()),
    }


# --------------------------------------------------------------------------------------------------
# classical baselines
# --------------------------------------------------------------------------------------------------


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()


def unsharp_mask(image: np.ndarray, *, sigma: float, amount: float) -> np.ndarray:
    """``x + amount * (x - G_sigma * x)``, clipped and rounded to uint8. Blind: it does not know the blur."""
    x = np.asarray(image, dtype=np.float64)
    out = x + amount * (x - convolve(x, gaussian_kernel(sigma)))
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def wiener_deconvolve(image: np.ndarray, kernel: np.ndarray, *, nsr: float) -> np.ndarray:
    """Non-blind Wiener deconvolution per channel with reflect padding: ``X = conj(H) Y / (|H|^2 + nsr)``."""
    x = np.asarray(image, dtype=np.float64) / 255.0
    k = kernel.shape[0]
    pad = 2 * k
    padded = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = padded.shape[:2]
    shift = (k - 1) // 2
    otf = np.fft.rfft2(np.roll(np.pad(kernel, ((0, h - k), (0, w - k))), (-shift, -shift), axis=(0, 1)))
    gain = np.conj(otf) / (np.abs(otf) ** 2 + nsr)
    out = np.empty_like(padded)
    for c in range(padded.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(padded[..., c]) * gain, s=(h, w))
    out = out[pad:-pad, pad:-pad]
    return np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)


def identity_outputs(records: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    return [np.asarray(r["blurred"]) for r in records]


def _best(records: Sequence[Mapping[str, Any]], candidates: Sequence[dict[str, float]], run: Callable[[Mapping[str, Any], dict[str, float]], np.ndarray]) -> dict[str, Any]:
    trials = []
    for params in candidates:
        trials.append({**params, "psnr": float(np.mean([psnr(run(r, params), r["sharp"]) for r in records]))})
    best = max(trials, key=lambda t: t["psnr"])
    return {"chosen": {k: v for k, v in best.items() if k != "psnr"}, "train_psnr": best["psnr"], "trials": trials}


def tune_unsharp(train: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose sigma and amount by mean PSNR on the training split (never on test)."""
    grid = [{"sigma": s, "amount": a} for s in UNSHARP_GRID["sigma"] for a in UNSHARP_GRID["amount"]]
    return _best(train, grid, lambda r, p: unsharp_mask(r["blurred"], **p))


def tune_wiener(train: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Choose the noise-to-signal ratio on the training split; None when any training record lacks its kernel."""
    if any(r.get("kernel_array") is None for r in train):
        return None
    return _best(train, [{"nsr": n} for n in WIENER_NSR_GRID], lambda r, p: wiener_deconvolve(r["blurred"], r["kernel_array"], **p))


def classical_outputs(records: Sequence[Mapping[str, Any]], unsharp: Mapping[str, Any], wiener: Mapping[str, Any] | None) -> dict[str, list[np.ndarray]]:
    """Outputs of the identity, unsharp and (when kernels are known) Wiener baselines with frozen settings."""
    outputs = {"identity": identity_outputs(records), "unsharp": [unsharp_mask(r["blurred"], **unsharp["chosen"]) for r in records]}
    if wiener is not None and all(r.get("kernel_array") is not None for r in records):
        outputs["wiener_oracle"] = [wiener_deconvolve(r["blurred"], r["kernel_array"], **wiener["chosen"]) for r in records]
    return outputs


# --------------------------------------------------------------------------------------------------
# no-reference proxy for unpaired images
# --------------------------------------------------------------------------------------------------


def gradient_energy(image: np.ndarray) -> float:
    """Mean squared luma gradient (finite differences). A sharpness *proxy* only: noise and ringing raise it too, so it
    is reported as a ratio for unpaired images and never as a quality score."""
    x = np.asarray(image, dtype=np.float64)
    luma = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    gx, gy = np.diff(luma, axis=1), np.diff(luma, axis=0)
    return float((gx**2).mean() + (gy**2).mean())
