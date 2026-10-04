"""PSNR/SSIM semantics, classical baselines and their train-only tuning."""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from conftest import synthetic_image, synthetic_pairs
from nafnet_deblurring_pipeline import metrics as M


def test_psnr_matches_definition():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    b = a.copy()
    b[...] = 10
    assert M.psnr(a, b) == pytest.approx(20 * np.log10(255 / 10))
    assert M.psnr(a, a) == float("inf")
    with pytest.raises(TypeError):
        M.psnr(a.astype(float), b)
    with pytest.raises(ValueError, match="shape mismatch"):
        M.psnr(a, b[:8])


def test_ssim_matches_brute_force_window_and_is_one_for_identical():
    a = synthetic_image(side=48)
    b = np.clip(a.astype(int) + np.random.default_rng(0).integers(-25, 25, a.shape), 0, 255).astype(np.uint8)
    g = np.exp(-((np.arange(11) - 5) ** 2) / (2 * 1.5**2))
    w = np.outer(g, g) / np.outer(g, g).sum()

    def f(z):
        return np.einsum("ijkl,kl->ij", sliding_window_view(z, (11, 11)), w)

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for c in range(3):
        x, y = a[..., c].astype(float), b[..., c].astype(float)
        mx, my = f(x), f(y)
        sxy = f(x * y) - mx * my
        vals.append((((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (f(x * x) - mx**2 + f(y * y) - my**2 + c2))).mean())
    assert M.ssim(a, b) == pytest.approx(float(np.mean(vals)), abs=1e-12)
    assert M.ssim(a, a) == pytest.approx(1.0)


def test_wiener_with_true_kernel_beats_blur_on_low_noise():
    records = synthetic_pairs(2, 1, side=96)
    r = records[0]
    restored = M.wiener_deconvolve(r["blurred"], r["kernel_array"], nsr=1e-2)
    assert restored.shape == r["sharp"].shape and restored.dtype == np.uint8
    assert M.psnr(restored, r["sharp"]) > M.psnr(r["blurred"], r["sharp"])


def test_tuning_uses_train_only_and_baselines_cover_known_kernels():
    train, test = synthetic_pairs(3, 1, side=64), synthetic_pairs(2, 1, side=64, seed=10)
    unsharp, wiener = M.tune_unsharp(train), M.tune_wiener(train)
    assert unsharp["chosen"]["sigma"] in M.UNSHARP_GRID["sigma"] and len(unsharp["trials"]) == 12
    assert wiener["chosen"]["nsr"] in M.WIENER_NSR_GRID
    outputs = M.classical_outputs(test, unsharp, wiener)
    assert set(outputs) == {"identity", "unsharp", "wiener_oracle"}
    no_kernel = [dict(r, kernel_array=None) for r in train]
    assert M.tune_wiener(no_kernel) is None
    assert "wiener_oracle" not in M.classical_outputs(test, unsharp, None)


def test_score_and_paired_difference():
    records = synthetic_pairs(2, 1, side=64)
    blurred = M.score([r["blurred"] for r in records], records, method="identity")
    perfect = M.score([r["sharp"] for r in records], records, method="oracle")
    assert blurred["n"] == 2 and perfect["ssim"] == pytest.approx(1.0)
    noisy = [np.clip(r["sharp"].astype(int) + 3, 0, 255).astype(np.uint8) for r in records]
    diff = M.paired_difference(blurred, M.score(noisy, records, method="noisy"))
    assert diff["records_improved_psnr"] == 2 and diff["psnr_gain_mean"] > 0


def test_gradient_energy_is_higher_for_sharper_image():
    sharp = synthetic_image(side=64)
    from nafnet_deblurring_pipeline.samples import degrade, motion_kernel

    blurred = degrade(sharp, motion_kernel(11, 0), 0.0, np.random.default_rng(0))
    assert M.gradient_energy(sharp) > M.gradient_energy(blurred)
