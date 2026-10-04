# ruff: noqa: E501
import builtins
import importlib.util

import numpy as np
import pytest

MODEL_LIBRARIES = {"torch", "safetensors", "torchvision", "huggingface_hub"}
HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("safetensors") is not None
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch + safetensors not installed")
needs_skimage = pytest.mark.skipif(not HAS_SKIMAGE, reason="scikit-image (the sample-photograph carrier) not installed")
TINY_CONFIG = {"img_channel": 3, "width": 8, "middle_blk_num": 1, "enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]}
TINY_TLC = (1, 3, 64, 64)


@pytest.fixture
def forbid_model_imports(monkeypatch):
    """Rejected requests must stop before importing model libraries (fleet RTM-001)."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.partition(".")[0] in MODEL_LIBRARIES:
            raise AssertionError(f"model dependency imported before rejection: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def synthetic_image(*, side: int = 96, seed: int = 0) -> np.ndarray:
    """A seeded uint8 RGB image with edges and texture (no model semantics)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side].astype(np.float32)
    base = np.stack([np.sin(x / (5 + seed)), np.cos(y / 7), np.sign(np.sin((x + y) / 9))], axis=-1)
    noise = rng.normal(0, 0.05, base.shape)
    return ((base + noise + 1.5) / 3.0 * 255).clip(0, 255).astype(np.uint8)


def synthetic_pairs(n_sources: int = 4, per_source: int = 2, *, side: int = 96, seed: int = 0):
    """Paired records with a known motion kernel, `per_source` crops per synthetic source image."""
    from nafnet_deblurring_pipeline.samples import degrade, motion_kernel

    rng = np.random.default_rng(seed)
    records = []
    for s in range(n_sources):
        sharp_full = synthetic_image(side=side, seed=seed + s)
        for i in range(per_source):
            kernel = motion_kernel(7 + 2 * i, 30.0 * s)
            blurred = degrade(sharp_full, kernel, 0.01, rng)
            records.append({"id": f"src{s}-{i}", "source": f"src{s}", "blurred": blurred, "sharp": sharp_full.copy(), "kernel": {"length_px": 7 + 2 * i, "angle_deg": 30.0 * s}, "kernel_array": kernel, "noise_sigma": 0.01})
    return records
