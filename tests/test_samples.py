"""Degradation model, sample construction, validation, split and BYOD reading (numpy + Pillow only)."""
# ruff: noqa: E501

from __future__ import annotations

import io
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import needs_skimage, synthetic_image, synthetic_pairs
from nafnet_deblurring_pipeline import samples as S


def test_motion_kernel_is_normalised_and_oriented():
    k = S.motion_kernel(15, 0.0)
    assert k.shape == (15, 15) and abs(k.sum() - 1) < 1e-12 and (k >= 0).all()
    assert np.allclose(k.sum(axis=0)[k.sum(axis=0) > 0].std(), 0, atol=0.02)  # horizontal line: mass spread along x
    assert k[7].sum() > 0.95  # concentrated on the centre row
    vertical = S.motion_kernel(15, 90.0)
    assert vertical[:, 7].sum() > 0.95
    with pytest.raises(ValueError, match="1..63"):
        S.motion_kernel(0.5, 0)


def test_convolve_with_delta_is_identity_and_degrade_is_seeded():
    img = synthetic_image(side=64)
    delta = np.zeros((3, 3))
    delta[1, 1] = 1
    assert np.abs(S.convolve(img.astype(float), delta) - img).max() < 1e-9
    k = S.motion_kernel(9, 45)
    a = S.degrade(img, k, 0.01, np.random.default_rng(1))
    b = S.degrade(img, k, 0.01, np.random.default_rng(1))
    assert a.dtype == np.uint8 and a.shape == img.shape and np.array_equal(a, b)
    assert not np.array_equal(a, img)


def test_validate_records_accepts_pairs_and_reports_ceilings():
    report = S.validate_records(synthetic_pairs())
    assert report["records"] == 8 and report["sources"] == 4 and report["ceilings"]["max_side"] == S.MAX_SIDE


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r[0].update(sharp=r[0]["sharp"][:80]), "identical size"),
        (lambda r: r[0].update(blurred=r[0]["blurred"][:40, :40], sharp=r[0]["sharp"][:40, :40]), "64..1024"),
        (lambda r: r[1].update(id=r[0]["id"]), "duplicate record id"),
        (lambda r: r[0].update(blurred=r[0]["blurred"].astype(np.float32)), "uint8"),
        (lambda r: r[0].update(blurred=r[0]["blurred"][..., 0]), "H x W x 3"),
        (lambda r: r[0].pop("source"), "missing required fields"),
        (lambda r: r[0].update(sharp=None), "paired mode needs a sharp reference"),
        (lambda r: [x.update(source="one") for x in r], "at least 2 distinct sources"),
    ],
)
def test_validate_records_refusals_name_the_rule(mutate, message):
    records = synthetic_pairs()
    mutate(records)
    with pytest.raises(ValueError, match=message):
        S.validate_records(records)


def test_too_few_pairs_and_unpaired_minimum():
    with pytest.raises(ValueError, match="4..200 are required"):
        S.validate_records(synthetic_pairs(2, 1))
    unpaired = [{"id": "a", "source": "a", "blurred": synthetic_image(), "sharp": None}]
    assert S.validate_records(unpaired, paired=False)["records"] == 1


def test_split_by_source_keeps_sources_together_and_manifest_detects_leakage():
    records = synthetic_pairs(4, 3)
    splits = S.split_by_source(records, seed=0)
    train_sources = {r["source"] for r in splits["train"]}
    test_sources = {r["source"] for r in splits["test"]}
    assert train_sources and test_sources and not train_sources & test_sources
    manifest = S.dataset_manifest(splits)
    assert manifest["disjoint_sources"] and len(manifest["digest"]) == 64
    assert S.dataset_manifest(S.split_by_source(records, seed=0))["digest"] == manifest["digest"]
    leaky = {"train": splits["train"], "test": splits["test"] + [splits["train"][0]]}
    with pytest.raises(ValueError, match="more than one split"):
        S.dataset_manifest(leaky)


@needs_skimage
def test_sample_dataset_is_pinned_deterministic_and_split_by_photograph():
    a, b = S.build_sample_dataset(), S.build_sample_dataset()
    assert len(a["train"]) == 48 and len(a["test"]) == 12
    assert S.dataset_manifest(a)["digest"] == S.dataset_manifest(b)["digest"]
    assert {r["source"] for r in a["train"]}.isdisjoint({r["source"] for r in a["test"]})
    assert all(r["blurred"].shape == (256, 256, 3) for r in a["train"] + a["test"])
    lengths = [r["kernel"]["length_px"] for r in a["train"]]
    assert min(lengths) >= 9 and max(lengths) <= 21
    new = S.build_new_data()
    assert [r["id"] for r in new] == ["text-synthetic", "clock-real-motion"] and new[1]["sharp"] is None


def test_sample_photograph_refused_on_digest_mismatch(tmp_path: Path):
    entry = S.SAMPLE_IMAGES["chelsea"]
    (tmp_path / entry["file"]).write_bytes(b"x" * entry["bytes"])
    with pytest.raises(ValueError, match="pinned"):
        S.load_sample_image("chelsea", tmp_path)


# ---- BYOD --------------------------------------------------------------------------------------------------------


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _zip(path: Path, members: dict[str, bytes], *, symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "target")
    return path


def _paired_members(n: int = 4, side: int = 96) -> dict[str, bytes]:
    members = {}
    for r in synthetic_pairs(n, 1, side=side):
        members[f"sharp/{r['id']}.png"] = _png(r["sharp"])
        members[f"blurred/{r['id']}.png"] = _png(r["blurred"])
    return members


def test_read_byod_paired_zip_and_directory(tmp_path: Path):
    result = S.read_byod(_zip(tmp_path / "pairs.zip", _paired_members()))
    assert result["mode"] == "paired" and len(result["records"]) == 4 and len(result["archive_digest"]) == 64
    folder = tmp_path / "dir"
    for name, data in _paired_members().items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    assert S.read_byod(folder)["mode"] == "paired"


def test_read_byod_wrapping_folder_greyscale_and_unpaired(tmp_path: Path):
    members = {f"mydata/{k}": v for k, v in _paired_members().items()}
    members["mydata/sharp/src0-0.png"] = _png(synthetic_image(seed=0)[..., 0])  # greyscale is converted and reported
    members["mydata/blurred/src0-0.png"] = _png(synthetic_image(seed=0)[..., 0])
    result = S.read_byod(_zip(tmp_path / "wrapped.zip", members))
    assert result["mode"] == "paired" and any("greyscale" in c for c in result["conversions"])
    unpaired = S.read_byod(_zip(tmp_path / "blurred_only.zip", {"a.png": _png(synthetic_image()), "b.jpg": _png(synthetic_image(seed=2))}))
    assert unpaired["mode"] == "unpaired" and all(r["sharp"] is None for r in unpaired["records"])


@pytest.mark.parametrize(
    ("members", "symlink", "message"),
    [
        ({"blurred/a.png": b"", "sharp/b.png": b""}, None, "without a partner"),
        ({"../evil.png": b"x"}, None, "absolute or traversing"),
        ({"/abs.png": b"x"}, None, "absolute or traversing"),
        ({}, "link.png", "symbolic link"),
        ({"readme.txt": b"hello"}, None, "no images found"),
        ({"other/a.png": b"x", "sharp/a.png": b"x"}, None, "not in sharp/ or blurred/"),
        ({"blurred/a.png": b"not an image", "sharp/a.png": b"not an image"}, None, "not a readable image"),
    ],
)
def test_read_byod_refusals(tmp_path: Path, members, symlink, message):
    if message == "without a partner":
        members = {"blurred/a.png": _png(synthetic_image()), "sharp/b.png": _png(synthetic_image())}
    with pytest.raises(ValueError, match=message):
        S.read_byod(_zip(tmp_path / "bad.zip", members, symlink=symlink))


def test_read_byod_size_mismatch_and_oversized(tmp_path: Path):
    members = _paired_members()
    members["blurred/src0-0.png"] = _png(synthetic_image(side=80))
    with pytest.raises(ValueError, match="identical size"):
        S.read_byod(_zip(tmp_path / "mismatch.zip", members))
    big = np.zeros((1100, 70, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="64..1024"):
        S.read_byod(_zip(tmp_path / "big.zip", {"blurred/x.png": _png(big)}))
    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        (tmp_path / "plain.txt").write_text("x")
        S.read_byod(tmp_path / "plain.txt")
