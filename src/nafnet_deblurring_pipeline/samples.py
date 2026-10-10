"""Sample data, the synthetic degradation model, validation, splitting and BYOD loading.

numpy + Pillow only: nothing in this module imports a model library, so every refusal happens before torch loads.

Default sample
    Eleven public-domain / CC0 photographs that ship inside the hash-locked ``scikit-image`` wheel (``skimage/data``),
    each pinned here by byte size and SHA-256 and read as files (``skimage`` itself is never imported). Sharp crops are
    degraded with seeded linear motion blur plus Gaussian noise and 8-bit quantisation. The split is drawn **by source
    photograph**, so no photograph contributes crops to both training and test (SPL5).

Degradation model (the tutorial's "domain")
    ``blurred = quantise(clip(sharp ⊛ k_{L,θ} + n))`` with ``k_{L,θ}`` an anti-aliased line of length ``L`` px at angle
    ``θ``, ``n ~ N(0, σ²)`` per pixel and channel, and reflect padding at the image border. The convolution is applied to
    the whole photograph before cropping, so the crops contain no border artefact.

BYOD
    A zip or directory with ``sharp/`` and ``blurred/`` folders whose files are matched by stem (paired mode: the full
    validate → split → baselines → adapt → evaluate → export → reload → infer sequence), or with blurred images only
    (unpaired mode: validate → infer, no metric because there is no reference).
"""
# ruff: noqa: E501  -- refusal messages name the record and the rule in full; they are kept on one line

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import stat
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

SAMPLE_PACKAGE = "scikit-image"
SAMPLE_PACKAGE_VERSION = "0.26.0"
SAMPLE_SEED = 2022
CROP = 256
TRAIN_CROPS_PER_IMAGE = 8
TEST_CROPS_PER_IMAGE = 4
BLUR_LENGTH_PX = (9, 21)
BLUR_ANGLE_DEG = (0.0, 180.0)
NOISE_SIGMA = 0.01  # in [0, 1] units, i.e. about 2.55 grey levels
# Every photograph: file inside skimage/data, exact bytes, SHA-256 of those bytes, licence as stated in the
# scikit-image 0.26.0 data docstrings, and its role. Greyscale photographs are replicated to three channels.
SAMPLE_IMAGES: dict[str, dict[str, Any]] = {
    "chelsea": {"file": "chelsea.png", "bytes": 240512, "sha256": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb", "license": "CC0 (photographer Stefan van der Walt)", "role": "train"},
    "coffee": {"file": "coffee.png", "bytes": 466706, "sha256": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7", "license": "CC0 (photographer Rachel Michetti)", "role": "train"},
    "immunohistochemistry": {"file": "ihc.png", "bytes": 477916, "sha256": "f8dd1aa387ddd1f49d8ad13b50921b237df8e9b262606d258770687b0ef93cef", "license": "no known copyright restrictions (CMMI)", "role": "train"},
    "brick": {"file": "brick.png", "bytes": 106634, "sha256": "7966caf324f6ba843118d98f7a07746d22f6a343430add0233eca5f6eaaa8fcf", "license": "CC0 (CC0Textures)", "role": "train"},
    "grass": {"file": "grass.png", "bytes": 217893, "sha256": "b6b6022426b38936c43a4ac09635cd78af074e90f42ffa8227ac8b7452d39f89", "license": "CC0 (linolafett)", "role": "train"},
    "camera": {"file": "camera.png", "bytes": 139512, "sha256": "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a", "license": "CC0 (photographer Lav Varshney)", "role": "train"},
    "astronaut": {"file": "astronaut.png", "bytes": 791555, "sha256": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5", "license": "public domain (NASA)", "role": "test"},
    "rocket": {"file": "rocket.jpg", "bytes": 112525, "sha256": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c", "license": "public domain (SpaceX)", "role": "test"},
    "gravel": {"file": "gravel.png", "bytes": 194247, "sha256": "c48615b451bf1e606fbd72c0aa9f8cc0f068ab7111ef7d93bb9b0f2586440c12", "license": "CC0 (CC0Textures)", "role": "test"},
    "text": {"file": "text.png", "bytes": 42704, "sha256": "bd84aa3a6e3c9887850d45d606c96b2e59433fbef50338570b63c319e668e6d1", "license": "public domain (Wikipedia)", "role": "new-paired"},
    "clock_motion": {"file": "clock_motion.png", "bytes": 58784, "sha256": "f029226b28b642e80113d86622e9b215ee067a0966feaf5e60604a1e05733955", "license": "public domain (photographer Stefan van der Walt)", "role": "new-unpaired"},
}
# The new paired image is degraded with one fixed kernel so that every run scores the same input.
NEW_DATA_KERNEL = {"length_px": 15, "angle_deg": 30.0}
NEW_DATA_SEED = 7

# Operational ceilings (VAL6), shared by the sample and BYOD paths.
MIN_SIDE = 64
MAX_SIDE = 1024
MAX_IMAGES = 200
MIN_PAIRS = 4
MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_MEMBER_BYTES = 64 * 1024**2
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
INPUT_SCHEMA = {
    "record": "{id, source, blurred, sharp} with blurred/sharp uint8 RGB arrays of identical shape H x W x 3 (sharp is None for unpaired inference)",
    "sides": f"{MIN_SIDE}..{MAX_SIDE} px on each side; larger images are refused, never resized silently",
    "count": f"paired: {MIN_PAIRS}..{MAX_IMAGES} pairs from at least 2 distinct sources; unpaired: 1..{MAX_IMAGES} images",
    "colour": "RGB; greyscale is replicated to 3 channels and an alpha channel is dropped (both reported)",
    "byod_layout": "a zip or directory with sharp/ and blurred/ folders, files matched by stem (paired), or blurred images only (unpaired)",
    "validation": "structural only: nothing checks that a pair shows the same scene or that the blur is motion blur",
}


# --------------------------------------------------------------------------------------------------
# sample photographs
# --------------------------------------------------------------------------------------------------


def skimage_data_dir() -> Path:
    """``skimage/data`` of the installed scikit-image wheel, located without importing scikit-image."""
    spec = importlib.util.find_spec("skimage")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"the sample photographs ship inside {SAMPLE_PACKAGE}=={SAMPLE_PACKAGE_VERSION}, which is not installed in this environment")
    return Path(next(iter(spec.submodule_search_locations))) / "data"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_rgb(image: Image.Image) -> tuple[np.ndarray, str | None]:
    """A uint8 RGB array and a note describing any colour conversion (None when the image was already RGB)."""
    mode = image.mode
    if mode == "RGB":
        return np.asarray(image, dtype=np.uint8).copy(), None
    if mode in ("L", "I;16", "I", "F", "1", "P", "LA", "RGBA", "CMYK", "YCbCr"):
        note = {"L": "greyscale replicated to RGB", "LA": "greyscale replicated to RGB, alpha dropped", "RGBA": "alpha channel dropped", "P": "palette expanded to RGB", "1": "bilevel expanded to RGB"}.get(mode, f"{mode} converted to RGB")
        if mode in ("I;16", "I", "F"):
            raise ValueError(f"{mode} images (16-bit or float) are not supported; save the image as 8-bit RGB")
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy(), note
    raise ValueError(f"image mode {mode} is not supported; save the image as 8-bit RGB")


def load_sample_image(name: str, data_dir: Path | None = None) -> np.ndarray:
    """One pinned sample photograph as a uint8 RGB array; refused on any byte-size or SHA-256 mismatch."""
    return load_sample_image_with_note(name, data_dir)[0]


def sample_conversions(data_dir: Path | None = None) -> dict[str, str | None]:
    """``{photograph: colour conversion}`` for every pinned sample photograph (None when the file is already RGB): the
    report the data contract promises for the sample, as BYOD reports it for the reader's images."""
    return {name: load_sample_image_with_note(name, data_dir)[1] for name in SAMPLE_IMAGES}


def load_sample_image_with_note(name: str, data_dir: Path | None = None) -> tuple[np.ndarray, str | None]:
    """``load_sample_image`` plus the note naming any colour conversion (for example greyscale replicated to RGB)."""
    entry = SAMPLE_IMAGES[name]
    path = (data_dir or skimage_data_dir()) / entry["file"]
    if not path.is_file():
        raise FileNotFoundError(f"sample photograph {entry['file']} is missing from {path.parent} ({SAMPLE_PACKAGE}=={SAMPLE_PACKAGE_VERSION} expected)")
    data = path.read_bytes()
    if len(data) != entry["bytes"] or sha256_bytes(data) != entry["sha256"]:
        raise ValueError(f"sample photograph {entry['file']}: {len(data)} bytes with sha256 {sha256_bytes(data)[:16]}… != pinned {entry['bytes']} bytes / {entry['sha256'][:16]}…")
    with Image.open(io.BytesIO(data)) as image:
        return to_rgb(image)


# --------------------------------------------------------------------------------------------------
# degradation model
# --------------------------------------------------------------------------------------------------


def motion_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """A normalised, anti-aliased linear motion-blur kernel (odd square size >= length)."""
    if not 1 <= length_px <= 63:
        raise ValueError(f"motion length must be within 1..63 px, got {length_px}")
    size = int(np.ceil(length_px)) | 1
    kernel = np.zeros((size, size), dtype=np.float64)
    centre = (size - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    samples = max(64, int(length_px * 16))
    for t in np.linspace(-(length_px - 1) / 2.0, (length_px - 1) / 2.0, samples):
        x, y = centre + t * np.cos(theta), centre - t * np.sin(theta)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < size and 0 <= y0 + dy < size:
                kernel[y0 + dy, x0 + dx] += w
    return kernel / kernel.sum()


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution of every channel of a float image with reflect padding (FFT), same size as the input."""
    image = np.asarray(image, dtype=np.float64)
    k = kernel.shape[0]
    pad = k
    planes = image[..., None] if image.ndim == 2 else image
    padded = np.pad(planes, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = padded.shape[:2]
    otf = np.fft.rfft2(kernel, s=(h, w))
    out = np.empty_like(padded)
    for c in range(padded.shape[2]):
        out[..., c] = np.fft.irfft2(np.fft.rfft2(padded[..., c]) * otf, s=(h, w))
    shift = (k - 1) // 2
    out = np.roll(out, (-shift, -shift), axis=(0, 1))[pad:-pad, pad:-pad]
    return out[..., 0] if image.ndim == 2 else out


def degrade(sharp: np.ndarray, kernel: np.ndarray, noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Blur a uint8 RGB image, add Gaussian noise (sigma in [0, 1] units) and quantise back to uint8."""
    blurred = convolve(np.asarray(sharp, dtype=np.float64) / 255.0, kernel)
    if noise_sigma:
        blurred = blurred + rng.normal(0.0, noise_sigma, blurred.shape)
    return np.clip(np.round(blurred * 255.0), 0, 255).astype(np.uint8)


def random_kernel_params(rng: np.random.Generator) -> dict[str, float]:
    return {"length_px": float(rng.integers(BLUR_LENGTH_PX[0], BLUR_LENGTH_PX[1] + 1)), "angle_deg": float(round(rng.uniform(*BLUR_ANGLE_DEG), 2))}


def make_crops(image: np.ndarray, *, source: str, split: str, n_crops: int, crop: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """``n_crops`` paired records from one photograph, each with its own seeded kernel, noise and crop window."""
    h, w = image.shape[:2]
    if min(h, w) < crop:
        raise ValueError(f"{source}: {w}x{h} px is smaller than the {crop} px crop")
    records = []
    for i in range(n_crops):
        params = random_kernel_params(rng)
        kernel = motion_kernel(params["length_px"], params["angle_deg"])
        blurred_full = degrade(image, kernel, NOISE_SIGMA, rng)
        y, x = int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1))
        records.append(
            {
                "id": f"{source}-{i:02d}",
                "source": source,
                "split": split,
                "blurred": blurred_full[y : y + crop, x : x + crop].copy(),
                "sharp": image[y : y + crop, x : x + crop].copy(),
                "kernel": params,
                "kernel_array": kernel,
                "noise_sigma": NOISE_SIGMA,
                "window": [x, y, crop, crop],
            }
        )
    return records


def build_sample_dataset(data_dir: Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    """The default train/test split: 6 x 8 training crops and 3 x 4 test crops of 256 px, split by photograph."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for name, entry in SAMPLE_IMAGES.items():
        if entry["role"] not in splits:
            continue
        n = TRAIN_CROPS_PER_IMAGE if entry["role"] == "train" else TEST_CROPS_PER_IMAGE
        splits[entry["role"]] += make_crops(load_sample_image(name, data_dir), source=name, split=entry["role"], n_crops=n, crop=CROP, rng=rng)
    return splits


def build_new_data(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """Two images that play no part in adaptation or evaluation: a synthetic blur of ``text`` (paired, one fixed kernel)
    and ``clock_motion``, a real camera-motion photograph with no sharp reference (unpaired)."""
    sharp = load_sample_image("text", data_dir)
    kernel = motion_kernel(NEW_DATA_KERNEL["length_px"], NEW_DATA_KERNEL["angle_deg"])
    blurred = degrade(sharp, kernel, NOISE_SIGMA, np.random.default_rng(NEW_DATA_SEED))
    real = load_sample_image("clock_motion", data_dir)
    return [
        {"id": "text-synthetic", "source": "text", "split": "new", "blurred": blurred, "sharp": sharp, "kernel": dict(NEW_DATA_KERNEL), "kernel_array": kernel, "noise_sigma": NOISE_SIGMA},
        {"id": "clock-real-motion", "source": "clock_motion", "split": "new", "blurred": real, "sharp": None, "kernel": None, "kernel_array": None, "noise_sigma": None},
    ]


# --------------------------------------------------------------------------------------------------
# validation, splitting, manifests
# --------------------------------------------------------------------------------------------------


def _check_array(record_id: str, role: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"record {record_id!r}: {role} must be a numpy array, got {type(value).__name__}")
    if value.dtype != np.uint8:
        raise ValueError(f"record {record_id!r}: {role} must be uint8 (8-bit), got {value.dtype}")
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"record {record_id!r}: {role} must be H x W x 3 RGB, got shape {value.shape}")
    h, w = value.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        raise ValueError(f"record {record_id!r}: {role} is {w}x{h} px; each side must be within {MIN_SIDE}..{MAX_SIDE} px (resize or crop the image before supplying it)")
    return h, w


def validate_records(records: Sequence[Mapping[str, Any]], *, paired: bool = True) -> dict[str, Any]:
    """Structural validation before any model runs (VAL1-VAL6, DAT19). Raises ValueError naming the record and rule."""
    if not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, source, blurred, sharp} dictionaries")
    minimum = MIN_PAIRS if paired else 1
    if not minimum <= len(records) <= MAX_IMAGES:
        raise ValueError(f"{len(records)} records supplied; {minimum}..{MAX_IMAGES} are required for {'paired adaptation' if paired else 'inference'}")
    seen: set[str] = set()
    sizes = []
    identical = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be a dictionary, got {type(record).__name__}")
        missing = [k for k in ("id", "source", "blurred", "sharp") if k not in record]
        if missing:
            raise ValueError(f"record {index} is missing required fields {missing}")
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index}: id must be a non-empty string")
        if record_id in seen:
            raise ValueError(f"duplicate record id {record_id!r}; ids must be unique")
        seen.add(record_id)
        if not isinstance(record["source"], str) or not record["source"]:
            raise ValueError(f"record {record_id!r}: source must be a non-empty string (it groups crops of one photograph for the split)")
        h, w = _check_array(record_id, "blurred", record["blurred"])
        if paired:
            if record["sharp"] is None:
                raise ValueError(f"record {record_id!r}: paired mode needs a sharp reference; use unpaired inference for blurred-only images")
            hs, ws = _check_array(record_id, "sharp", record["sharp"])
            if (hs, ws) != (h, w):
                raise ValueError(f"record {record_id!r}: blurred is {w}x{h} px but sharp is {ws}x{hs} px; both images of a pair must have identical size")
            if np.array_equal(record["blurred"], record["sharp"]):
                identical.append(record_id)
        elif record["sharp"] is not None:
            _check_array(record_id, "sharp", record["sharp"])
        sizes.append((w, h))
    sources = sorted({r["source"] for r in records})
    if paired and len(sources) < 2:
        raise ValueError(f"all {len(records)} pairs come from one source ({sources[0]!r}); at least 2 distinct sources are needed so that the test split shares no photograph with training")
    widths, heights = zip(*sizes, strict=True)
    return {
        "records": len(records),
        "sources": len(sources),
        "paired": paired,
        "width_range": [min(widths), max(widths)],
        "height_range": [min(heights), max(heights)],
        "identical_pairs": identical,
        "ceilings": {"min_side": MIN_SIDE, "max_side": MAX_SIDE, "max_images": MAX_IMAGES, "min_pairs": MIN_PAIRS},
    }


def split_by_source(records: Sequence[Mapping[str, Any]], *, seed: int = 0, test_fraction: float = 0.25) -> dict[str, list[dict[str, Any]]]:
    """A seeded split that keeps every crop of one source on the same side (SPL5): at least one source in each split."""
    sources = sorted({r["source"] for r in records})
    if len(sources) < 2:
        raise ValueError("splitting needs at least 2 distinct sources")
    order = list(np.random.default_rng(seed).permutation(len(sources)))
    n_test = min(len(sources) - 1, max(1, int(round(len(sources) * test_fraction))))
    test_sources = {sources[i] for i in order[:n_test]}
    splits = {"train": [dict(r, split="train") for r in records if r["source"] not in test_sources], "test": [dict(r, split="test") for r in records if r["source"] in test_sources]}
    return splits


def _array_digest(array: np.ndarray | None) -> str | None:
    if array is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(array).tobytes() + str(array.shape).encode()).hexdigest()


def dataset_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Counts, per-record digests, a disjointness check across splits (records and sources) and one dataset digest."""
    rows = []
    for name, records in splits.items():
        for r in records:
            rows.append({"split": name, "id": r["id"], "source": r["source"], "shape": list(r["blurred"].shape), "blurred_sha256": _array_digest(r["blurred"]), "sharp_sha256": _array_digest(r.get("sharp")), "kernel": r.get("kernel")})
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(row["split"], set()).add(row["source"])
    names = list(by_split)
    overlap = sorted({s for i, a in enumerate(names) for b in names[i + 1 :] for s in by_split[a] & by_split[b]})
    if overlap:
        raise ValueError(f"sources appear in more than one split: {overlap} (leakage)")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("a record id appears in more than one split")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"counts": {name: len(records) for name, records in splits.items()}, "sources": {k: sorted(v) for k, v in by_split.items()}, "disjoint_sources": True, "digest": digest, "records": rows}


# --------------------------------------------------------------------------------------------------
# BYOD
# --------------------------------------------------------------------------------------------------


def _decode(name: str, data: bytes) -> tuple[np.ndarray, str | None]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return to_rgb(image)
    except UnidentifiedImageError as exc:
        raise ValueError(f"BYOD: {name} is not a readable image (PNG/JPEG/BMP/TIFF/WebP expected)") from exc
    except ValueError as exc:
        raise ValueError(f"BYOD: {name}: {exc}") from exc


def _safe_members(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Path-safe, size-bounded reading of a zip without extracting it (§20): absolute paths, ``..`` traversal and
    symlinks are refused; nothing is written to disk."""
    files: dict[str, bytes] = {}
    total = 0
    for info in archive.infolist():
        name = info.filename
        if info.is_dir():
            continue
        path = PurePosixPath(name)
        if name.startswith(("/", "\\")) or ":" in path.parts[0] or ".." in path.parts or "\\" in name:
            raise ValueError(f"BYOD zip member {name!r} has an absolute or traversing path; refused")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"BYOD zip member {name!r} is a symbolic link; refused")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"BYOD zip member {name!r} expands to {info.file_size} bytes (> {MAX_MEMBER_BYTES}); refused")
        total += info.file_size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError(f"BYOD zip expands to more than {MAX_ARCHIVE_BYTES} bytes; refused")
        if path.name.startswith(".") or "__MACOSX" in path.parts:
            continue
        files[name] = archive.read(info)
    return files


def _directory_members(root: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"BYOD directory entry {path} is a symbolic link; refused")
        if not path.is_file() or path.name.startswith("."):
            continue
        size = path.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise ValueError(f"BYOD file {path} is {size} bytes (> {MAX_MEMBER_BYTES}); refused")
        total += size
        if total > MAX_ARCHIVE_BYTES:
            raise ValueError(f"BYOD directory holds more than {MAX_ARCHIVE_BYTES} bytes; refused")
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def read_byod(path: str | Path) -> dict[str, Any]:
    """Read a BYOD zip or directory. Paired when it has ``sharp/`` and ``blurred/`` folders (files matched by stem);
    unpaired when it has only blurred images (in ``blurred/`` or at the top level). Raises ValueError naming the file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BYOD path {path} does not exist")
    if path.is_dir():
        members = _directory_members(path)
    else:
        if not zipfile.is_zipfile(path):
            raise ValueError(f"BYOD path {path.name} is neither a directory nor a zip archive")
        with zipfile.ZipFile(path) as archive:
            members = _safe_members(archive)
    images = {name: data for name, data in members.items() if PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES}
    ignored = sorted(set(members) - set(images))
    if not images:
        raise ValueError(f"BYOD: no images found (expected {', '.join(IMAGE_SUFFIXES)} files in sharp/ and blurred/)")
    # tolerate one wrapping folder (e.g. my_data/sharp/x.png)
    parts = {PurePosixPath(n).parts[0] for n in images}
    if len(parts) == 1 and all(len(PurePosixPath(n).parts) > 2 for n in images):
        prefix = next(iter(parts))
        images = {str(PurePosixPath(*PurePosixPath(n).parts[1:])): d for n, d in images.items() if n.startswith(prefix + "/")}
    folders: dict[str, dict[str, tuple[str, bytes]]] = {"sharp": {}, "blurred": {}, "": {}}
    for name, data in images.items():
        p = PurePosixPath(name)
        folder = p.parts[0].lower() if len(p.parts) > 1 else ""
        if folder not in folders or len(p.parts) > 2:
            raise ValueError(f"BYOD: {name} is not in sharp/ or blurred/ (put pairs in sharp/ and blurred/, matched by file name)")
        stem = p.stem
        if stem in folders[folder]:
            raise ValueError(f"BYOD: two files in {folder or 'the top level'} share the stem {stem!r} ({folders[folder][stem][0]}, {name}); stems must be unique")
        folders[folder][stem] = (name, data)
    if folders["sharp"] and folders[""]:
        raise ValueError("BYOD: images at the top level and in sharp/ — put the degraded images in blurred/")
    conversions = []
    if folders["sharp"]:
        blurred = folders["blurred"]
        if not blurred:
            raise ValueError("BYOD: sharp/ exists but blurred/ is empty; paired data needs both folders")
        unmatched = sorted(set(folders["sharp"]) ^ set(blurred))
        if unmatched:
            detail = [f"{folders['sharp'][s][0] if s in folders['sharp'] else blurred[s][0]}" for s in unmatched]
            raise ValueError(f"BYOD: files without a partner (pairs are matched by file stem in sharp/ and blurred/): {detail}")
        records = []
        for stem in sorted(blurred):
            b_name, b_data = blurred[stem]
            s_name, s_data = folders["sharp"][stem]
            b, b_note = _decode(b_name, b_data)
            s, s_note = _decode(s_name, s_data)
            conversions += [f"{n}: {note}" for n, note in ((b_name, b_note), (s_name, s_note)) if note]
            records.append({"id": stem, "source": stem, "blurred": b, "sharp": s, "kernel": None, "kernel_array": None, "noise_sigma": None})
        mode = "paired"
    else:
        pool = folders["blurred"] or folders[""]
        records = []
        for stem in sorted(pool):
            name, data = pool[stem]
            b, note = _decode(name, data)
            if note:
                conversions.append(f"{name}: {note}")
            records.append({"id": stem, "source": stem, "blurred": b, "sharp": None, "kernel": None, "kernel_array": None, "noise_sigma": None})
        mode = "unpaired"
    report = validate_records(records, paired=mode == "paired")
    digest = hashlib.sha256(b"".join(hashlib.sha256(members[k]).digest() for k in sorted(members))).hexdigest()
    return {"mode": mode, "records": records, "validation": report, "conversions": conversions, "ignored_files": ignored, "archive_digest": digest}
