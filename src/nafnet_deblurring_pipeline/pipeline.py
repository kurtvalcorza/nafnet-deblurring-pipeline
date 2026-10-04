"""NAFNet-GoPro-width32 deblurring: identity, checkpoint acquisition and verification, the carried upstream
architecture, bounded fine-tuning, inference, and a safetensors artifact with a manifest.

Trust boundary. The upstream checkpoint is a PyTorch pickle (``.pth``). It is refused unless its byte size and SHA-256
match the pinned values, and it is then read with ``torch.load(..., weights_only=True)``, which restricts unpickling to
tensors and primitive containers. It is unpickled once, converted to safetensors, and every later load uses the
safetensors file. Path-safe handling does not make an untrusted pickle safe; the pinned digest is what limits the
bytes that reach ``torch.load``.

Architecture. ``NAFNet_arch.py``, ``arch_util.py`` and ``local_arch.py`` are carried verbatim from
``megvii-research/NAFNet`` at ``MODEL_REVISION`` under ``third_party/nafnet/`` and verified against
``UPSTREAM_ARCH_SHA256`` before they are executed. They import ``basicsr.*`` by absolute name; ``load_architecture``
registers empty ``basicsr`` package modules and a one-function ``basicsr.utils`` shim (``get_root_logger``, used only
by an unused helper) so that none of upstream's training framework, OpenCV or LMDB dependencies is needed.

Every model-library import (torch, safetensors) is inside a function body: importing this module, validating data and
refusing a bad request never load torch.
"""
# ruff: noqa: E501  -- provenance strings and refusal messages are kept on one line

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import types
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---- identity (MOD1-MOD3) -------------------------------------------------------------------------------------------
MODEL_ID = "megvii-research/NAFNet"
MODEL_REVISION = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
MODEL_LICENSE = "MIT"
MODEL_KEY = "nafnet-gopro-width32"
MODEL_VARIANT = "NAFNet-GoPro-width32"
UPSTREAM_REPOSITORY = "https://github.com/megvii-research/NAFNet"
UPSTREAM_CHECKPOINT_LISTING = "readme.md (Results and Pre-trained Models table), Google Drive file id 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj"
CHECKPOINT_FILE = "NAFNet-GoPro-width32.pth"
CHECKPOINT_BYTES = 68671121
CHECKPOINT_SHA256 = "19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c"
# Credential-free mirrors of the exact upstream bytes, tried in order. Each is accepted only when the downloaded bytes
# match CHECKPOINT_BYTES and CHECKPOINT_SHA256, so a mirror can never substitute a different model (MOD8).
CHECKPOINT_MIRRORS = (
    {"provider": "huggingface", "repository": "nyanko7/nafnet-models", "url": "https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth"},
    {"provider": "huggingface", "repository": "mikestealth/nafnet-models", "url": "https://huggingface.co/mikestealth/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth"},
)
ARCH_CONFIG: dict[str, Any] = {"img_channel": 3, "width": 32, "middle_blk_num": 1, "enc_blk_nums": [1, 1, 1, 28], "dec_blk_nums": [1, 1, 1, 1]}
TLC_TRAIN_SIZE = (1, 3, 256, 256)  # NAFNetLocal (test-time local converter) as in options/test/GoPro/NAFNet-width32.yml
PARAMETER_COUNT = 17111907
UPSTREAM_ARCH_SHA256 = {
    "basicsr/models/archs/NAFNet_arch.py": "01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967",
    "basicsr/models/archs/arch_util.py": "5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005",
    "basicsr/models/archs/local_arch.py": "c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26",
}
MANIFEST_NAME = "dimer-base-manifest.json"
BASE_SAFETENSORS = "model.safetensors"
ARTIFACT_FORMAT = "dimer_nafnet_restoration_artifact"
ARTIFACT_FORMAT_VERSION = 1
ARTIFACT_WEIGHTS = "model.safetensors"
ARTIFACT_MANIFEST = "manifest.json"
# Adaptation scopes (FT5): which top-level submodules receive gradient updates; everything else is frozen.
TRAINABLE_SCOPES: dict[str, tuple[str, ...] | None] = {
    "decoder": ("ups", "decoders", "ending"),
    "decoder+middle": ("ups", "decoders", "ending", "middle_blks"),
    "all": None,
}
DEFAULT_ARCH_DIR = Path(__file__).resolve().parents[2] / "third_party" / "nafnet"


# --------------------------------------------------------------------------------------------------
# file integrity and checkpoint acquisition
# --------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Parse a snapshot manifest strictly (duplicate keys refused) and check it names the pinned identity."""

    def no_duplicates(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{path.name}: duplicate keys {sorted({k for k in keys if keys.count(k) > 1})}")
        return dict(pairs)

    manifest = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    expected = {"modelId": MODEL_ID, "revision": MODEL_REVISION, "modelKey": MODEL_KEY, "license": MODEL_LICENSE}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path.name}: {key} {manifest.get(key)!r} != pinned {value!r}")
    files = manifest.get("files", [])
    if [f.get("path") for f in files] != [CHECKPOINT_FILE] or files[0].get("bytes") != CHECKPOINT_BYTES or files[0].get("sha256") != CHECKPOINT_SHA256:
        raise ValueError(f"{path.name}: the file entry does not match the pinned checkpoint {CHECKPOINT_FILE} ({CHECKPOINT_BYTES} bytes, sha256 {CHECKPOINT_SHA256[:16]}…)")
    return manifest


def verify_checkpoint(path: Path) -> dict[str, Any]:
    """Size and SHA-256 of the checkpoint against the pins (INT1-INT6). Raises ValueError naming the file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run the weights stage first")
    size = path.stat().st_size
    if size != CHECKPOINT_BYTES:
        raise ValueError(f"{path.name}: size {size} != manifest {CHECKPOINT_BYTES}")
    digest = sha256_file(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f"{path.name}: sha256 {digest} != manifest {CHECKPOINT_SHA256}")
    return {"file": path.name, "bytes": size, "sha256": digest}


def fetch_checkpoint(weights_dir: Path, *, allow_download: bool = True, attempts: int = 3) -> dict[str, Any]:
    """Make ``weights_dir/CHECKPOINT_FILE`` hold the pinned bytes, downloading from the mirrors only when absent.

    A download is written to ``*.part`` and renamed only after its size and SHA-256 match; a mismatching mirror is
    reported and the next mirror is tried for the *same* pinned bytes. Nothing unverified is ever left in place."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / CHECKPOINT_FILE
    if target.is_file():
        return {**verify_checkpoint(target), "fetched": False, "source": "already staged"}
    if not allow_download:
        raise FileNotFoundError(f"{target} is absent and downloads are disabled")
    failures = []
    for mirror in CHECKPOINT_MIRRORS:
        part = target.with_suffix(".part")
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(mirror["url"], headers={"User-Agent": "dimer-nafnet-tutorial"})
                with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
                    while block := response.read(1 << 20):
                        out.write(block)
                        if out.tell() > CHECKPOINT_BYTES:
                            break
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                part.unlink(missing_ok=True)
                if attempt == attempts - 1:
                    failures.append(f"{mirror['repository']}: {type(exc).__name__}: {exc}")
                else:
                    time.sleep(2**attempt)
        if not part.is_file():
            continue
        size, digest = part.stat().st_size, sha256_file(part)
        if size == CHECKPOINT_BYTES and digest == CHECKPOINT_SHA256:
            part.replace(target)
            return {"file": target.name, "bytes": size, "sha256": digest, "fetched": True, "source": mirror["url"]}
        part.unlink(missing_ok=True)
        failures.append(f"{mirror['repository']}: got {size} bytes with sha256 {digest}, pinned {CHECKPOINT_BYTES} / {CHECKPOINT_SHA256}")
    raise RuntimeError("no mirror delivered the pinned checkpoint bytes (refusing any other file): " + "; ".join(failures))


# --------------------------------------------------------------------------------------------------
# carried upstream architecture
# --------------------------------------------------------------------------------------------------


def load_architecture(arch_dir: Path | None = None) -> types.ModuleType:
    """Execute the verbatim upstream ``NAFNet_arch`` (and its two local dependencies) after verifying each file's
    SHA-256; returns the module, which defines ``NAFNet`` and ``NAFNetLocal``."""
    import importlib.util

    arch_dir = Path(arch_dir or DEFAULT_ARCH_DIR)
    if "basicsr.models.archs.NAFNet_arch" in sys.modules:
        module = sys.modules["basicsr.models.archs.NAFNet_arch"]
        if getattr(module, "__dimer_arch_dir__", None) == str(arch_dir):
            return module
    for rel, expected in UPSTREAM_ARCH_SHA256.items():
        path = arch_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"carried upstream file {rel} is missing under {arch_dir}")
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"carried upstream file {rel}: sha256 {digest} != pinned {expected} (megvii-research/NAFNet@{MODEL_REVISION[:7]})")
    import logging

    for name in ("basicsr", "basicsr.models", "basicsr.models.archs"):
        package = types.ModuleType(name)
        package.__path__ = [str(arch_dir / name.replace(".", "/"))]
        sys.modules[name] = package
    shim = types.ModuleType("basicsr.utils")
    shim.get_root_logger = logging.getLogger  # the only name arch_util imports from basicsr.utils
    sys.modules["basicsr.utils"] = shim
    module = None
    for stem in ("arch_util", "local_arch", "NAFNet_arch"):
        name = f"basicsr.models.archs.{stem}"
        spec = importlib.util.spec_from_file_location(name, arch_dir / "basicsr" / "models" / "archs" / f"{stem}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    module.__dimer_arch_dir__ = str(arch_dir)
    return module


def build_network(config: Mapping[str, Any] = ARCH_CONFIG, *, tlc_train_size: Sequence[int] = TLC_TRAIN_SIZE, arch_dir: Path | None = None, seed: int = 0):
    """``NAFNetLocal`` with the upstream GoPro test settings. Its construction runs one forward pass on a seeded random
    tensor to size the local pooling windows (upstream behaviour); the global RNG state is restored afterwards."""
    import torch

    arch = load_architecture(arch_dir)
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        net = arch.NAFNetLocal(train_size=tuple(tlc_train_size), fast_imp=False, **dict(config))
    finally:
        torch.random.set_rng_state(state)
    return net


# --------------------------------------------------------------------------------------------------
# checkpoint reading and conversion
# --------------------------------------------------------------------------------------------------


def read_checkpoint_state(path: Path) -> dict[str, Any]:
    """Verify, then unpickle with ``weights_only=True``; return the ``params`` state dict (tensors only)."""
    import torch

    verify_checkpoint(path)
    source = str(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(f"{Path(path).name}: expected a dict with a 'params' state dict, found keys {sorted(payload)[:5] if isinstance(payload, dict) else type(payload).__name__}")
    state = payload["params"]
    bad = [k for k, v in state.items() if not isinstance(v, torch.Tensor)]
    if bad:
        raise ValueError(f"{Path(path).name}: non-tensor entries in the state dict: {bad[:5]}")
    return dict(state)


def save_state(state: Mapping[str, Any], path: Path, metadata: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Write a float state dict as safetensors (contiguous CPU tensors, sorted keys); return bytes, digest, tensors."""
    from safetensors.torch import save_file

    tensors = {k: v.detach().to("cpu").contiguous() for k, v in sorted(state.items())}
    # one metadata key holding sorted JSON: safetensors does not order several metadata keys, which would make the
    # file bytes (and so the digest) differ between otherwise identical saves
    save_file(tensors, str(path), metadata={"dimer": json.dumps(dict(metadata or {}), sort_keys=True)})
    return {"file": Path(path).name, "bytes": Path(path).stat().st_size, "sha256": sha256_file(Path(path)), "tensors": len(tensors), "parameters": int(sum(t.numel() for t in tensors.values()))}


def load_state(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    from safetensors.torch import load_file

    if expected_sha256 is not None:
        digest = sha256_file(Path(path))
        if digest != expected_sha256:
            raise ValueError(f"{Path(path).name}: sha256 {digest} != recorded {expected_sha256}")
    return load_file(str(path))


def convert_checkpoint(checkpoint: Path, out: Path, *, arch_dir: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    """``.pth`` → safetensors (DER1-DER7): strict load into the carried architecture, save, reload the safetensors into a
    second network and compare outputs on a seeded input. Returns the derived file record and the comparison."""
    import torch

    state = read_checkpoint_state(checkpoint)
    reference = build_network(arch_dir=arch_dir)
    reference.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in reference.parameters())
    if n_params != PARAMETER_COUNT:
        raise ValueError(f"the checkpoint loads {n_params} parameters, expected {PARAMETER_COUNT}")
    record = save_state(state, out, metadata={"source_sha256": CHECKPOINT_SHA256, "model_id": MODEL_ID, "revision": MODEL_REVISION, "variant": MODEL_VARIANT})
    derived = build_network(arch_dir=arch_dir)
    derived.load_state_dict(load_state(out, record["sha256"]), strict=True)
    probe = torch.rand((1, 3, 64, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        reference.to(device).eval()
        derived.to(device).eval()
        diff = float((reference(probe.to(device)) - derived(probe.to(device))).abs().max())
    if diff != 0.0:
        raise ValueError(f"the safetensors conversion changed the output (max |diff| {diff}); refusing it")
    return {**record, "derived_from_sha256": CHECKPOINT_SHA256, "conversion": f"safetensors save_file (torch {torch.__version__})", "max_abs_output_diff": diff}


# --------------------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------------------


def to_tensor(images: Sequence[np.ndarray]):
    import torch

    batch = np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in images]).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(batch))


def to_uint8(tensor) -> list[np.ndarray]:
    array = tensor.detach().clamp(0, 1).mul(255.0).round().to("cpu").numpy().astype(np.uint8)
    return [a.transpose(1, 2, 0).copy() for a in array]


def psnr_loss(pred, target):
    """Upstream ``PSNRLoss`` (loss_weight 1, mean): ``10/ln(10) * mean_b log(mse_b + 1e-8)`` = minus the batch-mean PSNR
    on [0, 1] images, so a lower loss means a higher training-batch PSNR."""
    import torch

    return (10.0 / np.log(10.0)) * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


def runtime_versions() -> dict[str, Any]:
    import numpy
    import safetensors
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "safetensors": safetensors.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "cpu",
    }


def set_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeblurPipeline:
    """A NAFNetLocal network plus the identity of the weights it holds."""

    def __init__(self, net: Any, *, device: str, config: Mapping[str, Any], weights: Mapping[str, Any], tlc_train_size: Sequence[int] = TLC_TRAIN_SIZE) -> None:
        self.net = net.to(device).eval()
        self.device = device
        self.config = dict(config)
        self.weights = dict(weights)
        self.tlc_train_size = tuple(tlc_train_size)
        self.adaptation: dict[str, Any] | None = None

    # ---- construction --------------------------------------------------------------------------------------------
    @classmethod
    def from_base(cls, base_safetensors: Path, *, expected_sha256: str, device: str = "cpu", arch_dir: Path | None = None) -> DeblurPipeline:
        """The pretrained model from the verified, converted safetensors file (never from the pickle)."""
        net = build_network(arch_dir=arch_dir)
        net.load_state_dict(load_state(base_safetensors, expected_sha256), strict=True)
        return cls(net, device=device, config=ARCH_CONFIG, weights={"kind": "pretrained", "file": Path(base_safetensors).name, "sha256": expected_sha256, "derived_from_sha256": CHECKPOINT_SHA256})

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, seed: int = 0, device: str = "cpu", tlc_train_size: Sequence[int] = (1, 3, 64, 64), arch_dir: Path | None = None) -> DeblurPipeline:
        """A randomly initialised network of any size (offline tests use a tiny width)."""
        import torch

        torch.manual_seed(seed)
        net = build_network(config, tlc_train_size=tlc_train_size, arch_dir=arch_dir, seed=seed)
        return cls(net, device=device, config=config, weights={"kind": "random-init", "seed": seed}, tlc_train_size=tlc_train_size)

    # ---- inference -----------------------------------------------------------------------------------------------
    def restore(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Deblur uint8 RGB images one at a time (any size within the validated ceilings); uint8 outputs, same size."""
        import torch

        outputs = []
        self.net.eval()
        with torch.no_grad():
            for image in images:
                outputs += to_uint8(self.net(to_tensor([image]).to(self.device)))
        return outputs

    def restore_float(self, image: np.ndarray):
        """The unrounded float output for one image (used for reload equivalence)."""
        import torch

        self.net.eval()
        with torch.no_grad():
            return self.net(to_tensor([image]).to(self.device)).clamp(0, 1).to("cpu").numpy()[0]

    # ---- adaptation ----------------------------------------------------------------------------------------------
    def set_trainable(self, scope: str) -> dict[str, int]:
        if scope not in TRAINABLE_SCOPES:
            raise ValueError(f"unknown trainable scope {scope!r}; choose one of {sorted(TRAINABLE_SCOPES)}")
        modules = TRAINABLE_SCOPES[scope]
        trainable = total = 0
        for name, param in self.net.named_parameters():
            param.requires_grad = modules is None or name.split(".", 1)[0] in modules
            total += param.numel()
            trainable += param.numel() if param.requires_grad else 0
        return {"trainable_parameters": trainable, "total_parameters": total, "frozen_parameters": total - trainable}

    def finetune(self, train: Sequence[Mapping[str, Any]], *, steps: int, lr: float, batch_size: int, patch: int, scope: str, seed: int, log_every: int = 25) -> dict[str, Any]:
        """Bounded gradient fine-tuning with the upstream PSNR loss on random ``patch``-px crops (with the 8 flips and
        rotations as augmentation) of the training pairs. Mutates ``self.net``; callers start from the verified base."""
        import torch

        if not 1 <= steps <= 5000:
            raise ValueError(f"steps must be within 1..5000, got {steps}")
        if not 1 <= batch_size <= 32:
            raise ValueError(f"batch_size must be within 1..32, got {batch_size}")
        if not 0 < lr <= 1e-2:
            raise ValueError(f"lr must be within (0, 1e-2], got {lr}")
        small = [r["id"] for r in train if min(r["blurred"].shape[:2]) < patch]
        if small:
            raise ValueError(f"training pairs smaller than the {patch} px patch: {small[:5]}")
        set_determinism(seed)
        counts = self.set_trainable(scope)
        params = [p for p in self.net.parameters() if p.requires_grad]
        optimiser = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.9), weight_decay=0.0)
        rng = np.random.default_rng(seed)
        history = []
        started = time.perf_counter()
        self.net.train()
        running = []
        for step in range(1, steps + 1):
            blurred, sharp = [], []
            for index in rng.integers(0, len(train), size=batch_size):
                record = train[int(index)]
                h, w = record["blurred"].shape[:2]
                y, x = int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1))
                k, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
                pair = []
                for array in (record["blurred"], record["sharp"]):
                    crop = np.rot90(array[y : y + patch, x : x + patch], k)
                    pair.append(np.ascontiguousarray(crop[:, ::-1] if flip else crop))
                blurred.append(pair[0])
                sharp.append(pair[1])
            inputs, targets = to_tensor(blurred).to(self.device), to_tensor(sharp).to(self.device)
            optimiser.zero_grad(set_to_none=True)
            loss = psnr_loss(self.net(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {step}; lower the learning rate")
            loss.backward()
            optimiser.step()
            running.append(loss.item())
            if step % log_every == 0 or step == steps:
                entry = {"step": step, "loss": round(float(np.mean(running)), 4), "train_batch_psnr_db": round(-float(np.mean(running)), 2), "seconds": round(time.perf_counter() - started, 1)}
                history.append(entry)
                print(entry, flush=True)
                running = []
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = True
        self.adaptation = {
            "method": "gradient fine-tuning of a parameter subset (no adapter, no PEFT)",
            "scope": scope,
            "trainable_modules": list(TRAINABLE_SCOPES[scope] or ("all",)),
            **counts,
            "optimizer": "AdamW(betas=(0.9, 0.9), weight_decay=0)",
            "loss": "PSNRLoss (upstream), on [0, 1] RGB",
            "lr": lr,
            "steps": steps,
            "batch_size": batch_size,
            "patch": patch,
            "augmentation": "random crop + 8 flips/rotations, seeded",
            "seed": seed,
            "precision": "float32",
            "device": self.device,
            "train_pairs": len(train),
            "seconds": round(time.perf_counter() - started, 1),
            "history": history,
        }
        self.weights = {"kind": "adapted", "base": self.weights}
        return self.adaptation

    # ---- artifact ------------------------------------------------------------------------------------------------
    def save_artifact(self, directory: Path, *, metadata: Mapping[str, Any]) -> dict[str, Any]:
        """``model.safetensors`` (the full adapted state dict) + ``manifest.json`` with identity, base digests,
        architecture, adaptation configuration and the caller's metadata (ART1-ART8)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        record = save_state(self.net.state_dict(), directory / ARTIFACT_WEIGHTS, metadata={"format": ARTIFACT_FORMAT, "model_id": MODEL_ID, "revision": MODEL_REVISION})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "formatVersion": ARTIFACT_FORMAT_VERSION,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "variant": MODEL_VARIANT, "key": MODEL_KEY, "license": MODEL_LICENSE},
            "base": {"checkpoint_file": CHECKPOINT_FILE, "checkpoint_sha256": CHECKPOINT_SHA256, "checkpoint_bytes": CHECKPOINT_BYTES, "weights": self.weights.get("base", self.weights)},
            "architecture": {"class": "NAFNetLocal", "config": self.config, "tlc_train_size": list(self.tlc_train_size), "upstream_files_sha256": dict(UPSTREAM_ARCH_SHA256)},
            "weights": record,
            "files": [ARTIFACT_WEIGHTS, ARTIFACT_MANIFEST],
            "serialization": {"type": "safetensors", "code_capable": False},
            "adaptation": self.adaptation,
            **dict(metadata),
        }
        (directory / ARTIFACT_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
        return manifest

    @classmethod
    def from_artifact(cls, directory: Path, *, device: str = "cpu", arch_dir: Path | None = None) -> DeblurPipeline:
        """Verify the manifest (format, identity, file list, weight bytes and SHA-256) **before** deserialising, then
        build the network from the recorded architecture and load the weights strictly."""
        directory = Path(directory)
        manifest_path = directory / ARTIFACT_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} is missing")

        def no_duplicates(pairs):
            keys = [k for k, _ in pairs]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{ARTIFACT_MANIFEST}: duplicate keys")
            return dict(pairs)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
        if (manifest.get("format"), manifest.get("formatVersion")) != (ARTIFACT_FORMAT, ARTIFACT_FORMAT_VERSION):
            raise ValueError(f"unsupported artifact format {manifest.get('format')!r} v{manifest.get('formatVersion')}")
        model = manifest.get("model", {})
        if (model.get("id"), model.get("revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(f"artifact names {model.get('id')}@{model.get('revision')}, expected {MODEL_ID}@{MODEL_REVISION}")
        present = sorted(p.name for p in directory.iterdir())
        if present != sorted(manifest["files"]):
            raise ValueError(f"artifact directory holds {present}, manifest lists {sorted(manifest['files'])}; unexpected or missing files")
        weights = directory / manifest["weights"]["file"]
        if weights.stat().st_size != manifest["weights"]["bytes"]:
            raise ValueError(f"{weights.name}: size {weights.stat().st_size} != manifest {manifest['weights']['bytes']}")
        architecture = manifest["architecture"]
        if architecture.get("upstream_files_sha256") != UPSTREAM_ARCH_SHA256:
            raise ValueError("artifact was built with different upstream architecture files")
        state = load_state(weights, manifest["weights"]["sha256"])
        net = build_network(architecture["config"], tlc_train_size=architecture["tlc_train_size"], arch_dir=arch_dir)
        net.load_state_dict(state, strict=True)
        pipe = cls(net, device=device, config=architecture["config"], weights={"kind": "artifact", "sha256": manifest["weights"]["sha256"]}, tlc_train_size=architecture["tlc_train_size"])
        pipe.adaptation = manifest.get("adaptation")
        pipe.manifest = manifest
        return pipe


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() and os.environ.get("DIMER_FORCE_CPU") != "1" else "cpu"
