"""Model-side tests on CPU without the real checkpoint: the carried upstream architecture (hash-verified), a tiny
random-init NAFNet, the fine-tuning scope, the safetensors artifact and its refusals, checkpoint verification and the
mirror fallback, and the .pth -> safetensors conversion on a random full-width stand-in."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from conftest import TINY_CONFIG, TINY_TLC, needs_torch, synthetic_image, synthetic_pairs
from nafnet_deblurring_pipeline import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
pytestmark = needs_torch


def tiny(seed: int = 0):
    return P.DeblurPipeline.from_config(TINY_CONFIG, seed=seed, tlc_train_size=TINY_TLC)


def test_upstream_files_match_the_pinned_upstream_digests():
    for rel, digest in P.UPSTREAM_ARCH_SHA256.items():
        assert hashlib.sha256((P.DEFAULT_ARCH_DIR / rel).read_bytes()).hexdigest() == digest
    manifest = json.loads((ROOT / "weights" / P.MODEL_KEY / P.MANIFEST_NAME).read_text())
    assert {f["path"]: f["sha256"] for f in manifest["architecture_source"]["files"]} == P.UPSTREAM_ARCH_SHA256
    assert manifest["architecture_source"]["networkConfig"]["width"] == P.ARCH_CONFIG["width"]


def test_tampered_upstream_file_is_refused(tmp_path: Path):
    copy = tmp_path / "nafnet"
    shutil.copytree(P.DEFAULT_ARCH_DIR, copy)
    target = copy / "basicsr/models/archs/NAFNet_arch.py"
    target.write_text(target.read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="sha256"):
        P.load_architecture(copy)
    P.load_architecture()  # restore the verified modules for later tests


def test_full_architecture_has_the_documented_parameter_count():
    arch = P.load_architecture()
    net = arch.NAFNet(**P.ARCH_CONFIG)
    assert sum(p.numel() for p in net.parameters()) == P.PARAMETER_COUNT
    counts = {scope: 0 for scope in P.TRAINABLE_SCOPES}
    for name, param in net.named_parameters():
        top = name.split(".", 1)[0]
        for scope, modules in P.TRAINABLE_SCOPES.items():
            counts[scope] += param.numel() if modules is None or top in modules else 0
    assert counts == {"decoder": 1322307, "decoder+middle": 3174211, "all": P.PARAMETER_COUNT}


def test_tlc_equals_global_pooling_for_small_inputs():
    import torch

    arch = P.load_architecture()
    plain = arch.NAFNet(**TINY_CONFIG)
    local = P.build_network(TINY_CONFIG, tlc_train_size=TINY_TLC)
    local.load_state_dict(plain.state_dict())
    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(plain.eval()(x), local.eval()(x), atol=1e-6)


def test_restore_keeps_size_and_dtype_for_odd_sizes():
    pipe = tiny()
    out = pipe.restore([synthetic_image(side=70), synthetic_image(side=64)[:50]])
    assert out[0].shape == (70, 70, 3) and out[1].shape == (50, 64, 3) and out[0].dtype == np.uint8


def test_finetune_updates_only_the_scope_and_is_reproducible():
    import torch

    records = synthetic_pairs(3, 2, side=64)
    a = tiny()
    before = {k: v.clone() for k, v in a.net.state_dict().items()}
    run = a.finetune(records, steps=3, lr=1e-3, batch_size=2, patch=32, scope="decoder", seed=1, log_every=1)
    assert run["trainable_parameters"] + run["frozen_parameters"] == run["total_parameters"]
    changed = {k.split(".", 1)[0] for k, v in a.net.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and changed <= {"ups", "decoders", "ending"}
    b = tiny()
    b.finetune(records, steps=3, lr=1e-3, batch_size=2, patch=32, scope="decoder", seed=1, log_every=1)
    for k, v in a.net.state_dict().items():
        assert torch.allclose(v, b.net.state_dict()[k]), k
    assert all(p.requires_grad for p in a.net.parameters())  # scope is undone after training
    with pytest.raises(ValueError, match="unknown trainable scope"):
        tiny().finetune(records, steps=1, lr=1e-3, batch_size=1, patch=32, scope="encoder", seed=0)
    with pytest.raises(ValueError, match="smaller than the 128 px patch"):
        tiny().finetune(records, steps=1, lr=1e-3, batch_size=1, patch=128, scope="all", seed=0)


def test_artifact_roundtrip_and_refusals(tmp_path: Path):
    records = synthetic_pairs(2, 2, side=64)
    pipe = tiny()
    pipe.finetune(records, steps=2, lr=1e-3, batch_size=2, patch=32, scope="all", seed=0)
    manifest = pipe.save_artifact(tmp_path / "art", metadata={"data": {"digest": "x"}})
    assert manifest["files"] == ["model.safetensors", "manifest.json"] and manifest["adaptation"]["scope"] == "all"
    reloaded = P.DeblurPipeline.from_artifact(tmp_path / "art")
    image = records[0]["blurred"]
    assert np.abs(reloaded.restore_float(image) - pipe.restore_float(image)).max() <= 1e-6
    # an unexpected file is refused
    extra = tmp_path / "extra"
    shutil.copytree(tmp_path / "art", extra)
    (extra / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="unexpected or missing"):
        P.DeblurPipeline.from_artifact(extra)
    # changed weights bytes are refused before deserialisation
    bad = tmp_path / "bad"
    shutil.copytree(tmp_path / "art", bad)
    data = bytearray((bad / "model.safetensors").read_bytes())
    data[-1] ^= 1
    (bad / "model.safetensors").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        P.DeblurPipeline.from_artifact(bad)
    # a different model identity is refused
    other = tmp_path / "other"
    shutil.copytree(tmp_path / "art", other)
    m = json.loads((other / "manifest.json").read_text())
    m["model"]["revision"] = "0" * 40
    (other / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="expected megvii-research/NAFNet"):
        P.DeblurPipeline.from_artifact(other)


def test_verify_checkpoint_refuses_wrong_size(tmp_path: Path):
    path = tmp_path / P.CHECKPOINT_FILE
    path.write_bytes(b"0" * 10)
    with pytest.raises(ValueError, match="size 10 != manifest"):
        P.verify_checkpoint(path)


def test_fetch_checkpoint_refuses_wrong_bytes_and_accepts_pinned_bytes(tmp_path: Path, monkeypatch):
    payload = b"pinned-bytes" * 100
    monkeypatch.setattr(P, "CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(P, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest())
    served = iter([b"tampered" * 150, payload])

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(P.urllib.request, "urlopen", lambda request, timeout: Response(next(served)))
    result = P.fetch_checkpoint(tmp_path)
    assert result["fetched"] and result["source"] == P.CHECKPOINT_MIRRORS[1]["url"]
    assert (tmp_path / P.CHECKPOINT_FILE).read_bytes() == payload and not list(tmp_path.glob("*.part"))
    (tmp_path / P.CHECKPOINT_FILE).unlink()
    monkeypatch.setattr(P.urllib.request, "urlopen", lambda request, timeout: Response(b"wrong" * 10))
    with pytest.raises(RuntimeError, match="refusing any other file"):
        P.fetch_checkpoint(tmp_path)
    assert not (tmp_path / P.CHECKPOINT_FILE).exists()


def test_manifest_identity_is_checked(tmp_path: Path):
    manifest = json.loads((ROOT / "weights" / P.MODEL_KEY / P.MANIFEST_NAME).read_text())
    assert P.load_manifest(ROOT / "weights" / P.MODEL_KEY / P.MANIFEST_NAME)["variant"] == P.MODEL_VARIANT
    manifest["files"][0]["sha256"] = "ab" * 32
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match the pinned checkpoint"):
        P.load_manifest(tmp_path / "m.json")
    (tmp_path / "d.json").write_text('{"modelId": "a", "modelId": "b"}')
    with pytest.raises(ValueError, match="duplicate keys"):
        P.load_manifest(tmp_path / "d.json")


def test_convert_checkpoint_on_a_random_full_width_stand_in(tmp_path: Path, monkeypatch):
    """The conversion path of the weights stage with a random checkpoint in the upstream layout (not the real bytes)."""
    import torch

    arch = P.load_architecture()
    torch.manual_seed(0)
    net = arch.NAFNet(**P.ARCH_CONFIG)
    pth = tmp_path / P.CHECKPOINT_FILE
    torch.save({"params": net.state_dict()}, pth)
    monkeypatch.setattr(P, "CHECKPOINT_BYTES", pth.stat().st_size)
    monkeypatch.setattr(P, "CHECKPOINT_SHA256", P.sha256_file(pth))
    record = P.convert_checkpoint(pth, tmp_path / "model.safetensors")
    assert record["max_abs_output_diff"] == 0.0 and record["parameters"] == P.PARAMETER_COUNT
    second = P.convert_checkpoint(pth, tmp_path / "again.safetensors")
    assert second["sha256"] == record["sha256"]  # the conversion is deterministic
    torch.save({"state_dict": net.state_dict()}, pth)
    monkeypatch.setattr(P, "CHECKPOINT_BYTES", pth.stat().st_size)
    monkeypatch.setattr(P, "CHECKPOINT_SHA256", P.sha256_file(pth))
    with pytest.raises(ValueError, match="'params'"):
        P.read_checkpoint_state(pth)


def test_psnr_loss_is_minus_batch_psnr():
    import torch

    pred, target = torch.zeros(2, 3, 8, 8), torch.full((2, 3, 8, 8), 0.1)
    assert float(P.psnr_loss(pred, target)) == pytest.approx(-20.0, abs=1e-4)
