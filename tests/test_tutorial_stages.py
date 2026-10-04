"""The isolated-environment tutorial path (NOTEBOOK_SPEC 2.2 §25.13): the kernel's carrier and `run_stage` helper, and
a CPU pre-flight of every stage of the carried runner.

* Kernel side (no model library needed): the generated notebook's own carrier cell writes and hash-verifies the carried
  files into a run directory, and a failing stage stops the kernel with a RuntimeError that repeats the stage's message.
* Pre-flight (torch required): every stage runs in order, in-process, against a tiny random-init NAFNet, a small
  synthetic dataset and synthetic new images. Each stage builds its own model, so everything a later stage uses crosses
  over through files. It proves the stage plumbing and the hand-offs, not the model's quality.
"""
# ruff: noqa: E501

from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import TINY_CONFIG, TINY_TLC, needs_torch, synthetic_image, synthetic_pairs

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE


def _infrastructure_sources() -> tuple[str, str]:
    notebook = build.render(ROOT, TEMPLATE, "test-revision")
    code = [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]
    carrier = next(s for s in code if s.startswith("# @title Infrastructure: write and verify the carried"))
    install = next(s for s in code if s.startswith("# @title Infrastructure: install the locked runtime"))
    return carrier, install


def _kernel(tmp_path: Path) -> dict:
    """The kernel namespace after the carrier cell and the `run_stage` definition, with the current interpreter
    standing in for the isolated environment's Python."""
    carrier, install = _infrastructure_sources()
    run_root = tmp_path / "run"
    run_root.mkdir()
    namespace = {"ROOT": run_root, "WEIGHTS": tmp_path / "weights", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec("import hashlib\nimport json\nimport subprocess\n" + carrier, namespace)  # noqa: S102 - the notebook's own cell
    definition = install[install.index("def run_stage(") : install.index("def load_record(")]
    exec(definition, namespace)  # noqa: S102
    return namespace


def test_carrier_writes_and_verifies_every_carried_file(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    run_root = kernel["ROOT"]
    for dest, source in TEMPLATE["carried"].items():
        assert (run_root / dest).read_bytes() == (ROOT / source).read_text(encoding="utf-8").encode("utf-8"), dest
    assert kernel["NOTEBOOK_SOURCE"]["revision"] == "test-revision"


def test_run_stage_reraises_the_stage_message(tmp_path: Path) -> None:
    kernel = _kernel(tmp_path)
    with pytest.raises(RuntimeError, match=r"Stage 'baseline' failed \(exit 2\): RuntimeError: data.json is missing"):
        kernel["run_stage"]("baseline")
    with pytest.raises(RuntimeError, match="BYOD path .* does not exist"):
        kernel["run_stage"]("byod", "--byod", tmp_path / "absent.zip")


# ---- CPU pre-flight ----------------------------------------------------------------------------------------------


def _png(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def stages(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    from nafnet_deblurring_pipeline import DeblurPipeline

    module = _load("tutorial_stages")
    root = tmp_path / "run"
    shutil.copytree(ROOT / "third_party", root / "third_party")

    def tiny(*_args):
        return DeblurPipeline.from_config(TINY_CONFIG, seed=0, tlc_train_size=TINY_TLC)

    def fake_checkpoint(run):
        return {"checkpoint": {"file": "stub.pth", "fetched": False, "source": "test stub"}, "base": {"file": "model.safetensors", "bytes": 1, "sha256": "0" * 64, "tensors": 1, "parameters": 1, "max_abs_output_diff": 0.0}, "parameters": 1, "distribution_source": {}, "upstream_source": {}}

    def sample(run):
        train = [dict(r, split="train") for r in synthetic_pairs(3, 2, side=80)]
        test = [dict(r, split="test", id="t" + r["id"], source="t" + r["source"]) for r in synthetic_pairs(2, 2, side=64, seed=10)]
        return {"train": train, "test": test}

    def new_data(run):
        paired = synthetic_pairs(1, 1, side=72, seed=20)[0]
        return [dict(paired, id="new-paired"), {"id": "new-unpaired", "source": "x", "blurred": synthetic_image(side=80, seed=3), "sharp": None, "kernel": None, "kernel_array": None}]

    monkeypatch.setattr(module, "stage_checkpoint", fake_checkpoint)
    monkeypatch.setattr(module, "load_base", tiny)
    monkeypatch.setattr(module, "load_sample", sample)
    monkeypatch.setattr(module, "load_new_data", new_data)
    module.ROOT_FOR_TEST = root
    return module


def _run(stages, stage: str, *extra: str) -> None:
    root = stages.ROOT_FOR_TEST
    code = stages.main(["--root", str(root), "--weights", str(root.parent / "weights"), "--stage", stage, *extra])
    error = root / "state" / f"{stage}.error.json"
    assert code == 0, error.read_text() if error.exists() else stage


@needs_torch
def test_every_stage_runs_in_order_and_hands_off_through_files(stages, tmp_path: Path) -> None:
    root = stages.ROOT_FOR_TEST
    for stage, extra in (("weights", ()), ("prepare", ()), ("baseline", ()), ("adapt", ("--steps", "2", "--batch-size", "2")), ("evaluate", ()), ("reload", ())):
        _run(stages, stage, *extra)
    out = root / "outputs"
    names = {p.name for p in out.iterdir()}
    for expected in ("weights.json", "dataset.json", "nafnet_deblurring_sample_pairs.png", "nafnet_deblurring_sample_pairs.csv", "nafnet_deblurring_baseline_report.json", "nafnet_deblurring_baselines.png", "nafnet_deblurring_adaptation.json", "nafnet_deblurring_artifact", "nafnet_deblurring_evaluation_report.json", "nafnet_deblurring_test_metrics.csv", "nafnet_deblurring_evaluation.png", "nafnet_deblurring_result.json", "nafnet_deblurring_new_images.png", "nafnet_deblurring_new_image_predictions.csv"):
        assert expected in names, expected
    dataset = json.loads((out / "dataset.json").read_text())
    assert all(v.startswith("rejected") for v in dataset["refusal_probes"].values())
    evaluation = json.loads((out / "nafnet_deblurring_evaluation_report.json").read_text())
    assert set(evaluation["table"]) == {"identity", "unsharp", "wiener_oracle", "pretrained", "adapted"}
    result = json.loads((out / "nafnet_deblurring_result.json").read_text())
    assert result["reload_parity"]["max_abs_float_diff"] <= stages.RELOAD_TOLERANCE
    assert result["new_images"][1]["note"].startswith("no sharp reference")
    assert "new-paired" in {r["id"] for r in result["new_images"]} and "adapted_psnr_db" in result["new_images"][0]
    assert sorted(p.name for p in (out / "nafnet_deblurring_artifact").iterdir()) == ["manifest.json", "model.safetensors"]

    # the optional activity starts from the base and changes only the scope
    _run(stages, "activity", "--scope", "all")
    activity = json.loads((out / "nafnet_deblurring_activity.json").read_text())
    assert activity["changed"]["scope"] == ["decoder", "all"] and activity["held_fixed"]["steps"] == 2
    rows = list(activity["comparison"].values())
    assert abs(rows[1]["first_logged_loss"] - rows[2]["first_logged_loss"]) < 1.0  # both started from the same weights

    # BYOD paired: the full sequence, with results written and the reload verified (DTR-m2)
    archive = tmp_path / "pairs.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for r in synthetic_pairs(4, 1, side=64, seed=30):
            handle.writestr(f"sharp/{r['id']}.png", _png(r["sharp"]))
            handle.writestr(f"blurred/{r['id']}.png", _png(r["blurred"]))
    _run(stages, "byod", "--byod", str(archive), "--steps", "2", "--batch-size", "2")
    paired_dir = next((out / "byod").glob("paired-*"))
    byod = json.loads((paired_dir / "byod_result.json").read_text())
    assert byod["mode"] == "paired" and byod["reload_parity"]["max_abs_float_diff"] <= stages.RELOAD_TOLERANCE
    assert set(byod["reports"]) == {"identity", "unsharp", "pretrained", "adapted"}  # no oracle without kernels
    assert (paired_dir / "byod_test_metrics.csv").exists() and (paired_dir / "artifact" / "manifest.json").exists()

    # BYOD unpaired: inference only, no metric
    unpaired = tmp_path / "blurred.zip"
    with zipfile.ZipFile(unpaired, "w") as handle:
        handle.writestr("blurred/a.png", _png(synthetic_image(side=70)))
    _run(stages, "byod", "--byod", str(unpaired))
    byod = json.loads(next((out / "byod").glob("unpaired-*/byod_result.json")).read_text())
    assert paired_dir.exists(), "a second BYOD run must not remove the first run's results"
    assert byod["mode"] == "unpaired" and "psnr" not in json.dumps(byod["predictions"]) and byod["adapted_model"] == "canonical sample artifact"

    # BYOD refusal: an unmatched file stops the stage with an actionable message (DAT19)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as handle:
        handle.writestr("sharp/a.png", _png(synthetic_image()))
        handle.writestr("blurred/b.png", _png(synthetic_image()))
    assert stages.main(["--root", str(root), "--weights", str(root.parent / "weights"), "--stage", "byod", "--byod", str(bad)]) == 2
    assert "without a partner" in json.loads((root / "state" / "byod.error.json").read_text())["message"]


@needs_torch
def test_later_stage_refuses_a_changed_dataset(stages, monkeypatch) -> None:
    _run(stages, "weights")
    _run(stages, "prepare")
    original = stages.load_sample
    monkeypatch.setattr(stages, "load_sample", lambda run: {k: v[1:] for k, v in original(run).items()})
    assert stages.main(["--root", str(stages.ROOT_FOR_TEST), "--weights", str(stages.ROOT_FOR_TEST.parent / "weights"), "--stage", "baseline"]) == 2
    assert "changed since 'prepare'" in (stages.ROOT_FOR_TEST / "state" / "baseline.error.json").read_text()
