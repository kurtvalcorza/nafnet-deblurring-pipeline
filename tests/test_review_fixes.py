"""Regression tests for the 2026-10-05 notebook review findings (NAF-m1..m5).

They need only CI's lightweight dependencies (no torch): the generated notebook's own cell sources are executed with
stand-ins (a fake isolated interpreter, a stubbed Colab upload), and the `prepare` stage runs on the pinned
scikit-image photographs. None of this is model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"naf_fix_{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


def _markdown() -> str:
    return "\n".join(c["source"] for c in _cells() if c["cell_type"] == "markdown")


def test_naf_m1_no_stale_hosted_run_statement_and_timings_quote_the_t4_run() -> None:
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "has not been recorded yet" not in text
    prerequisites = next(c["source"] for c in _cells() if c["source"].startswith("## Prerequisites"))
    assert "Tesla T4, 4 October 2026" in prerequisites and "66 s" in prerequisites and "0.18 s per step" in prerequisites
    assert "an estimate" in prerequisites  # the 9 GB disk figure is labelled


def test_naf_no_leftover_placeholders_or_kernel_installs() -> None:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    text = "\n".join(c["source"] for c in cells if not c["metadata"].get("dimer", {}).get("embedded_sources"))  # carried code may hold f-strings
    assert "{{" not in text and "{MODEL_ID}" not in text and "@P:" not in text
    kernel = "\n".join(c["source"] for c in _cells() if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert "%pip" not in kernel and "!pip" not in kernel and "'-m', 'pip'" not in kernel


def test_naf_m2_prepare_reports_and_records_every_greyscale_replication(tmp_path: Path) -> None:
    pytest.importorskip("skimage")
    stages = _load("tutorial_stages")
    run = stages.Run(tmp_path / "run", tmp_path / "weights", argparse.Namespace())
    stages.stage_prepare(run)
    expected = {"brick", "grass", "camera", "gravel", "text", "clock_motion"}
    dataset = json.loads((run.out / "dataset.json").read_text(encoding="utf-8"))
    assert {k for k, v in dataset["colour_conversions"].items() if v} == expected
    with (run.out / f"{stages.STEM}_sample_pairs.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["source"] for r in rows if r["source_conversion"] == "greyscale replicated to RGB"} == {"brick", "grass", "camera", "gravel"}
    assert sum(r["source_conversion"] != "none" for r in rows if r["split"] == "train") == 24
    assert "Four of the nine photographs are greyscale" in _markdown()


def test_naf_m3_prose_matches_the_recorded_run() -> None:
    md = _markdown()
    assert "sixteen times" not in md and "about thirteen times" in md and round(17_111_907 / 1_322_307) == 13
    assert "(gravel) is the hardest to restore and often gains least" not in md
    assert "printed-text" not in md and "handwritten formulae" in md
    assert "ratio **below 1**" in md


class _Files:
    def __init__(self, result: dict) -> None:
        self.result = result

    def upload(self) -> dict:
        return self.result


def _byod_cell() -> str:
    return next(c["source"] for c in _cells() if c["cell_type"] == "code" and "USE_BYOD = False" in c["source"])


@pytest.mark.parametrize("uploaded", [{}, {"a.zip": b"x", "b.zip": b"y"}])
def test_naf_m4_cancelled_or_multi_file_upload_stops_with_the_rule(tmp_path: Path, monkeypatch, uploaded: dict) -> None:
    colab = types.ModuleType("google.colab")
    colab.files = _Files(uploaded)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    calls = []
    namespace = {"ROOT": tmp_path, "run_stage": lambda *a: calls.append(a), "LEARNING_RATE": 1e-4, "BATCH_SIZE": 4, "TRAINABLE_SCOPE": "decoder"}
    with pytest.raises(ValueError, match=r"Upload exactly one \.zip file"):
        exec(_byod_cell().replace("USE_BYOD = False", "USE_BYOD = True"), namespace)  # noqa: S102
    assert not calls




# ---- NAF-m5: the uv-template re-run fixes (idempotent Section 1, environment reuse, run_stage guard) ------------------


def _cells() -> list[dict]:
    return build.render(ROOT, TEMPLATE, "test-revision")["cells"]


def _check_cell() -> str:
    return next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: check the runtime"))


def _run_check_cell(namespace: dict, source: str | None = None) -> dict:
    exec(source or _check_cell(), namespace)  # noqa: S102 - the notebook's own cell
    return namespace


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_naf_m5_section1_rerun_keeps_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    namespace = _run_check_cell({})
    first = namespace["ROOT"]
    assert first.is_dir() and first.parent == tmp_path / "outputs" / TEMPLATE["stem"]
    _run_check_cell(namespace)  # re-running Section 1 alone
    assert namespace["ROOT"] == first, "a Section 1 re-run must not strand the later cells in a new, empty run directory"
    _run_check_cell(namespace, _check_cell().replace("NEW_RUN_DIRECTORY = False", "NEW_RUN_DIRECTORY = True"))
    assert namespace["ROOT"] != first


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_naf_m5_environment_is_keyed_on_the_lock_not_the_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: types.SimpleNamespace(free=10**13))  # the disk check is not under test
    a = _run_check_cell({})
    b = _run_check_cell({})
    assert a["ROOT"] != b["ROOT"] and a["ENV_ROOT"] == b["ENV_ROOT"]
    assert a["ROOT"].name not in str(a["ENV_ROOT"])


def _fake_ipython(monkeypatch) -> None:
    display = types.ModuleType("IPython.display")
    display.Image = display.display = lambda *a, **k: None
    package = types.ModuleType("IPython")
    package.display = display
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display)


@pytest.mark.skipif(sys.platform != "linux", reason="executes the Section 1/3 kernel cells, which refuse a non-Linux x86_64 runtime and run a POSIX venv/bin/python (Linux runtimes only)")
def test_naf_m5_install_cell_reuses_a_complete_environment_without_downloading(tmp_path: Path, monkeypatch) -> None:
    _fake_ipython(monkeypatch)
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    env_root = tmp_path / "env"
    python = env_root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\necho '" + json.dumps({"python": "3.12.12", "torch": "x", "numpy": "x", "safetensors": "x", "cuda": False}) + "'\n", encoding="utf-8")
    python.chmod(0o755)
    lock_sha = "a" * 64
    spec = {"lock_sha256": lock_sha, "python": TEMPLATE["managed_python"], "uv": TEMPLATE["uv"]["version"]}
    (env_root / "ready.json").write_text(json.dumps(spec), encoding="utf-8")

    def no_network(*_a, **_k):
        raise AssertionError("a matching environment must be reused, not rebuilt")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    namespace = {"ENV_ROOT": env_root, "ROOT": tmp_path, "CARRIED_HASHES": {TEMPLATE["lock"]: lock_sha}, "NOTEBOOK_SOURCE": {"revision": "r"}, "SESSION_START": time.perf_counter(), "Path": Path}
    exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert namespace["ENV_REUSED"] is True
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        assert name not in namespace["ENV"]
    assert namespace["ENV"]["MPLBACKEND"] == "Agg"
    namespace["CARRIED_HASHES"] = {TEMPLATE["lock"]: "b" * 64}  # a different lock is never reused
    with pytest.raises(AssertionError, match="must be reused"):
        exec("import hashlib, json, os, shutil, subprocess, time\n" + install, namespace)  # noqa: S102
    assert not (env_root / "ready.json").exists()


def test_naf_m5_run_stage_names_the_cells_to_rerun_when_the_run_directory_is_empty(tmp_path: Path) -> None:
    install = next(c["source"] for c in _cells() if c["cell_type"] == "code" and c["source"].startswith("# @title Infrastructure: install the locked runtime"))
    namespace = {"ROOT": tmp_path / "empty", "WEIGHTS": tmp_path / "w", "PYTHON": Path(sys.executable), "ENV": dict(os.environ)}
    exec(install[install.index("def run_stage(") : install.index("def load_record(")], namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match=r"run the three Infrastructure cells again in order \(Sections 1, 2 and 3\)"):
        namespace["run_stage"]("prepare")
    troubleshooting = next(c["source"] for c in _cells() if c["source"].startswith("## Troubleshooting"))
    assert "run the three Infrastructure cells again in order (Sections 1, 2 and 3)" in troubleshooting


def test_stage_processes_import_neither_ipython_nor_google() -> None:
    """Stages run as `tutorial_stages.py` subprocesses in the isolated environment, which has neither IPython nor
    google.colab: only kernel cells use them (`IPython.display` in the install cell, the BYOD upload dialog). A carried
    module that imported either would fail on Colab; there is no worker and no google.colab stub to give a ModuleSpec."""
    import re

    carried = [ROOT / source for dest, source in TEMPLATE["carried"].items() if dest.endswith(".py")]
    assert any(path.name == "tutorial_stages.py" for path in carried)
    offenders = [str(path) for path in carried if re.search(r"^\s*(from|import)\s+(IPython|google)\b", path.read_text(encoding="utf-8"), re.M)]
    assert not offenders, offenders
    sources = "\n".join("".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"])
    stubs = ("sys.modules['google", 'sys.modules["google', "ModuleType('google", 'ModuleType("google')
    assert not [marker for marker in stubs if marker in sources]
