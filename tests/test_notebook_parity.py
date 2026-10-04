"""NOTEBOOK_SPEC 2.2 parity tests (PAR1–PAR3) for the standalone tutorial notebook (isolated-environment carrier).

The notebook carries the repository's package, the stage runner, the hash-locked requirements, the snapshot manifests
and the licence as text in one carrier cell; these tests fail whenever a carried file, its recorded SHA-256, the lock or
the generated notebook diverges from the repository at HEAD.
"""
# ruff: noqa: E501

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

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
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
LOCK = ROOT / TEMPLATE["carried"][TEMPLATE["lock"]]


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _carrier(notebook: dict) -> tuple[dict, dict[str, str], dict[str, str]]:
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c.get("metadata", {}).get("dimer", {}).get("embedded_sources")]
    assert len(cells) == 1, "exactly one carrier cell is expected"
    source = _source(cells[0])
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("CARRIED_FILES", "CARRIED_HASHES"):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return cells[0], values["CARRIED_FILES"], values["CARRIED_HASHES"]


def test_par1_carried_files_equal_repository_sources(notebook: dict) -> None:
    """Every carried file is the repository file byte for byte (UTF-8 text, LF newlines)."""
    _cell, files, _hashes = _carrier(notebook)
    assert list(files) == [*TEMPLATE["carried"], build.SOURCE_RECORD]
    for dest, source in TEMPLATE["carried"].items():
        assert files[dest] == (ROOT / source).read_text(encoding="utf-8"), f"carried {dest} drifted from {source}; regenerate the notebook"


def test_par1_carried_hashes_are_correct_and_recorded(notebook: dict) -> None:
    cell, files, hashes = _carrier(notebook)
    assert set(hashes) == set(files)
    for dest, text in files.items():
        assert hashes[dest] == hashlib.sha256(text.encode("utf-8")).hexdigest(), dest
    assert cell["metadata"]["dimer"]["files"] == hashes
    assert notebook["metadata"]["dimer"]["generated_from"]["files"] == hashes
    record = json.loads(files[build.SOURCE_RECORD])
    assert record["revision"] == notebook["metadata"]["dimer"]["generated_from"]["revision"]
    assert record["files"] == {d: h for d, h in hashes.items() if d != build.SOURCE_RECORD}


def test_par2_carried_lock_is_the_committed_lock_and_pins_pyproject(notebook: dict) -> None:
    _cell, files, _hashes = _carrier(notebook)
    lock_text = LOCK.read_text(encoding="utf-8")
    assert files[TEMPLATE["lock"]] == lock_text
    build.check_lock(build._pins(ROOT), lock_text)  # every direct pin at its version, every entry hashed
    locked = build.lock_packages(lock_text)
    assert locked["torch"] == "2.14.0" and locked["scikit-image"] == "0.26.0" and "cuda-bindings" in locked
    assert "--only-binary :all:" in lock_text.splitlines()[1], "the lock must be compiled wheel-only"
    assert "--python-platform x86_64-manylinux" in lock_text.splitlines()[1], "the lock must target manylinux x86_64"


def test_par2_carried_manifests_are_the_committed_manifests(notebook: dict) -> None:
    _cell, files, _hashes = _carrier(notebook)
    manifests = [d for d in files if d.startswith("weights/")]
    assert len(manifests) == 1
    for dest in manifests:
        assert json.loads(files[dest]) == json.loads((ROOT / dest).read_text(encoding="utf-8"))
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, TEMPLATE)["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "git" not in re.findall(r"subprocess\.run\(\[([^\]]*)\]", code).__str__()
    kernel = "\n".join(_source(c) for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources"))
    assert "github.com" not in kernel
    # carried files may name the upstream repository as provenance (MODEL_ASSET_SPEC SRC1) but never fetch from it
    assert not re.search(r"git\s+clone|urlopen\([^)]*github|raw\.githubusercontent", code)
    assert TEMPLATE["package"] not in kernel, "the kernel must not import the package; stages import the carried copy"


def test_carried_upstream_files_are_verbatim_and_hash_pinned(notebook: dict) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from nafnet_deblurring_pipeline.pipeline import UPSTREAM_ARCH_SHA256

    _cell, files, _hashes = _carrier(notebook)
    for rel, digest in UPSTREAM_ARCH_SHA256.items():
        text = files[f"third_party/nafnet/{rel}"]
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, rel


def test_no_pip_install_into_the_kernel(notebook: dict) -> None:
    """RUN10/ENV6: nothing is installed into the notebook kernel; the only installer is uv into the isolated env."""
    code = [c for c in notebook["cells"] if c["cell_type"] == "code" and not c["metadata"].get("dimer", {}).get("embedded_sources")]
    for cell in code:
        source = _source(cell)
        assert "pip install" not in source and "'-m', 'pip'" not in source and "sys.executable" not in source
        assert not re.search(r"(?m)^\s*[!%]", source), "no shell escapes or magics"
    install = [_source(c) for c in code if _source(c).startswith("# @title Infrastructure: install the locked runtime")]
    assert len(install) == 1
    assert "'--require-hashes', '--only-binary', ':all:'" in install[0] and "'--managed-python', '--python', '3.12.12'" in install[0]
    assert "MPLBACKEND='Agg'" in install[0] and "'HF_TOKEN'" in install[0]
    assert "Restart the runtime" not in json.dumps(notebook)
