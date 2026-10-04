"""Import-boundary contract (fleet RTM-001): importing the package, validating data and refusing a bad request never
import torch or safetensors; every model-library import is inside a function body."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import MODEL_LIBRARIES, synthetic_pairs

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "nafnet_deblurring_pipeline"


def test_package_import_does_not_import_model_libraries(forbid_model_imports):
    import importlib

    import nafnet_deblurring_pipeline

    importlib.reload(nafnet_deblurring_pipeline)


def test_no_module_level_model_imports():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.partition(".")[0] not in MODEL_LIBRARIES, f"{path.name} imports {name} at module level"


def test_invalid_inputs_are_rejected_before_model_imports(forbid_model_imports, tmp_path):
    from nafnet_deblurring_pipeline import read_byod, validate_records

    with pytest.raises(ValueError, match="4..200 are required"):
        validate_records(synthetic_pairs(1, 1))
    (tmp_path / "x.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # an empty zip
    with pytest.raises(ValueError, match="no images found"):
        read_byod(tmp_path / "x.zip")
