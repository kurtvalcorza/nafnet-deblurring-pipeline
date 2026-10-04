# Release verification

`tutorials/nafnet_deblurring_colab.ipynb` (`E2E` / `GUIDED`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top to bottom in a clean supported runtime with the real checkpoint. Unit tests, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record.

## Automatic coverage (static and offline, every pull request)

CI (`.github/workflows/ci.yml`) installs CPU torch and the other pinned libraries, then runs:

- `ruff check src tests tools`;
- `pytest`, which runs offline and never downloads the real checkpoint:
  - `tests/test_samples.py` — motion kernels, the seeded degradation, the pinned sample (48 / 12 pairs, split by photograph, deterministic digest), `validate_records` refusals, `split_by_source` and leakage detection, and `read_byod` for paired and unpaired archives and directories and for traversal, symlink, unmatched, size-mismatch, oversize and unreadable inputs;
  - `tests/test_metrics.py` — PSNR against its definition, SSIM against a brute-force window, Wiener with the true kernel, train-only tuning, paired differences, the gradient-energy proxy;
  - `tests/test_import_boundary.py` — importing the package and refusing bad input never import torch or safetensors;
  - `tests/test_pipeline.py` — the carried upstream files match their upstream digests and a tampered copy is refused; the full architecture has 17,111,907 parameters and the documented per-scope trainable counts; `NAFNetLocal` equals global pooling on small inputs; a tiny random-init network restores odd sizes, fine-tunes only its scope and is reproducible; the artifact round-trips within `1e-6` and refuses extra files, changed bytes and another identity; `fetch_checkpoint` refuses wrong bytes, falls through to the second mirror and leaves no partial file; the manifest identity check; the `.pth` → safetensors conversion on a random full-width stand-in is exact and byte-deterministic;
  - `tests/test_notebook_parity.py` — every carried file equals the repository file, every hash is correct, the carried lock is the committed lock and pins every `pyproject.toml` pin with hashes for manylinux x86_64, the carried upstream files hash to their upstream digests, no kernel cell pip-installs, uses `sys.executable` or a shell escape, the install cell uses `--managed-python --python 3.12.12` and `--require-hashes --only-binary :all:`, strips tokens and sets `MPLBACKEND='Agg'`, and the notebook is byte-identical to the generator output;
  - `tests/test_tutorial_stages.py` — the generated carrier writes and verifies every file; `run_stage` re-raises a stage's own message; and a CPU pre-flight runs `weights` → `prepare` → `baseline` → `adapt` → `evaluate` → `reload` → `activity` → `byod` (paired, unpaired, refused) in order against a tiny random-init network and synthetic data, checking every expected output file;
- `tools/validate_release_assets.py` — model-card structure and front matter, identity consistency, weight facts against the manifest, release-status agreement, notebook structure, carrier, lock, isolated install, stage order, form fields, forbidden patterns, the guided layer and the infrastructure collapse;
- `tools/build_notebook.py --check`.

None of this is execution evidence for the real checkpoint (REL8).

## Manual coverage required before promotion (REL9, REL10)

1. Open the notebook from `main` in a fresh Colab (or Kaggle) T4 runtime and choose **Run all** with no field edited.
2. Record: notebook revision (commit and blob), runtime, the Section 3 output (`fetched_on_this_run: True`, the mirror URL, the converted `model.safetensors` SHA-256), the Section 5 and 7 tables, the reload check, the stage timings, and whether any cell needed a restart (it must not).
3. Run the REL12 BYOD journey on the same runtime: one paired archive (accepted, reaches adapt/evaluate/export/reload), one unpaired archive (accepted, inference only), one refused archive (a file without a partner).
4. Pin the converted `model.safetensors` digest and replace the timing estimates in the notebook prerequisites with the measured values.

## Recorded executions

### 2026-10-04 — local CPU execution of a pre-commit build of the notebook (not a hosted Run all)

- **Date:** 2026-10-04
- **Subject:** `tutorials/nafnet_deblurring_colab.ipynb` generated from the uncommitted working tree before the first commit (`source.json` revision label `22dc27297022cc1a093b49a1413ac86229d6c19f`, the empty initial commit). The carried upstream files, lock and manifest are identical to the first commit; the carried package and stage runner later received small changes (a deterministic safetensors metadata layout, a size-independent refusal probe, lint fixes, one added prediction prompt), so this record does not describe the committed blob.
- **Runtime:** development container, CPU only (4 cores shared with unrelated processes, load average 15–20), no GPU; kernel: CPython 3.12 with IPython 9.17.1; stage environment: a CPython 3.12 virtual environment with `torch 2.14.0`, `numpy 2.5.3`, `pillow 11.3.0`, `safetensors 0.8.0`, `scikit-image 0.26.0` (the lock's pins).
- **Procedure:** every code cell of the notebook executed in order in one kernel namespace by a small driver that sets form fields in a copy of the cell (EXE6). Deviations from a hosted run, all forced by the container: (a) the install cell's `uv` download, `uv venv` and `uv pip install` lines were replaced by a pointer to the pre-built virtual environment, because the container had about 5 GB of free disk and the CUDA build of torch needs more; its `ENV`, probe and `run_stage` definitions ran verbatim; (b) the Section 1 disk requirement for the isolated environment was set to zero for the same reason; (c) `huggingface.co` is not reachable from the container, so a random-init full-width NAFNet saved in the upstream layout (`{'params': state_dict}`) was staged as the checkpoint, and the carried copy of `pipeline.py` and the carried manifest in the run directory were rewritten to pin that file's size and SHA-256; (d) `STEPS = 60` instead of 300.
- **Observed result:** all 11 code cells completed in one pass with no restart (cell times 0.0, 0.0, 2.1, 23.9, 6.1, 46.3, 956.3, 16.5, 14.3, 0.0, 0.0 s; the fine-tuning time reflects a heavily shared CPU).
  - Section 1: `accelerator: none (CPU only)`; Section 2: 13 carried files verified; stage environment Python 3.12.12, `torch 2.14.0+cu130`, `cuda: False`, 38 locked packages.
  - Section 3 (stand-in checkpoint): size and SHA-256 verified, `fetched_on_this_run: False`; converted `model.safetensors` 664 tensors, 17,111,907 parameters, `max_abs_output_diff_vs_pth: 0.0`.
  - Section 4: 48 training / 12 test pairs, training photographs `brick`, `camera`, `chelsea`, `coffee`, `grass`, `immunohistochemistry`, test photographs `astronaut`, `gravel`, `rocket`, `disjoint_sources: True`, blur lengths 9..21 px; all five refusal probes rejected with the rule named.
  - Section 5: unsharp settings chosen on train `sigma 3.0, amount 0.25`, Wiener `nsr 0.03`; test PSNR/SSIM — `identity` 23.32 dB / 0.6264, `unsharp` 23.37 / 0.6186, `wiener_oracle` 25.18 / 0.7134, stand-in `pretrained` 18.70 / 0.3246.
  - Section 6 (`STEPS = 60`, `decoder` scope, 1,322,307 trainable parameters): training-batch PSNR 20.40 dB at step 25, 22.45 at step 50, 22.05 at step 60.
  - Section 7: `adapted` 22.70 dB / 0.5744; `adapted_vs_pretrained` +4.00 dB mean (min +2.12, max +6.05), 12 of 12 pairs improved.
  - Section 8: artifact manifest verified before loading; `reload_parity` `max_abs_float_diff 0.0`, `uint8_equal_fraction 1.0`; `text-synthetic` PSNR input 25.35 dB, stand-in pretrained 19.57, adapted 25.05; `clock-real-motion` gradient-energy ratios only (no metric).
  - Sections 9 and 10 at their defaults printed the skip messages.
- **Caveats:** with random weights, the pretrained and adapted PSNR/SSIM say nothing about NAFNet's quality; the run exercises the stage chain, the validator and its refusals, the classical baselines, bounded fine-tuning, artifact export, fresh-process reload and equivalence, and new-image inference. The real-checkpoint path (download, digest match, conversion of the real bytes) and the `uv` environment build were not executed. Not REL1/REL2 evidence.

### `uv` lock dry run

- **Date:** 2026-10-04
- **Subject:** `tutorials/requirements-colab.lock.txt` at the first commit
- **Runtime:** development container, `uv` 0.8.17
- **Procedure:** the notebook's own `uv` steps outside a notebook: the `uv` 0.12.15 wheel was checked against the pinned size (20,081,404) and SHA-256 and unpacked, `uv venv --managed-python --python 3.12.12` created an environment, and `uv pip install --dry-run --require-hashes --only-binary :all: --index-url https://pypi.org/simple -r tutorials/requirements-colab.lock.txt` was run against it.
- **Observed result:** the managed interpreter reported Python 3.12.12; `uv` resolved all 38 locked packages from PyPI with hash checking and reported `Would download 38 packages`, including `torch==2.14.0`, `scikit-image==0.26.0` and the CUDA 13 runtime wheels.
- **Caveats:** a dry run resolves and checks the lock against the index; it does not download or install the wheels.
