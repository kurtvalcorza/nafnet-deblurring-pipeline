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

### 2026-10-04 — Google Colab T4, default `Run all` with the real checkpoint

- **Date:** 2026-10-04
- **Subject:** `tutorials/nafnet_deblurring_colab.ipynb`, blob `a62888122407ee450e0256db60658e8161ee68b9` (branch `ccr-24656dfc-ax1ln2` at `886e6a1`; carried files of `31db8bb`). Every source cell of the executed copy is byte-identical to that blob. The executed copy is `docs/execution-evidence/2026-10-04/nafnet_deblurring_colab_a628881_colab-t4.ipynb`, summarised in `colab_t4_run_summary.json` beside it.
- **Runtime:** fresh Google Colab runtime, Tesla T4 (15,360 MiB), kernel CPython 3.13.15. Stage environment built by the notebook: CPython 3.12.12 managed by `uv`, the 38-package lock, `torch 2.14.0+cu130` with CUDA, `numpy 2.5.3`, `safetensors 0.8.0`; built in 66 s.
- **Procedure:** `Run all` with no field edited (`STEPS = 300`, `RUN_ACTIVITY = False`, `USE_BYOD = False`).
- **Observed result:** all 11 code cells completed in one pass, in execution order 1–11, with no error, restart, credential or upload dialog.
  - Section 3: the checkpoint was downloaded from `https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth` with `fetched_on_this_run: True` and `checkpoint_verified: True` at the pinned 68,671,121 bytes and SHA-256 `19394e61…1a5c`. This confirms that the first mirror serves the pinned bytes. Converted `model.safetensors`: 68,510,388 bytes, SHA-256 `768444b5d1dde023b21564a541e3aeeac1c7c084ce9b91a8c0408709b05c2ef0`, 664 tensors, 17,111,907 parameters, `max_abs_output_diff_vs_pth: 0.0` (13.5 s).
  - Section 4: 48 training / 12 test pairs, disjoint photographs; all five refusal probes rejected (6.2 s).
  - Section 5: test PSNR / SSIM — `identity` 23.32 dB / 0.6264, `unsharp` 23.37 / 0.6186, `wiener_oracle` 25.18 / 0.7134, `pretrained` 26.13 / 0.7711; pretrained beat the blurred input by +2.81 dB on average (min −0.24, max +4.48), on 11 of 12 pairs (22.3 s).
  - Section 6: `decoder` scope, 1,322,307 trainable / 15,789,600 frozen parameters, 300 steps on CUDA in 54.9 s (about 0.18 s per step); training-batch PSNR 25.66 dB at step 25 and 27.20 dB at step 300; artifact weights SHA-256 `8d849baf…e969` (70.8 s for the stage).
  - Section 7: `adapted` 26.36 dB / 0.7881; `adapted_vs_pretrained` +0.23 dB mean (min −0.17, max +0.42), 11 of 12 pairs improved; `adapted_vs_wiener_oracle` +1.18 dB; per test photograph, `astronaut` 25.69 → 26.02, `gravel` 23.59 → 23.81, `rocket` 29.11 → 29.25 dB (12.3 s).
  - Section 8: artifact manifest verified before loading; `reload_parity` `max_abs_float_diff 0.0`, `uint8_equal_fraction 1.0`; `text-synthetic` PSNR input 25.35, pretrained 30.91, adapted 31.29 dB; `clock-real-motion` gradient-energy ratio 1.081 pretrained, 0.807 adapted (no reference, no score) (11.3 s).
- **Caveats:** one seeded run on one 12-pair test split, so the +0.23 dB adaptation gain is small and may not survive another seed or other photographs. The optional activity and the BYOD branch were not run, so the hosted REL12 BYOD journey is still open. GPU results are not expected to equal CPU results exactly.


### 2026-10-04 — local CPU execution of the committed notebook at `31db8bb` (not a hosted Run all)

- **Date:** 2026-10-04
- **Subject:** `tutorials/nafnet_deblurring_colab.ipynb` at commit `31db8bb`, blob `9fd1ab250e96c97c89ba8cfce2df8b452627b587` (`source.json` revision label `d6e1d0b52bfe2ed92caae8cea6dbb5adcf0e0e73`, the parent commit; the carried files are those of `31db8bb`).
- **Runtime:** development container, CPU only (4 cores, shared), no GPU. Kernel: CPython 3.12.12 with IPython 9.17.1. Stage environment: CPython 3.12.12 with `torch 2.14.0+cu130`, `numpy 2.5.3`, `pillow 11.3.0`, `safetensors 0.8.0`, `scikit-image 0.26.0` (the lock's pins; torch read from an existing local installation of the same wheel because of limited disk).
- **Procedure:** all 11 code cells executed in order in one kernel namespace by a driver that sets form fields in a copy of each cell (EXE6): `STEPS = 60`, `RUN_ACTIVITY = True`, `ACTIVITY_SCOPE = 'decoder+middle'`, `USE_BYOD = True`, `BYOD_PATH` = a local paired archive, `BYOD_STEPS = 10`; every other field at its default. Forced deviations from a hosted run: (a) the install cell's `uv` download, `uv venv` and `uv pip install` lines were replaced by a pointer to the stage environment above (its `ENV`, version probe and `run_stage` ran verbatim); (b) the Section 1 disk requirement for the isolated environment was set to zero; (c) `huggingface.co` is unreachable, so a random-init full-width NAFNet in the upstream layout (`{'params': state_dict}`, 68,681,301 bytes) was staged, and the carried copy of `pipeline.py` and the carried manifest in the run directory were rewritten to pin its size and SHA-256 (`74189d84…`). The BYOD archive held five 256-px crops of other `scikit-image` photographs (`cell`, `coins`, `hubble_deep_field`, `moon`, `retina`) with seeded synthetic motion blur; it is not part of the repository.
- **Observed result:** all 11 code cells completed in one pass with no restart; cell times 0.0, 0.0, 2.3, 4.5, 6.5, 24.5, 144.0, 17.9, 14.1, 147.8 and 30.1 s.
  - Sections 1–3: `accelerator: none (CPU only)`; 13 carried files verified; stage environment Python 3.12.12, `cuda: False`, 38 locked packages; stand-in checkpoint verified, `fetched_on_this_run: False`; converted `model.safetensors` 664 tensors, 17,111,907 parameters, `max_abs_output_diff_vs_pth: 0.0`.
  - Section 4: 48 training / 12 test pairs; training photographs `brick`, `camera`, `chelsea`, `coffee`, `grass`, `immunohistochemistry`; test photographs `astronaut`, `gravel`, `rocket`; `disjoint_sources: True`; blur lengths 9..21 px; all five refusal probes rejected, each naming its rule (for example `blurred is 256x256 px but sharp is 256x248 px; both images of a pair must have identical size`).
  - Section 5: settings chosen on train — unsharp `sigma 3.0, amount 0.25`, Wiener `nsr 0.03`; test PSNR / SSIM — `identity` 23.32 dB / 0.6264, `unsharp` 23.37 / 0.6186, `wiener_oracle` 25.18 / 0.7134, stand-in `pretrained` 18.70 / 0.3246.
  - Section 6: `decoder` scope, 1,322,307 trainable / 15,789,600 frozen / 17,111,907 total parameters, 256-px crops; training-batch PSNR 20.40 dB at step 25 and 22.05 dB at step 60; 132.7 s of training (2.2 s per step).
  - Section 7 (fresh process, artifact verified first): `adapted` 22.70 dB / 0.5744; `adapted_vs_pretrained` +4.00 dB mean (min +2.12, max +6.05), 12 of 12 pairs improved; `adapted_vs_identity` −0.63 dB, 0 of 12 improved.
  - Section 8 (second fresh process): `reload_parity` `max_abs_float_diff 0.0`, `uint8_equal_fraction 1.0` (tolerance `1e-05`); `text-synthetic` PSNR blurred input 25.35 dB, stand-in pretrained 19.57, adapted 25.05; `clock-real-motion` gradient-energy ratios only.
  - Section 9 (activity, `decoder+middle`, 3,174,211 trainable parameters): 22.70 dB / 0.5753 against 22.70 dB / 0.5744 for the default scope; first logged losses −20.407 and −20.3995, showing that both runs started from the same weights.
  - Section 10 (paired BYOD): 5 pairs validated, split 4 / 1 by image (test image `hubble_deep_field`); unsharp chosen on train `sigma 1.0, amount 0.25`; no oracle Wiener (kernels unknown); test PSNR `identity` 26.05 dB, `unsharp` 26.01, stand-in `pretrained` 23.36, `adapted` 24.30 after 10 steps; reload parity `0.0`; results, metrics CSV, restored image and artifact written.
- **Caveats:** the `pretrained` and `adapted` numbers come from random weights and carry no information about NAFNet's quality; the run exercises the stage chain, validation and refusals, the classical baselines, bounded fine-tuning, artifact export, fresh-process reload and equivalence, new-image inference, the optional activity and the paired BYOD branch. The real-checkpoint path (mirror download, digest match, conversion of the real bytes) and the `uv` environment build were not executed. Not REL1/REL2 evidence.

### 2026-10-04 — BYOD branches with the runner of `227730f`

- **Date:** 2026-10-04
- **Subject:** `tools/tutorial_stages.py` at commit `227730f` (the per-archive BYOD output directories), run against the run directory of the record above
- **Runtime:** as above
- **Procedure:** `tutorial_stages.py --stage byod` (the command `run_stage('byod', …)` issues) with three archives in sequence: the paired archive above with `--steps 10`, an unpaired archive (the real `clock_motion` photograph and one blurred crop), and a refused archive (`sharp/a.png` with `blurred/b.png`).
- **Observed result:** paired — as above, written to `outputs/byod/paired-d78e02364f3e/`; unpaired — 2 images validated and deblurred by the stand-in pretrained model and the adapted sample artifact, gradient-energy ratios reported and no PSNR/SSIM, written to `outputs/byod/unpaired-c3bd7fe358f7/` while the paired directory remained; refused — the stage stopped with `BYOD: files without a partner (pairs are matched by file stem in sharp/ and blurred/): ['sharp/a.png', 'blurred/b.png']` before any model ran.
- **Caveats:** stand-in weights; local CPU only; the Colab upload dialog (`BYOD_PATH` empty) was not exercised.

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
