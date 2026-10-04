# NAFNet Deblurring Pipeline

DIMER-oriented pipeline for **NAFNet-GoPro-width32** single-image deblurring (`megvii-research/NAFNet`, a 17.1 M-parameter activation-free U-shaped restoration network trained by its authors on the GoPro motion-blur dataset). The repository carries the upstream architecture verbatim and hash-verified, pins the 68.7 MB checkpoint by size and SHA-256, and exposes PSNR/SSIM evaluation against classical baselines, a paired blurred/sharp data contract with explicit ceilings, bounded restoration fine-tuning with a safetensors artifact, a `MODEL_CARD.md` at DIMER Model Card Specification 1.2, and a standalone `E2E` / `GUIDED` tutorial at DIMER Notebook Specification 2.2.

## Upstream alignment

- Model: `megvii-research/NAFNet`, variant `NAFNet-GoPro-width32`
- Upstream revision: `2b4af71ebe098a92a75910c233a3965a3e93ede4` (architecture source, GoPro test configuration and the README that lists the checkpoint)
- Checkpoint: `NAFNet-GoPro-width32.pth`, 68,671,121 bytes, SHA-256 `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`, fetched from a public Hugging Face mirror and refused unless both match; `docs/WEIGHTS.md` records the provenance chain and its residual trust gap
- Licence: **MIT** for NAFNet (code and weights); **Apache-2.0** for the BasicSR-derived parts, including the carried `arch_util.py`
- Upstream task: image restoration (deblurring) — blurred RGB image in, restored RGB image of the same size out
- Runtime: `torch==2.14.0`, `numpy==2.5.3`, `pillow==11.3.0`, `safetensors==0.8.0`, `scikit-image==0.26.0` (the last only as the carrier of the pinned sample photographs)
- Repository adaptation: **E2E** — bounded gradient fine-tuning of a parameter subset (default: the decoder half, 1,322,307 of 17,111,907 parameters), exported as a full safetensors state dict with a manifest

## Two things to know before you start

**The checkpoint is a pickle, read once.** Upstream publishes only a PyTorch `.pth`. `fetch_checkpoint` accepts the download only at the pinned size and SHA-256; `convert_checkpoint` reads it with `torch.load(..., weights_only=True)`, loads it strictly into the carried architecture, writes `model.safetensors`, and accepts the conversion only when the safetensors copy reproduces the outputs exactly. Every later load uses the safetensors file.

**A restoration metric needs a reference.** PSNR and SSIM compare an output with the sharp original, so the sample is made by degrading sharp public-domain photographs with seeded synthetic motion blur and noise. A real blurred photograph has no reference and gets no PSNR; the tutorial says so and reports only a labelled sharpness proxy for it.

## Quick start

```python
from pathlib import Path
from nafnet_deblurring_pipeline import DeblurPipeline, build_sample_dataset, fetch_checkpoint, convert_checkpoint
from nafnet_deblurring_pipeline import classical_outputs, score, tune_unsharp, tune_wiener

weights = Path("weights/nafnet-gopro-width32")
fetch_checkpoint(weights)                                              # pinned bytes only, from the mirrors
base = convert_checkpoint(weights / "NAFNet-GoPro-width32.pth", weights / "model.safetensors")
pipe = DeblurPipeline.from_base(weights / "model.safetensors", expected_sha256=base["sha256"])
splits = build_sample_dataset()                                        # 48 train / 12 test pairs, split by photograph
unsharp, wiener = tune_unsharp(splits["train"]), tune_wiener(splits["train"])
for name, outputs in classical_outputs(splits["test"], unsharp, wiener).items():
    print(name, score(outputs, splits["test"], method=name)["psnr"])
print("pretrained", score(pipe.restore([r["blurred"] for r in splits["test"]]), splits["test"], method="pretrained")["psnr"])
pipe.finetune(splits["train"], steps=300, lr=1e-4, batch_size=4, patch=256, scope="decoder", seed=0)
pipe.save_artifact("outputs/artifact", metadata={})
reloaded = DeblurPipeline.from_artifact("outputs/artifact")            # manifest and SHA-256 checked before loading
```

## Repository layout

```
src/nafnet_deblurring_pipeline/   pipeline.py (identity, checkpoint, architecture loader, fine-tuning, artifact)
                                  samples.py (degradation model, sample, validation, split, BYOD reader)
                                  metrics.py (PSNR, SSIM, unsharp / Wiener baselines, gradient-energy proxy)
third_party/nafnet/               NAFNet_arch.py, arch_util.py, local_arch.py and LICENSE, verbatim from upstream
weights/nafnet-gopro-width32/     dimer-base-manifest.json (the checkpoint itself is fetched at run time, git-ignored)
tools/                            build_notebook.py, notebook_template.py, tutorial_stages.py, validate_release_assets.py
tutorials/                        nafnet_deblurring_colab.ipynb (generated), requirements-colab.lock.txt, README.md
```

## Data contract

A paired record is `{id, source, blurred, sharp}`: two uint8 RGB arrays of identical size, each side 64..1024 px; `source` groups crops of one photograph so `split_by_source` and `dataset_manifest` can keep them on one side of the split. `validate_records` refuses mismatched sizes, out-of-range sides, non-RGB or non-uint8 arrays, duplicate ids, fewer than 4 or more than 200 pairs, and data from a single source, each with a message naming the record and the rule, before any model library is imported. `read_byod` reads a zip or directory with `sharp/` and `blurred/` folders matched by file stem (paired) or blurred images only (unpaired); it refuses traversing paths, symbolic links, oversize members, unmatched files and unreadable images, and reads zip members in memory without extracting them.

## Tests

```
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy==2.5.3 pillow==11.3.0 safetensors==0.8.0 scikit-image==0.26.0 pytest==8.4.2 ruff==0.16.6
pip install -e . --no-deps
pytest
python tools/validate_release_assets.py
python tools/build_notebook.py --check
```

The suite runs offline on a CPU: a tiny random-init NAFNet built from the carried upstream source, synthetic images, a random full-width stand-in for the conversion path, the generated notebook's own carrier and `run_stage` code, and a pre-flight of every tutorial stage. It never downloads the real checkpoint, so it is not evidence about the pretrained model's behaviour.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/tutorials/nafnet_deblurring_colab.ipynb)

`tutorials/nafnet_deblurring_colab.ipynb` is declared `E2E` / `GUIDED` and is **standalone** (DIMER Notebook Specification 2.2 §4). It is generated by `tools/build_notebook.py` from `tools/notebook_template.py` and carries, byte for byte and SHA-256-verified, the package's four modules, the three upstream architecture files and their licence, the stage runner `tools/tutorial_stages.py`, the hash-locked requirements `tutorials/requirements-colab.lock.txt`, the checkpoint manifest and the repository licence.

The notebook installs nothing into its own kernel. It downloads a pinned `uv` wheel (checked by size and SHA-256), builds an isolated CPython 3.12.12 environment with `uv venv --managed-python`, and installs the lock into it with `uv pip install --require-hashes --only-binary :all:`; Hugging Face tokens and the kernel's `PYTHONPATH` are removed from that environment and `MPLBACKEND=Agg` is set. Each stage (`weights`, `prepare`, `baseline`, `adapt`, `evaluate`, `reload`, then the optional `activity` and `byod`) runs there in its own process, hands results to the next only through files, and writes figures to files that the kernel displays. A hosted runtime's preloaded packages are never replaced, so `Run all` needs no restart. Only Linux x86_64 is supported, and other platforms are refused with a message. The lock is compiled from the `pyproject.toml` pins: `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --index-url https://pypi.org/simple --only-binary :all: -o tutorials/requirements-colab.lock.txt`; recompile it and regenerate the notebook whenever a pin changes. Every locked package is a wheel, so nothing is built from source.

## Release status

**Candidate** — initial development. The notebook and the package pass the offline suite and the static validator, and the stage chain has been executed on a CPU with a random-init stand-in checkpoint, and a fresh Google Colab T4 runtime completed `Run all` with the real checkpoint in one pass on 2026-10-04: the mirror served the pinned bytes, and on the 12-pair test split PSNR went from 23.32 dB (blurred input) to 26.13 dB (pretrained) and 26.36 dB (adapted). The hosted BYOD journey and the pin of the converted safetensors digest are still open. See `STATUS.md` and `docs/release-verification.md`.

## Licensing

- Upstream code and weights: MIT (`megvii-research/NAFNet`); BasicSR-derived parts Apache-2.0. Both texts are in `third_party/nafnet/LICENSE`.
- Tutorial data: photographs from the `scikit-image` 0.26.0 data directory, public domain or CC0 according to its documentation (licences per photograph in `samples.py`).
- This repository's code and documentation: Apache-2.0 (`LICENSE`).

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
