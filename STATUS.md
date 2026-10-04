# Release status

Current status: **Candidate** — initial development. The `E2E` / `GUIDED` tutorial `tutorials/nafnet_deblurring_colab.ipynb` (DIMER Notebook Specification 2.2, standalone) and the package pass the offline test suite, `tools/validate_release_assets.py` and `tools/build_notebook.py --check`. The notebook runs its stages in an isolated hash-locked `uv` environment and installs nothing into the kernel, so `Run all` is designed to need no restart.

What has been executed and what has not is recorded in `docs/release-verification.md`. In short:

- **Executed (CPU, development container):** every infrastructure and learner cell of the committed notebook (commit `31db8bb`), top to bottom in one kernel, including the optional activity and the paired BYOD branch, plus the unpaired and refused BYOD archives with the runner of `227730f`. The isolated environment was replaced by an equivalent pinned environment and the checkpoint by a random-init full-width stand-in, because the container cannot reach `huggingface.co` and had too little free disk for the CUDA build of torch. This exercises the stage plumbing, validation and refusals, baselines, fine-tuning, artifact export, fresh reload and new-image inference; it says nothing about the pretrained model's quality. The lock was also resolved with the pinned `uv` 0.12.15 against a managed CPython 3.12.12 (`--dry-run --require-hashes`, 38 packages).
- **Not yet executed:** a hosted `Run all` (Colab or Kaggle T4) of the current revision with the real checkpoint; the confirmation that the Hugging Face mirror serves bytes matching the pinned SHA-256; the hosted REL12 BYOD journey; the pinning of the converted safetensors digest.

Open items before promotion:

1. Record one-pass hosted `Run all` evidence for the current revision (RUN1, REL1–REL7), including `fetched_on_this_run: True` with the pinned digest in Section 3.
2. Pin the SHA-256 of the converted `model.safetensors` recorded by that run.
3. Record the REL12 BYOD journey on a hosted runtime (one paired archive, one unpaired archive, one refused archive).
4. Replace the timing estimates in the notebook prerequisites with measured hosted timings.
