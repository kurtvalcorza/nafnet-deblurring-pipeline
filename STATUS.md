# Release status

Current status: **Candidate** — initial development. The `E2E` / `GUIDED` tutorial `tutorials/nafnet_deblurring_colab.ipynb` (DIMER Notebook Specification 2.2, standalone) and the package pass the offline test suite, `tools/validate_release_assets.py` and `tools/build_notebook.py --check`. The notebook runs its stages in an isolated hash-locked `uv` environment and installs nothing into the kernel, so `Run all` is designed to need no restart.

What has been executed and what has not is recorded in `docs/release-verification.md`. In short:

- **Executed (CPU, development container):** every infrastructure and learner cell of the committed notebook (commit `31db8bb`), top to bottom in one kernel, including the optional activity and the paired BYOD branch, plus the unpaired and refused BYOD archives with the runner of `227730f`. The isolated environment was replaced by an equivalent pinned environment and the checkpoint by a random-init full-width stand-in, because the container cannot reach `huggingface.co` and had too little free disk for the CUDA build of torch. This exercises the stage plumbing, validation and refusals, baselines, fine-tuning, artifact export, fresh reload and new-image inference; it says nothing about the pretrained model's quality. The lock was also resolved with the pinned `uv` 0.12.15 against a managed CPython 3.12.12 (`--dry-run --require-hashes`, 38 packages).
- **Executed (Google Colab T4, 2026-10-04):** a one-pass default `Run all` of blob `a628881` with the real checkpoint. The first Hugging Face mirror served the pinned bytes (`fetched_on_this_run: True`, SHA-256 verified); the conversion was exact; test PSNR / SSIM `pretrained` 26.13 dB / 0.7711 and `adapted` 26.36 dB / 0.7881 against `identity` 23.32 / 0.6264 and `wiener_oracle` 25.18 / 0.7134; reload parity `0.0`.
- **Executed (Google Colab T4, 2026-10-10):** a one-pass default `Run all` of the review-fix blob `3c99755` (commit `7b90ac9`, NAF-m1..m5) on a fresh session with the Colab CLI: 11/11 code cells, no error, no restart; every Section 3–8 number equals the 2026-10-04 run, including the converted digest `768444b5…2ef0`. Evidence in `docs/execution-evidence/2026-10-10-7b90ac9/`.
- **Not yet executed:** the hosted REL12 BYOD journey; the pinning of the converted safetensors digest recorded by that run.

Open items before promotion:

1. Pin the SHA-256 of the converted `model.safetensors` recorded by the Colab run (`768444b5…2ef0`, 68,510,388 bytes) and replace the timing estimates in the notebook prerequisites with the measured hosted timings. Both change the notebook, so the default `Run all` must then be repeated on the new blob.
2. Record the REL12 BYOD journey on a hosted runtime (one paired archive, one unpaired archive, one refused archive), on the same revision as item 1.
