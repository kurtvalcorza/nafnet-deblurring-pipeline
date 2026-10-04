# Weight provenance, integrity and trust boundary

This page records where the NAFNet-GoPro-width32 weights come from, how the tutorial obtains and checks them, and what remains unverified. The machine-readable record is `weights/nafnet-gopro-width32/dimer-base-manifest.json`; the pinned values in `src/nafnet_deblurring_pipeline/pipeline.py` must agree with it, and `tests/test_pipeline.py` checks that they do.

## Identity

| Field | Value |
|---|---|
| Upstream model | `megvii-research/NAFNet`, variant `NAFNet-GoPro-width32` |
| Upstream revision | `2b4af71ebe098a92a75910c233a3965a3e93ede4` (the code, test configuration and README listing used here) |
| Checkpoint file | `NAFNet-GoPro-width32.pth` |
| Pinned size | 68,671,121 bytes |
| Pinned SHA-256 | `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c` |
| Parameters | 17,111,907 (checked when the state dict is loaded strictly) |
| Network configuration | `NAFNetLocal`, width 32, `enc_blk_nums [1, 1, 1, 28]`, `middle_blk_num 1`, `dec_blk_nums [1, 1, 1, 1]`, `train_size (1, 3, 256, 256)`, from `options/test/GoPro/NAFNet-width32.yml` |
| Licence | MIT (Copyright (c) 2022 megvii-model) for NAFNet; Apache-2.0 for the BasicSR-derived parts, including the carried `arch_util.py`. Both texts are in `third_party/nafnet/LICENSE`. |
| Publication date | 2022-04-16: commit `b4ad6514720a4b1784ef1719be827b22002c7ab0` adds the `NAFNet-GoPro-width32` row and its Google Drive link to the upstream README. The repository itself was created on 2022-04-10. |

## Provenance chain

1. **Authors' publication.** The upstream README at the pinned revision lists `NAFNet-GoPro-width32` with a Google Drive file (id `1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj`) and a Baidu Pan copy. The authors publish no checksum.
2. **Digest of the authors' file.** Two independent third-party projects pin the same SHA-256 for a download of that exact Google Drive file id: `Blankke/Image_manage` at commit `80fdeab1f4bd431498b6da730f6c9aacff1eebe1` (`scripts/install_optional_models.ps1`) and `uniplanck/Agent-2D` at commit `5bd23cfe5cd824f70ea76319448f0db5eeab7bf6` (`crates/agent2d-pipeline/src/restoration.rs`). Both record `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`. This repository adopts that value; it did not download the Drive file itself, because Google Drive is not reachable from the environment in which the repository was built.
3. **Mirror used by the tutorial.** Google Drive downloads of large files need an interactive confirmation and fail intermittently in hosted notebooks, which would break the credential-free `Run all` path. The tutorial therefore downloads the same file name from the public Hugging Face repository [`nyanko7/nafnet-models`](https://huggingface.co/nyanko7/nafnet-models) and, only if that fails, from [`mikestealth/nafnet-models`](https://huggingface.co/mikestealth/nafnet-models). Both repositories hold `NAFNet-GoPro-width32.pth` with an observed size of 68,671,121 bytes, equal to the pinned size. Neither mirror is published by the NAFNet authors. Third-party projects that use the first mirror include `senmei-app/senmei` (`models/metadata.json`), which records it as the source of the MIT NAFNet-GoPro width-32 weights.
4. **Verification in the tutorial.** The `weights` stage writes the download to a `.part` file, accepts it only if the size and SHA-256 equal the pinned values, and otherwise deletes it and tries the next mirror for the *same* pinned bytes. If no mirror delivers them the stage stops with `no mirror delivered the pinned checkpoint bytes (refusing any other file)`. There is no fallback to a different file, a different variant or an unverified download.

### Residual trust gap

- The pinned SHA-256 is a third-party statement about the authors' Google Drive file. It was not computed by this repository from the authors' bytes, and the authors do not publish one. Two independent projects agreeing on it is good evidence that it is the digest of the file behind that Drive link, not proof of authorship.
- The Hugging Face metadata available while this repository was built exposed the mirror file's size but not its LFS SHA-256 or the mirror's commit hash. The mirror is addressed through its `main` branch; its identity in this pipeline is the SHA-256, not the branch. The Google Colab run of 2026-10-04 downloaded the file from `nyanko7/nafnet-models` and recorded `fetched_on_this_run: True` with this digest, which confirms that the first mirror served the pinned bytes on that date (`docs/release-verification.md`); the second mirror has not been exercised. If it does not, the notebook refuses the file, which is the intended behaviour, and the digest must not be edited to make it pass.
- Matching bytes prove byte equality with the pinned file, not that the authors produced it (integrity, not authenticity).

## Trust boundary: a pickle, read once

`.pth` is a PyTorch pickle, which can execute code when it is loaded. The pipeline treats it as trusted executable serialisation:

1. the bytes are verified against the pinned size and SHA-256 **before** `torch.load` sees them;
2. `torch.load(path, map_location="cpu", weights_only=True)` restricts unpickling to tensors and plain containers;
3. the payload must be a dict with a `params` state dict of tensors only, which must load strictly into the carried architecture with exactly 17,111,907 parameters;
4. it is converted once to safetensors (`model.safetensors`, sorted keys, float32), reloaded from that file into a second network, and both networks are compared on a seeded input; the conversion is accepted only when the outputs are identical;
5. every later stage loads the safetensors file after checking the SHA-256 the `weights` stage recorded. The exported artifact is safetensors as well.

The SHA-256 of the converted `model.safetensors` is **not pinned** yet. Each run records it in `outputs/weights.json`; it is to be pinned once a hosted run has recorded it.

## Architecture source

The architecture is not installed from a package index: NAFNet is not a maintained PyPI package. The three files the network needs are carried verbatim from the upstream revision under `third_party/nafnet/basicsr/models/archs/` and are checked against these digests before they are executed (`load_architecture`):

| File | Bytes | SHA-256 | Git blob |
|---|---|---|---|
| `NAFNet_arch.py` | 6,396 | `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967` | `5735e0963b4b1db46f34807e6607c04e70702e91` |
| `arch_util.py` | 12,046 | `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005` | `09beabfb4fb14e5323dc2a6a4234bddbd43138c2` |
| `local_arch.py` | 4,399 | `c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26` | `bc459c64a200b9af2870cc1f405fc1df17c6a83a` |

The upstream `LICENSE` is carried beside them (SHA-256 `a29ecef3456149898f08e4c71b11b33e7d333664e087bc212e84e18ddd6599ad`). The files import `basicsr.*` by absolute name. `load_architecture` registers empty `basicsr`, `basicsr.models` and `basicsr.models.archs` package modules and a one-function `basicsr.utils` module whose `get_root_logger` is `logging.getLogger` (used only by an unused helper in `arch_util.py`), so none of upstream's training framework, OpenCV or LMDB dependencies is needed. No other upstream file is executed.

## Hosting and redistribution

MIT and Apache-2.0 both permit redistribution with the licence and notices preserved. This Git repository does not vendor the checkpoint (`weights/**/*.pth` and `weights/**/*.safetensors` are git-ignored); it carries the manifest only. A mirror of the pinned bytes may be hosted under the upstream licence, provided the provenance above is kept with it.
