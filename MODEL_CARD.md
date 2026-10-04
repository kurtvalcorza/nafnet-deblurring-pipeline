---
license: mit
model_card_spec: "1.2"
pipeline_tag: image-to-image
task: "Image Restoration — single-image deblurring (NAFNet, GoPro-trained, width 32)"
base_model: megvii-research/NAFNet
base_model_variant: NAFNet-GoPro-width32
date_published: "2022-04-16"
date_published_source: "upstream commit `b4ad6514720a4b1784ef1719be827b22002c7ab0` (2022-04-16) adds the NAFNet-GoPro-width32 row and its Google Drive link to the upstream README; the repository was created on 2022-04-10"
---

# NAFNet-GoPro-width32 (megvii-research/NAFNet at 2b4af71) — Image Deblurring and Restoration Fine-Tuning

[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-megvii--research%2FNAFNet-181717?style=flat&logo=github&logoColor=white)](https://github.com/megvii-research/NAFNet)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2204.04676-b31b1b.svg)](https://arxiv.org/abs/2204.04676)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/megvii-research/NAFNet/blob/main/LICENSE)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** The upstream weights are fetched unmodified under their MIT licence and converted to safetensors at run time; the accompanying code and notebook are Apache-2.0. Nothing here is validated for production, and no benchmark result is claimed. The tutorial is a release candidate: no hosted run with the real checkpoint has been recorded yet.

> [!IMPORTANT]
> **The checkpoint is a pickle, and a restoration metric needs a reference.** Upstream publishes only `NAFNet-GoPro-width32.pth`. `src/nafnet_deblurring_pipeline/pipeline.py` accepts it only at the pinned size and SHA-256, reads it once with `torch.load(..., weights_only=True)` and converts it to safetensors with an exact output check. PSNR and SSIM are computed only where a sharp reference exists, which is why the sample degrades sharp photographs with known synthetic blur.

---

## Interactive Colab Tutorials

This pipeline provides a self-contained Google Colab notebook. It carries the repository's code and the upstream architecture source in its own cells and runs end to end without cloning the repository. It installs nothing into the notebook kernel: every stage runs in an isolated environment built from a committed hash lock, so `Run all` needs no runtime restart:

- **End-to-End Deblurring and Restoration Fine-Tuning Tutorial** (`E2E`, `GUIDED`):  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/tutorials/nafnet_deblurring_colab.ipynb) [`nafnet_deblurring_colab.ipynb`](https://github.com/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/tutorials/nafnet_deblurring_colab.ipynb)  
  *A digest-pinned checkpoint converted to safetensors; 60 synthetic motion-blur pairs from public-domain and CC0 photographs, validated and split by photograph; identity, unsharp-mask and oracle Wiener baselines and the pretrained network scored with PSNR and SSIM; bounded fine-tuning of the decoder half; a paired held-out comparison of the exported artifact; a fresh reload with an output-equivalence check; inference on a synthetic and a real blurred image; and a bring-your-own-data branch for paired or unpaired images.*

---

#### Description

`megvii-research/NAFNet` at revision `2b4af71ebe098a92a75910c233a3965a3e93ede4` is the reference implementation of NAFNet, the Nonlinear Activation Free Network for image restoration (Chen et al., 2022). This pipeline packages the authors' `NAFNet-GoPro-width32` checkpoint: a width-32 network with 17,111,907 parameters, trained by the authors on the GoPro dynamic-scene deblurring dataset (Nah et al., 2017).

NAFNet is a U-shaped convolutional network. An intro convolution feeds four encoder levels (1, 1, 1 and 28 blocks), one middle block and four decoder levels (one block each) joined by skip connections, and an ending convolution. Each block uses layer normalisation, a depthwise convolution, a *SimpleGate* that multiplies one half of the channels by the other in place of an activation function, and a simplified channel attention. The network predicts a residual that is added to the blurred input, so the output is a restored RGB image of the same size. At evaluation and inference the pipeline uses `NAFNetLocal` with `train_size (1, 3, 256, 256)`, as the upstream GoPro test configuration does; it replaces global average pooling with pooling over a window derived from the training size.

The upstream weights do not change at inference. Adaptation in this repository is ordinary gradient fine-tuning of a parameter subset. The default `decoder` scope trains the upsampling layers, the decoder blocks and the ending convolution, 1,322,307 of 17,111,907 parameters; `decoder+middle` and `all` are selectable. No adapter or PEFT method is used.

This repository adds, as code a reader runs:

- the three upstream architecture files, carried verbatim and checked against their SHA-256 before they are executed (`load_architecture`);
- checkpoint acquisition from a public mirror with size and SHA-256 verification (`fetch_checkpoint`), and a one-time conversion to safetensors (`convert_checkpoint`);
- a synthetic degradation model, a pinned sample, input validation and a split by photograph (`samples.py`);
- PSNR, SSIM, unsharp-mask and Wiener baselines (`metrics.py`);
- bounded fine-tuning and a safetensors artifact whose manifest is verified before loading (`DeblurPipeline.finetune`, `save_artifact`, `from_artifact`);
- the standalone tutorial notebook and its stage runner.

#### Intended Use and Limitations

The pipeline was built to teach and test how a pretrained restoration network is evaluated and adapted, not to serve as a production deblurring component.

###### Primary Intended Uses

The task is single-image deblurring. The input is one blurred RGB image as a uint8 array of shape H × W × 3, with each side 64..1024 px. The output is a restored uint8 RGB image of the same shape. With paired data (a blurred image and its sharp original) the pipeline also scores the output with PSNR and SSIM and can fine-tune the network.

Application domains envisioned during development:

- teaching and self-study of image restoration evaluation: reference-based metrics, classical baselines, held-out splits by scene, and why a real blurred photograph has no metric;
- research prototypes that need a pretrained motion-deblurring baseline to compare against, on photographs of everyday scenes similar to GoPro's handheld video frames;
- adapting the network to a specific, known degradation for which the user can produce paired data, such as a fixed camera shake pattern simulated on their own sharp images.

The intended role in a larger system is a reproducible baseline and reference implementation: a component that a reader's own evaluation or application code calls through `DeblurPipeline.restore`, or the artifact it exports.

###### Primary Intended Users

The intended users are machine-learning engineers, computer-vision researchers, students and instructors. The envisioned settings are teaching, research and self-hosted experimentation on a single machine or a hosted notebook runtime such as Google Colab or Kaggle.

Users are assumed to understand:

- that PSNR and SSIM need a sharp reference and measure pixel and structural fidelity, not perceived quality;
- that a model adapted to one synthetic degradation may perform worse on another, including real camera blur;
- that a held-out evaluation is only meaningful when the test scenes did not contribute training crops;
- that the upstream checkpoint is a pickle whose safety here depends on the pinned digest.

The pipeline is robust to malformed inputs in the sense that it refuses them with a message before any model runs. It is not robust to inputs that are well formed but semantically wrong, such as a "pair" whose two images show different scenes.

###### Out-of-scope use cases

Capability boundaries:

- **Not for denoising, deraining, super-resolution or stereo super-resolution.** The packaged checkpoint is the GoPro deblurring model. The SIDD denoising, REDS and NAFSSR checkpoints exist upstream but are not packaged here.
- **Not for video deblurring.** The network processes one frame at a time and uses no temporal information.
- **Not for defocus, lens or atmospheric blur, or spatially varying blur.** The sample and the fine-tuning target spatially uniform linear motion blur; other degradations were not evaluated.
- **Not for recovering detail that is absent from the input.** Restored texture is a statistical estimate, not recovered information.

Input boundaries:

- **Not for images with a side shorter than 64 px or longer than 1024 px.** `validate_records` and `read_byod` refuse them; they are never resized silently.
- **Not for 16-bit, floating-point or non-RGB data** (for example raw sensor data, medical DICOM, multispectral imagery). 16-bit and float images are refused; greyscale is replicated to RGB and alpha is dropped, with a reported conversion.
- **Not for fine-tuning with fewer than 4 pairs or with pairs from a single scene.** These are refused, because the split by scene would leave one side empty.
- **Degrades on JPEG-compressed, heavily noisy or low-light inputs.** The sample uses mild Gaussian noise (sigma 0.01 of the 0..1 range) and no compression; behaviour outside that is not measured.

Decision boundaries:

- **Not for forensic, evidential or legal use.** A restored image must not be presented as what a scene looked like, for example to identify a person, read a licence plate or establish an event.
- **Not for medical or safety decisions without independent domain validation**, such as reading clinical images or inspection imagery.

---

#### Factors

Restoration quality varies with the content of the image, the kind and strength of the blur, and the capture conditions; the subsections below say what was and was not examined.

###### Groups

The pipeline is not human-centric: it operates on pixels and has no notion of who or what is pictured. The evaluation data contains one portrait (the public-domain `astronaut` photograph) among three test photographs; no group-level analysis was possible or attempted.

The upstream training corpus, GoPro, consists of handheld video frames of street and indoor scenes that may include people. The authors did not publish a demographic audit of it, and none was performed here. Whether restoration quality differs across skin tones, faces or other human attributes is therefore unknown. An operator who applies the model to photographs of people should first evaluate restoration error on paired data that represents the people in their own application. That evaluation should report the error separately for each group of interest.

###### Instrumentation

The checkpoint was trained on GoPro, whose blurred images were made by averaging consecutive frames of 240 fps video from a GoPro Hero 4 Black camera (Nah et al., 2017); the sharp reference is the middle frame. The blur therefore reflects real handheld camera motion and that camera's sensor, lens, gamma and compression.

The sample in this repository is produced by software, not a camera. Sharp photographs from the `scikit-image` 0.26.0 data directory, from various cameras and sources, are convolved with straight-line motion kernels 9..21 px long at random angles. Gaussian noise of sigma 0.01 is added, and the result is quantised to 8 bits. There is no camera response curve, no demosaicing and no compression. Four of the nine sample photographs used for training and test (`brick`, `grass`, `camera`, `gravel`) are greyscale and are replicated to three identical channels.

Instrument effects reach the model directly as pixel error. A different camera pipeline, compression, noise level or blur shape changes the input distribution. Nothing in the pipeline detects such a shift: validation checks only structure, size and type.

###### Environment

Operating environment:

- Any Linux x86_64 machine with Python 3.12 and the pinned libraries; a CUDA GPU is optional. The tutorial's documented runtime is a Colab or Kaggle T4 GPU. A CPU-only runtime completes, more slowly.
- All computation is float32, and cuDNN is set to deterministic kernels. No mixed precision, quantisation or compilation is used.
- Memory grows with image size; the 1024 px side limit bounds it. Fine-tuning with the `all` scope needs more memory than the default `decoder` scope.
- The tutorial's isolated environment uses wheels for manylinux x86_64 only; other platforms are refused with a message.

Data environment:

- The pretrained model assumes blur similar to GoPro's real handheld camera motion on everyday scenes. The fine-tuned model in the tutorial assumes the sample's synthetic linear motion blur with mild noise.
- Fine-tuning assumes that the training and test pairs come from the same degradation process and that the scenes are independent; the split by scene protects only against scene overlap, not against a different degradation at test time.
- When the test degradation differs from the training degradation, quality degrades by an amount this pipeline does not measure. Adapting to synthetic blur may reduce quality on real camera blur.

---

#### Metrics

The pipeline measures fidelity to a known sharp reference, because that is the only quantity it can compute without human judgement.

###### Performance Measures

The pipeline reports two measures per image and their means over a split, under the keys `psnr` and `ssim` (`metrics.score`):

- `psnr` — peak signal-to-noise ratio in dB between the uint8 RGB output and the sharp reference, over all pixels and channels, with peak 255 and no border crop, as in the upstream GoPro test configuration. It captures squared reconstruction error on a log scale; +1 dB means about 21 % less mean squared error.
- `ssim` — structural similarity (Wang et al., 2004), with an 11 × 11 Gaussian window of sigma 1.5, computed per RGB channel over the valid region and averaged. It captures agreement of local means, contrasts and correlations, from −1 to 1.

PSNR is the training objective's own scale (the upstream `PSNRLoss`) and the standard deblurring metric. SSIM is reported beside it because the two disagree in informative ways: PSNR penalises a slightly misplaced sharp edge heavily, while a smooth, blurry output can keep a moderate PSNR and a poor SSIM. Reading only one of them would hide one of these failure modes. The upstream repository's default SSIM uses a 3-D Gaussian variant, so this pipeline's `ssim` is not numerically comparable with the upstream README table.

The comparison is always against baselines on the same pairs: `identity` (the blurred input), `unsharp` (blind unsharp masking), `wiener_oracle` (Wiener deconvolution given the true kernel), `pretrained` and `adapted`. `paired_difference` reports the per-pair gain, its minimum and maximum, and how many pairs improved.

For an unpaired image no reference exists, and the pipeline reports no performance measure. It reports only `gradient_energy_ratio_*`, a sharpness proxy that noise and ringing also raise. A caller who wants a measure for real blur must supply paired captures of the same scene, or a validated no-reference metric or human rating, none of which this pipeline provides.

No measured values for the real checkpoint are stated in this card, because no run with the real checkpoint has been recorded (see Verification records). The GoPro test results in the upstream README are the authors' report and are not reproduced or relied on here.

###### Decision thresholds

The pipeline makes no classification decision, so no score threshold is applied to its output. The output is a continuous image; the only implicit rule is that the network's float output is clamped to 0..1 and rounded to the nearest 8-bit value before scoring and export.

Thresholds set during development:

- the reload equivalence check accepts an exported artifact only if the largest absolute difference between its float outputs and those of the trained in-memory model, on two test pairs, is at most `1e-5`. The value allows for floating-point reordering on the same device and catches any change to the weights;
- the checkpoint conversion is accepted only if the safetensors copy reproduces the source outputs exactly (difference `0.0`);
- the ceilings 64..1024 px, 4..200 pairs and at least 2 scenes are operational limits that bound memory and keep a split possible, not quality thresholds.

No quality threshold is shipped, for example a minimum PSNR gain below which an output is rejected. A deployment that wants to fall back to the unprocessed image when restoration may have failed has to set that rule itself, from paired data of its own. It should weigh the cost of showing an artefact-laden restoration (a false acceptance) against the cost of discarding a useful one (a false rejection).

###### Approaches to uncertainty and variability

The estimation procedure is a single held-out split: 12 test crops from 3 test photographs, never used for training or for any choice. The classical baselines' settings are chosen on the 48 training crops only. Fine-tuning runs a fixed number of steps chosen in advance, so no model selection takes place.

No dispersion is reported: there is one seeded run, no repeated seeds, no bootstrap and no confidence interval. A small difference between two methods may not survive a different seed or a different set of test photographs.

Sources of variability and their control:

- the sample (kernels, noise, crop positions) is fully determined by seed `2022`;
- fine-tuning batches, crops and flips are determined by the training seed (default `0`);
- the `NAFNetLocal` construction pass uses a fixed seed and restores the global random state;
- cuDNN deterministic kernels are requested; GPU and CPU still produce slightly different floating-point results, so numbers from different devices are not expected to agree exactly.

The pipeline emits no probability or confidence output, so calibration does not arise.

---

#### Ethical considerations and biases

No external ethics review or group testing of this pipeline has taken place.

###### Data

The checkpoint was trained by the authors on GoPro (Nah et al., 2017): 3,214 blurred/sharp frame pairs from handheld video of everyday scenes, of which 2,103 are training pairs. The authors' training procedure is described in Chen et al. (2022) and the upstream configuration files. The authors do not enumerate the content of every frame, so whether it contains identifiable people or other sensitive content is not known; it is not ruled out.

This repository distributes code, the three upstream architecture files with their licence, and a manifest. It does not distribute the checkpoint, which is fetched at run time from a public mirror, or any image data. The tutorial's sample photographs ship inside the `scikit-image` 0.26.0 wheel and are public domain or CC0 according to that project's data documentation; one of them is a public-domain NASA portrait of an astronaut.

The pipeline performs no audit of the images a user supplies. An operator who supplies their own images, including through the notebook's bring-your-own-data branch, is responsible for having the right to process them and for checking them for personal, sensitive or proprietary content. The tutorial keeps uploaded files inside the runtime and sends them to no service.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life. It has not been validated for any such use by anyone.

Foreseeable misuse in a sensitive domain includes "enhancing" surveillance or evidential images, or sharpening medical images before they are read. Such use would be admissible only with independent validation on paired data from that domain, human review of every output, disclosure that the image was computationally restored, and whatever regulatory clearance the domain requires. Without those conditions, a restored image must not inform such a decision.

###### Mitigations

Mitigations implemented in this repository:

- **Supply-chain integrity.** The checkpoint is accepted only when its size equals 68,671,121 bytes and its SHA-256 equals `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`. A mismatching download is deleted, and there is no fallback to other bytes (`fetch_checkpoint`). The three upstream architecture files are checked against their SHA-256 before execution (`load_architecture`). The artifact manifest's file list, weight size and SHA-256 and the architecture digests are checked before deserialisation (`from_artifact`).
- **Serialisation safety.** The pickle is read only after verification, with `torch.load(..., weights_only=True)`, and only once. It must contain a `params` dict of tensors that loads strictly with exactly 17,111,907 parameters. Every later load and every export uses safetensors.
- **Input integrity.** `validate_records` refuses non-uint8, non-RGB, mismatched, undersized, oversized and duplicate inputs, and single-scene datasets, naming the record and the rule. `read_byod` refuses traversing paths, symbolic links, oversize archives and members, unmatched files and unreadable images, and reads zip members in memory without extracting them.
- **Evaluation integrity.** Splits are drawn by photograph or scene, and `dataset_manifest` refuses any overlap. Classical baselines are tuned on training data only. Later tutorial stages rebuild the sample and refuse to run if its digest changed.
- **Reproducibility.** The sample and training are seeded; the tutorial's dependencies are installed from a hash lock with `--require-hashes`; the notebook carries its code with per-file SHA-256 and records the generating revision; every run writes a `result.json` with model identity, runtime versions and configuration.
- **Refusals.** The pipeline does not compute PSNR or SSIM without a reference, and does not resize oversized inputs silently.

###### Risks and harms

- **Hallucinated detail presented as fact.** The network can produce plausible edges and texture that were not in the scene. Viewers of the output, and third parties depicted or affected by decisions based on it, bear the harm. It is likely whenever blur is strong, and the harm can be severe if the image is used as evidence.
- **Silent degradation out of distribution.** On blur types, noise levels or compression the model was not trained for, outputs may contain ringing or artefacts, with no warning from the pipeline. The operator bears the harm; it is likely for real-world inputs that differ from GoPro or from the sample.
- **Regression after fine-tuning.** Adapting to the synthetic sample can reduce quality on real camera blur, and the tutorial does not measure that. The operator who ships the adapted artifact bears it.
- **Overconfidence in small evaluations.** Twelve crops from three photographs with no dispersion estimate can make a small difference look meaningful. Decision-makers who read the number bear the harm.
- **Automation bias.** Users may accept restored images without inspection because a metric improved. This harms the operator and anyone affected downstream, and is likely in batch use.
- **Bias from unaudited training data.** GoPro was not audited for demographic balance, so restoration quality on faces or skin tones may vary in ways that were not measured. People depicted in the images bear the harm.
- **Data leakage in user evaluations.** A user who splits crops of the same scene across training and test inflates their results. The pipeline prevents it only when `source` is set correctly; the BYOD reader treats each image as its own scene.
- **Trusted-code exposure.** If the pinned digest were ever edited to accept other bytes, a malicious pickle could reach `torch.load`. `weights_only=True` limits this but does not make an untrusted pickle safe.

###### Use cases

Beyond the out-of-scope uses above, the developers consider these uses unacceptable even where the model would run:

- enhancing surveillance footage or photographs to identify, track or profile people, including biometric identification;
- producing "restored" images presented as authentic records in journalism, insurance claims, legal proceedings or disputes without disclosing that they were computationally altered;
- deceptive or manipulative uses, such as fabricating sharp versions of images to mislead about events;
- any use that violates the MIT or Apache-2.0 licence terms, for example redistribution without the licence notices, or that violates the terms of the runtime or the rights in the images processed.

---

## Immutable provenance

| Item | Value |
|---|---|
| Upstream model | `megvii-research/NAFNet`, variant `NAFNet-GoPro-width32` |
| Upstream revision | `2b4af71ebe098a92a75910c233a3965a3e93ede4` |
| Checkpoint | `NAFNet-GoPro-width32.pth`, 68,671,121 bytes, SHA-256 `19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c` |
| Digest provenance | stated for the authors' Google Drive file (id `1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj`) by two independent third-party projects; not computed by this repository; see `docs/WEIGHTS.md` |
| Distribution | public Hugging Face mirrors `nyanko7/nafnet-models`, then `mikestealth/nafnet-models`, addressed by branch and accepted only at the pinned digest; not yet confirmed to serve the pinned bytes |
| Architecture source | `NAFNet_arch.py` `01b22270cc93f1bb90c0e3e4490e98b023fcf73f8552860b4a9ee880ce5c6967`, `arch_util.py` `5a11af2e7c2d7a7b57c1fbd7e19cf0a50b4b4e8c7ae7dd203a915d7a707e7005`, `local_arch.py` `c4df2ba4d896442a0f6ec984accd6e68f31edce3afdf066add202c25a0d1af26` (SHA-256, verbatim at the upstream revision) |
| Converted serving file | `model.safetensors`, derived from the checkpoint; SHA-256 recorded per run in `outputs/weights.json`, not yet pinned |
| Manifest | `weights/nafnet-gopro-width32/dimer-base-manifest.json` |
| Licence | MIT (NAFNet); Apache-2.0 (BasicSR-derived parts) |

## Input/output contract

| Item | Contract |
|---|---|
| Inference input | uint8 RGB array H × W × 3, 64 ≤ H, W ≤ 1024 |
| Inference output | uint8 RGB array of the same shape (`DeblurPipeline.restore`) |
| Paired record | `{id, source, blurred, sharp}`, identical shapes; `source` groups crops of one scene |
| Fine-tuning | 4..200 pairs from ≥ 2 sources; `steps` 1..5000, `batch_size` 1..32, `lr` in (0, 1e-2]; scopes `decoder`, `decoder+middle`, `all` |
| Artifact | directory with exactly `model.safetensors` and `manifest.json` (format `dimer_nafnet_restoration_artifact`, version 1) |
| BYOD archive | zip or directory: `sharp/` + `blurred/` matched by file stem (paired), or blurred images only (unpaired) |

## Verification records

The tutorial is a release candidate. A static check (tests, validator, notebook parity) is not an execution and is not listed here.

- **Date:** 2026-10-04
- **Subject:** `tutorials/nafnet_deblurring_colab.ipynb` at commit `31db8bb`
- **Runtime:** CPU only (4 shared cores, no GPU), Python 3.12.12, `torch 2.14.0`, `numpy 2.5.3`, `safetensors 0.8.0`, `scikit-image 0.26.0`
- **Procedure:** every code cell run in order in one kernel, with `STEPS = 60`, the optional activity (`decoder+middle`) and the paired bring-your-own-data branch switched on. Not a hosted `Run all`: the `uv` download and environment build were replaced by an equivalent pinned environment because of limited disk, and the real checkpoint was replaced by a random-init full-width stand-in with its own digest because the mirror host was unreachable.
- **Observed result:** all 11 code cells completed in one pass; all five refusal probes were rejected; test PSNR/SSIM `identity` 23.32 dB / 0.6264, `unsharp` 23.37 dB / 0.6186, `wiener_oracle` 25.18 dB / 0.7134; the reloaded artifact reproduced the trained model's outputs with `max_abs_float_diff 0.0`; a refused bring-your-own-data archive stopped with a message naming the unmatched files.
- **Caveats:** the `pretrained` and `adapted` values of this run come from random weights and say nothing about NAFNet; the real-checkpoint path (download, digest match, conversion of the real bytes) and the `uv` environment build were not executed. Full stage-by-stage records are in `docs/release-verification.md`.

## References

- Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). Simple baselines for image restoration. In *Computer Vision – ECCV 2022* (pp. 17–33). Springer. https://doi.org/10.1007/978-3-031-20071-7_2
- Chu, X., Chen, L., Chen, C., & Lu, X. (2022). Improving image restoration by revisiting global information aggregation. In *Computer Vision – ECCV 2022* (pp. 53–71). Springer. https://doi.org/10.1007/978-3-031-20071-7_4
- Nah, S., Kim, T. H., & Lee, K. M. (2017). Deep multi-scale convolutional neural network for dynamic scene deblurring. In *2017 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 257–265). https://doi.org/10.1109/CVPR.2017.35
- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing, 13*(4), 600–612. https://doi.org/10.1109/TIP.2003.819861
- van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., et al. (2014). scikit-image: Image processing in Python. *PeerJ, 2*, e453. https://doi.org/10.7717/peerj.453
- Upstream repository and licence: https://github.com/megvii-research/NAFNet
