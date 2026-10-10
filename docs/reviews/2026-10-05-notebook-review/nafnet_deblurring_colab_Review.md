# NAFNet Deblurring E2E Notebook — Review

**Verdict: Needs revision** (no Major findings; five Minors)  
**Review date:** 5 October 2026  
**Repository:** `kurtvalcorza/nafnet-deblurring-pipeline`  
**Notebook:** `tutorials/nafnet_deblurring_colab.ipynb`  
**Reviewed commit:** `2e8078d` (`main`, the merge of PR #1)  
**Notebook Git blob:** `a62888122407ee450e0256db60658e8161ee68b9`, generated from `31db8bb`. This is the blob executed in the recorded Colab T4 run of 2026-10-04. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `NAF`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2, `ml-worker` `origin/main` at `b9fdd1f`.

## Executive assessment

The notebook is standalone, generator-built and declared `E2E` / `GUIDED`. It is technically sound.

- **Isolation.** It installs nothing into the kernel: a pinned `uv` builds managed CPython 3.12.12 from a 38-package hash lock, and every stage runs in a subprocess.
- **Checkpoint handling.** The pickle checkpoint is refused unless its pinned size and SHA-256 match, from either of two mirrors serving the same bytes. It is then read once with `weights_only=True`, converted to safetensors, and the conversion is accepted only on identical outputs.
- **Data and evaluation.**
  - The 60 synthetic blurred/sharp pairs are split by photograph.
  - Baselines are chosen on training pairs only: unsharp masking and an oracle Wiener filter, with the oracle's privilege stated plainly.
  - The bounded decoder fine-tuning always restarts from the verified base, so a re-run repeats the experiment instead of continuing it.
  - The exported artifact is evaluated in a fresh process and reloaded in a second fresh process with a stated float tolerance.
- **Real-blur honesty.** A real blurred photograph gets no PSNR, only a sharpness proxy that the notebook labels as such.
- **Guided layer.** Complete, from audience and how-to-use through glossary, predictions, checkpoints, a scope activity, troubleshooting, a conclusion scaffold and a transfer prompt.

The checkpoint could not be fetched from this container (`huggingface.co` is denied by the egress policy). This review therefore relies on the recorded Colab run for model behaviour, and on source inspection plus torch-free probes for the data, BYOD and upload paths.

| Measure (Colab T4, blob `a628881`, 2026-10-04) | Value |
|---|---|
| Code cells | 11/11, one pass, no restart; environment build 66 s, stages ≈ 136 s |
| Checkpoint | first mirror served the pinned bytes; conversion exact (`max_abs_output_diff_vs_pth` 0.0) |
| Test PSNR / SSIM (12 pairs) | identity 23.32 / 0.6264 · unsharp 23.37 / 0.6186 · Wiener oracle 25.18 / 0.7134 · pretrained 26.13 / 0.7711 · adapted 26.36 / 0.7881 |
| `adapted_vs_pretrained` | +0.227 dB mean (−0.165 to +0.417), 11/12 pairs improved |
| Per photograph (pretrained → adapted) | astronaut 25.69 → 26.02 · gravel 23.59 → 23.81 · rocket 29.11 → 29.25 |
| Reload parity | `max_abs_float_diff` 0.0 (tolerance 1e-5) |
| New images | text: 25.35 → 30.91 (pretrained) / 31.29 (adapted) dB · clock gradient-energy ratio 1.081 (pretrained) / 0.807 (adapted) |

Five Minors stand between this notebook and `Ready for intended use`:

1. **Stale statements (NAF-m1).** The opening cell and the Prerequisites say no hosted T4 run has been recorded, and quote only CPU estimates.
2. **Unreported greyscale conversion (NAF-m2).** Six of the eleven sample photographs are greyscale, including half the training sources, one test source and both new images. They are replicated to RGB without the report that the data contract promises.
3. **Guided prose that contradicts the numbers (NAF-m3).** For example "sixteen times as many trainable parameters", which is 12.9×. A gradient-energy ratio below 1 is left unexplained.
4. **BYOD upload branch (NAF-m4).** A cancelled dialog raises a bare `StopIteration`, and a multi-file upload silently uses the first file.
5. **Re-run behaviour (NAF-m5).** Re-running the Section 1 cell alone strands later cells, and each Run all rebuilds the several-GB environment.

The REL12 BYOD journey on a hosted runtime has not been recorded. Neither has the pin of the converted safetensors digest (both are `STATUS.md` open items).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata, opening cell) |
| Declared spec | DIMER Notebook Specification **2.2**, standalone |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Stated: learners who run hosted notebook cells and read short Python; no deblurring background assumed |
| Supported runtime | Linux x86_64; Colab or Kaggle T4 documented, CPU completes more slowly; float32, deterministic cuDNN |
| Promised outcomes | <ul><li>isolated install with no restart</li><li>verified and converted checkpoint</li><li>60 validated pairs split by photograph, with five refusal probes</li><li>four-method baseline table</li><li>300-step decoder fine-tuning</li><li>fresh-process evaluation of the artifact</li><li>fresh reload with parity</li><li>two new images: one with a reference, one real-blur image without</li><li>provenance record</li><li>optional scope activity</li><li>paired and unpaired BYOD</li></ul> |
| Generator | `tools/build_notebook.py` (`build_notebook.py/3.0-nafnet`) + `tools/notebook_template.py`; generating revision `31db8bb` |
| Release status | `Candidate` (`STATUS.md`): default path recorded on Colab; hosted REL12 BYOD and the converted-digest pin open |

### Evidence actually obtained

- **Source inspection.**
  - All 40 cells (11 code); cell 9 is the 13-file carrier.
  - `tools/tutorial_stages.py`: every stage, including activity and BYOD.
  - `src/nafnet_deblurring_pipeline/samples.py`: degradation, validation, `read_byod`, `split_by_source`, `to_rgb`, `load_sample_image`.
  - `pipeline.py`: `TRAINABLE_SCOPES` and the checkpoint path.
  - `STATUS.md`, `docs/WEIGHTS.md`, `docs/release-verification.md`.
- **Documented execution evidence.**
  - The Colab T4 run of the reviewed blob: `docs/execution-evidence/2026-10-04/…_colab-t4.ipynb` and `colab_t4_run_summary.json`. The default path was run; activity and BYOD were not.
  - The development-container CPU run with a random-init stand-in checkpoint, which covered activity and BYOD plumbing (`docs/release-verification.md`). It is not model evidence.
- **Direct execution (this review), without torch or the checkpoint.**
  - Environment: Python 3.12 with `numpy` 2.5.3, `pillow` 11.3.0 and `scipy` 1.18.1. The data files came from the locked `scikit-image` 0.26.0 wheel (SHA-256 `7df650e7…`, which is in the lock).
  - P3: the sample builder, plus the colour mode of every pinned photograph.
  - P4: `read_byod` and `split_by_source` on 11 constructed archives (4 compatible, 7 incompatible).
  - P5: the BYOD cell's upload branch, with `google.colab.files.upload` stubbed.
  - Scripts and results are in `nafnet_deblurring_colab_Review_Probes.zip`.
- **Not executed here:** any torch stage.
  - `huggingface.co` and `download.pytorch.org` are denied by this container's egress policy.
  - The locked torch is the multi-GB CUDA build.
  - Model behaviour is taken from the Colab record and is labelled so throughout.
- **Learner observation:** none.

## 2. Separate judgments

- **Technical correctness:** good.
  - Checkpoint trust boundary, safe conversion with an output-equality check, split by source with a leakage check, baselines tuned on training only, restart-from-base fine-tuning, artifact manifest verification before weights load, and reload parity at 0.0.
  - One data-handling gap: silent greyscale replication in the sample path (NAF-m2).
- **Promise fulfilment:** every promised stage ran on Colab and wrote its outputs. The gaps are prose that does not match the recorded run (NAF-m1, NAF-m3).
- **Learner experience:** strong.
  - The oracle is framed honestly, the training loss is not treated as evidence, and the real-blur image has no metric.
  - The prediction prompts have sound sample answers.
  - Friction: misleading numbers in the guided prose (NAF-m3), BYOD upload errors (NAF-m4) and partial re-runs (NAF-m5).
- **Spec conformance:**
  - Unresolved MUSTs: SRC3 (NAF-m1, NAF-m3), VAL7 by analogy together with the notebook's own stated contract (NAF-m2), and UX12 (NAF-m1).
  - SHOULD deviations: UX10 and DAT19 (NAF-m4); UX10 and GDL13 (NAF-m5).
  - REL12 hosted evidence is absent.
  - GDL1–GDL15 are satisfied.

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result (Colab unless noted) | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass Run all, no restart | cells 6, 9, 12 | 11/11 in order, `setup_seconds` 66 | Section 2 prose | Met |
| Pinned checkpoint, safe conversion | cell 15 / `weights` | verified, first mirror; conversion diff 0.0 | trust boundary explained | Met |
| 60 pairs, split by photograph, refusals | cell 18 / `prepare` | 48/12, `disjoint_sources: True`, 5 refusals naming rules | leakage checkpoint | Met, but greyscale replication unreported (NAF-m2) |
| Baselines and pretrained on test | cell 21 / `baseline` | 4 rows; pretrained beats the oracle by 0.95 dB; 11/12 improved over the input | oracle and unsharp checkpoint | Met |
| Bounded decoder fine-tuning from the base | cell 24 / `adapt` | 1,322,307 trainable / 17,111,907; 54.9 s; artifact exported | loss-is-not-evidence checkpoint | Met |
| Fresh-process held-out evaluation | cell 27 / `evaluate` | +0.227 dB, 11/12; per-photograph lines | honest-sentence checkpoint; "gravel often gains least" (rocket gained least) | Met (see NAF-m3) |
| Reload parity; new images; provenance | cell 30 / `reload` | parity 0.0; text +5.6 / +5.9 dB; clock ratio 1.081 / 0.807; `result.json` and listing | no-reference checkpoint; a ratio below 1 is not explained | Met (see NAF-m3) |
| Activity: change the trainable scope | cell 33 / `activity` | not run on Colab; same seed and steps, starts from the base, prints three rows | "sixteen times as many trainable parameters" (12.9×) | Met in source and stand-in run; NAF-m3 |
| BYOD paired / unpaired | cell 36 / `byod` | P4: paired (including a wrapping folder and greyscale, both reported), unpaired, and a `README.txt` ignored and listed; 7 incompatible archives refused with the rule | contract stated first | Met locally (reader); hosted REL12 pending; upload branch NAF-m4 |
| Runtime statements | cells 0, 4 | "Its duration on a T4 has not been recorded yet"; "no hosted T4 run … has been recorded yet" | — | **Not met** (NAF-m1) |

All eight learning objectives are observable learner actions, and each is backed by a prediction or checkpoint with a sample answer.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 40 cells | The guided layer is complete. The learner meets: <ul><li>a "no hosted run yet" claim the record contradicts (NAF-m1);</li><li>half-greyscale training data the notebook never mentions (NAF-m2);</li><li>a parameter ratio, a "gravel gains least" expectation and a "printed-text" description that the output contradicts (NAF-m3).</li></ul> |
| **Clean default** | Documented (Colab T4, reviewed blob) | 11/11 in one pass; numbers in the table above. Not re-executed here (no checkpoint access). |
| **Active learning** | Source + documented stand-in run | Re-running Section 6 restarts from the verified base (`load_base` in `stage_adapt`), so a changed field gives a clean comparison. The activity copies steps, learning rate, batch size and seed from the default run and prints the first logged loss of both runs to show the shared start. Not run on a hosted runtime. |
| **Reuse and recovery** | Direct (P4, P5) + source | See the three parts below. |

**Reuse and recovery, BYOD reader (P4).**

- Accepted:
  - 6 RGB pairs, split 4/2;
  - 4 greyscale pairs inside a wrapping folder, with the conversions reported;
  - 3 unpaired top-level images;
  - 4 pairs plus `README.txt`. The text file is ignored and listed in `ignored_files`.
- Refused, each naming the file and the rule:
  - an unmatched stem;
  - a pair whose sizes differ;
  - an 1100 px image;
  - 3 pairs (below the minimum of 4);
  - a `../` member;
  - a nested folder.

**Reuse and recovery, upload branch (P5).** A cancelled dialog (`{}`) raises a bare `StopIteration`. Two uploaded files run the stage on the first and silently drop the second (NAF-m4).

**Reuse and recovery, re-runs (source).** Cell 6 has the same `uuid4` run-directory pattern as the MediaPipe notebook, where re-running Section 1 alone was shown to strand later cells (NAF-m5).

## 5. Findings

### Minor

#### NAF-m1 — The opening cell and Prerequisites say no hosted run exists; the timing figures are CPU estimates only

- **Cell/section:** cell 0 (**Run all**: "Its duration on a T4 has not been recorded yet; see the Prerequisites for what has been measured"); cell 4 (*Measured timings*: "no hosted T4 run of this notebook has been recorded yet" and a 2.3 s/step CPU estimate); cell 4 disk ("about 9 GB"). Generator: `tools/notebook_template.py`.
- **Observed issue:** the 2026-10-04 Colab T4 run of this exact blob is recorded in `docs/release-verification.md`, `STATUS.md` and `docs/execution-evidence/`:
  - environment 66 s;
  - weights 13.5 s, prepare 6.2 s, baseline 22.3 s, adapt 70.8 s (54.9 s of training, ≈ 0.18 s/step), evaluate 12.3 s, reload 11.3 s.

  The notebook still tells the learner none exists and gives only a CPU extrapolation. `STATUS.md` item 1 already plans the fix, together with the digest pin.
- **Consequence:** the learner's only runtime guidance is wrong for the documented runtime, by roughly 13× per step.
- **Evidence:** documented, from the Colab outputs above.
- **Recommended correction:**
  - Quote the recorded T4 timings, with runtime, date and revision, beside the labelled CPU estimate.
  - Label the 9 GB figure as an estimate or measure it.
  - Do this in the same regeneration as the converted-digest pin, so only one new hosted run is needed.
- **Acceptance check:** `grep -n "has not been recorded yet" tutorials/nafnet_deblurring_colab.ipynb` returns nothing, and the Prerequisites quote a T4 timing that matches a run in `docs/release-verification.md`.
- **Spec:** SRC3, UX12.

#### NAF-m2 — Six of eleven sample photographs are greyscale and are replicated to RGB without the promised report

- **Cell/section:**
  - code: `samples.py` `load_sample_image` (`array, _note = to_rgb(image)` discards the note);
  - prose: cell 4 (*Data contract*: "Greyscale is replicated to three channels and alpha is dropped, and both are reported"), Section 4 (cells 17–19), Section 8 (cell 29).
- **Observed issue:**
  - `brick`, `grass` and `camera` are greyscale: 3 of the 6 training sources, so 24 of 48 training pairs.
  - `gravel` is greyscale: 1 of the 3 test sources, so 4 of 12 test pairs.
  - Both new images are greyscale: `text` and `clock_motion`.
  - All are replicated to RGB silently. The prepare records carry no conversion field.
  - The BYOD path does report the same conversion, so the two paths differ.
  - No prose mentions that half the fine-tuning data is colourless.
- **Consequence:**
  - The notebook breaks its own stated data contract on its own sample.
  - The learner cannot see that the adaptation is partly a fit to grey images, which changes how the "adapted to a new blur distribution" result should be read.
  - The learner also cannot see that the real-blur check in Section 8 is on a grey photograph.
- **Evidence:** direct (P3), using the pinned files from the locked `scikit-image` 0.26.0 wheel:
  - `brick`, `grass`, `camera`, `gravel`, `text` and `clock_motion` are mode `L`;
  - the sample record keys are `blurred, id, kernel, kernel_array, noise_sigma, sharp, source, split, window`, with no conversion note.
- **Recommended correction:**
  - Keep the note in `load_sample_image`, record it per photograph in `dataset.json` and the sample-pairs CSV, and print it in `prepare`.
  - State in Section 4 that four of the nine sources are greyscale, and say what that means for the adaptation (the colour channels of those pairs are identical).
- **Acceptance check:** a default run prints the six greyscale replications in Section 4, and `nafnet_deblurring_sample_pairs.csv` carries a conversion column.
- **Spec:** VAL7 (by analogy: inputs changed silently), DAT12, plus the notebook's own data contract (SRC3).

#### NAF-m3 — Guided prose contradicts the printed numbers in three places, and leaves a sub-1 sharpness ratio unexplained

- **Cell/section:**
  - Section 9 predict prompt (cell 32);
  - Section 7 prompt and sample answer (cells 26, 28);
  - Section 8 description (cell 29) and "What to notice" (cell 31).
- **Observed issue:**

  | Where | The notebook says | The run shows |
  |---|---|---|
  | Section 9 | "sixteen times as many trainable parameters (`all`)" | 17,111,907 / 1,322,307 = **12.9×**, from the counts the notebook prints |
  | Section 7 | "gravel … is the hardest to restore and often gains least", hedged with "read your own" | rocket gained least (+0.14 dB), gravel +0.22, astronaut +0.33 |
  | Section 8 | `text-synthetic` is "a printed-text image" | the pinned `text.png` is handwritten formulae on paper |

  The adapted model's gradient-energy ratio on the real clock is **0.807**: its output has *weaker* edges than the blurred input. The prose explains only why a *higher* ratio proves nothing.
- **Consequence:**
  - The prediction scaffold anchors learners to a ratio the output contradicts.
  - The only sample answer about per-photograph gains points the wrong way on the recorded run.
  - The most informative real-blur signal (the adapted model smooths real blur) goes unremarked.
- **Evidence:** documented (Colab outputs) and source; the `text.png` content was checked from the pinned file (P3) and the Colab new-image sheet.
- **Recommended correction:**
  - Write "about thirteen times", or compute the ratio in the activity output.
  - Make the gravel sentence conditional, or drop it.
  - Say "handwritten text".
  - Add one sentence: a ratio below 1 means the output has less edge energy than the input (smoother). That is consistent with a model tuned to straight-line blur under-correcting real blur, but it is still not a quality score.
- **Acceptance check:** no number or comparative claim in the learner prose contradicts the recorded default run, and Section 8 explains a ratio below 1.
- **Spec:** SRC3, UX4.

#### NAF-m4 — BYOD upload: a cancelled dialog raises a bare `StopIteration`, and extra files are silently dropped

- **Cell/section:** cell 36 (`uploaded = files.upload(); file_name, payload = next(iter(uploaded.items()))`).
- **Observed issue:** with no `BYOD_PATH`:
  - cancelling the dialog (`{}`) raises `StopIteration` with no message;
  - selecting several files, for example the `sharp` and `blurred` folders' files instead of one zip, runs the stage on whichever file comes first and ignores the rest without a word.
- **Consequence:** the first BYOD attempt of a learner who selects their images directly gets either an empty traceback, or a result for one file when they expected a dataset.
- **Evidence:** direct (P5), executing the cell's own source with `files.upload` stubbed:

  | Upload | Outcome |
  |---|---|
  | `{}` | `StopIteration` |
  | `{a.zip, b.zip}` | `run_stage('byod', '--byod', …/a.zip, …)`, with `b.zip` unused |
- **Recommended correction:** require exactly one uploaded file, with a message such as "Upload exactly one zip (or set `BYOD_PATH` to a directory)", as the MediaPipe notebook's BYOD cell does. Say in Section 10 that a folder must be zipped first.
- **Acceptance check:** a cancelled dialog and a two-file upload each stop with a message naming the rule.
- **Spec:** UX10, DAT19.

#### NAF-m5 — Re-running the Section 1 cell alone strands later cells; every Run all rebuilds the several-GB environment

- **Cell/section:** cell 6 (`ROOT` and `ENV_ROOT` from a fresh `uuid4`); cells 9 and 12; Troubleshooting (cell 38).
- **Observed issue:**
  - Re-running only cell 6 points `ROOT` at an empty directory while `PYTHON` still names the earlier environment. The next learner cell fails with `can't open file '<new ROOT>/tutorial_stages.py'` and `Stage '…' failed (exit 2): see the log above`, and no Troubleshooting row covers it.
  - Every Run all creates a new `ENV_ROOT` with its own `uv` cache, so the torch CUDA wheels are downloaded and installed again (66 s on Colab), and the Section 1 disk check demands another 9 GB.
- **Consequence:**
  - A natural "check the GPU again" re-run after switching runtime type yields an unguided error.
  - Repeated Run alls in one session cost minutes and disk.
- **Evidence:** inferred from source. The code path is identical to the MediaPipe notebook's cell 6, where this review executed it (`mediapipe-face-landmarker-pipeline` review, P5); not executed here.
- **Recommended correction:**
  - In `run_stage`, check that `ROOT/tutorial_stages.py` and `PYTHON` exist and name the cells to re-run.
  - Add the Troubleshooting row.
  - Key `ENV_ROOT` on the lock digest so that a verified environment is reused.
- **Acceptance check:** re-running cell 6 and then cell 18 stops with a message naming Sections 2–3. A second Run all in the same runtime reuses the environment.
- **Spec:** SRC2, UX10, GDL13.

### Suggestions

- **NAF-S1** — The new-image sheet renders the 448 × 172 text image tiny, with large blank areas. Size rows to their content so that the clock comparison the learner is asked to judge by eye is legible.
- **NAF-S2** — A 4-pair BYOD archive splits 3/1 (P4), so the BYOD comparison then rests on one image. Print that count beside the metrics; the prose already warns, briefly.
- **NAF-S3** — Report the per-photograph gains with the number of pairs that improved, so that the "11/12" result can be traced to the photograph that did not improve.
- **NAF-S4** — Once the converted digest is pinned, let later stages check the pinned value rather than the value recorded by `weights` in the same run.

## 6. Readiness

**Needs revision.** There are no Majors, but NAF-m1, NAF-m2 and NAF-m3 involve MUST-level wording or contract deviations.

All five Minors are generator or package edits. NAF-m1 fits the regeneration that `STATUS.md` item 1 already plans (the converted-safetensors digest pin).

Remaining gates after the fixes:

- one hosted one-pass Run all of the regenerated blob;
- the REL12 BYOD journey on a hosted runtime, including the upload dialog: one paired archive, one unpaired archive and one refused archive, as `STATUS.md` specifies.

## 7. Verified versus inferred

- **Verified by direct execution (torch-free):**
  - the colour modes of all 11 pinned photographs, and the absence of any conversion note in the sample records;
  - the `read_byod` and `split_by_source` acceptance and refusal matrix;
  - the upload-branch behaviour of the BYOD cell, with its own source and a stubbed dialog.
- **Verified from documented evidence:** every model number (Colab T4, reviewed blob); activity and BYOD plumbing (development stand-in run; not model evidence).
- **Inferred from source:** restart-from-base in `adapt` and `activity`; the NAF-m5 re-run behaviour (identical to the code executed in the MediaPipe review); the per-run environment rebuild.
- **Not verified:** the second Hugging Face mirror; the activity and BYOD on a hosted runtime; the 9 GB disk figure.
- **Most likely to be wrong:** NAF-m2's weight. A maintainer could treat greyscale replication of sample photographs as harmless. It is rated a finding because the notebook promises to report it and does so on the BYOD path.
