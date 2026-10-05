# NAFNet deblurring notebook — review fixes

**Review:** `nafnet_deblurring_colab_Review.md` (5 October 2026, NAF-m1..m5, no Majors).
**Fixed in:** the generator (`tools/build_notebook.py`, `tools/notebook_template.py`, `tools/tutorial_stages.py`) and the carried package (`samples.py`, `__init__.py`); the notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| NAF-m1 | Fixed (timings); converted-digest pin not done | The Run-all line and the Prerequisites quote the recorded T4 run (4 October 2026, revision `31db8bb`, 66 s environment, per-stage times, 0.18 s per training step) beside the CPU estimate, labelled as an estimate from the development container. The 9 GB disk figure is labelled an estimate. *Running it* and the install note quote the measured build time. The converted-safetensors digest pin (`STATUS.md` item 1) is a new enforcement that this cycle cannot verify on a GPU: left to the maintainer. | `run_all`, Prerequisites, How to use, install note, Troubleshooting | `test_naf_m1_no_stale_hosted_run_statement_and_timings_quote_the_t4_run`; `grep "has not been recorded yet"` returns nothing |
| NAF-m2 | Fixed | `samples.py` keeps the colour-conversion note (`load_sample_image_with_note`, `sample_conversions`). `prepare` prints the conversions, writes them to `dataset.json` (`colour_conversions`) and adds a `source_conversion` column to `nafnet_deblurring_sample_pairs.csv`. Section 4 states that four of the nine sources (24/48 training and 4/12 test pairs) and both new images are greyscale, and what that means for the adaptation; Section 8 names the new images as greyscale. The sample arrays and the dataset digest are unchanged. | `samples.py`, `__init__.py`, `stage_prepare`; Sections 4 and 8 | `test_naf_m2_prepare_reports_and_records_every_greyscale_replication` (runs on the pinned scikit-image 0.26.0 photographs: real input); dataset digest `dd35d755…5ba7` identical before and after |
| NAF-m3 | Fixed | Section 9: "about thirteen times" with both counts. Section 7 sample answer: no gravel claim; the recorded per-photograph gains (rocket +0.14, gravel +0.22, astronaut +0.33 dB) with a seed caveat. Section 8: "handwritten formulae on paper"; a ratio below 1 is explained (smoother output; consistent with under-correcting real blur; not a quality score), quoting the recorded 1.08 / 0.81. | Sections 7, 8, 9 prose | `test_naf_m3_prose_matches_the_recorded_run` |
| NAF-m4 | Fixed | The upload branch requires exactly one file and otherwise raises "Upload exactly one .zip file (received N files): zip a folder … first. Or set BYOD_PATH …"; Section 10 says the dialog takes one `.zip`; a Troubleshooting row. | BYOD cell, Section 10, Troubleshooting | `test_naf_m4_cancelled_or_multi_file_upload_stops_with_the_rule` (`{}` and two files; `run_stage` not called) |
| NAF-m5 | Fixed (stronger than the acceptance check) | Same change as the MediaPipe pilot: the Section 1 cell keeps this session's run directory on a re-run (`NEW_RUN_DIRECTORY` form field for a fresh one); the environment directory is keyed on the lock digest, managed Python and `uv` version, and a `ready.json` marker lets a later Run all reuse it with no download; `run_stage` names Sections 1–3 when the run directory or interpreter is missing; Troubleshooting row; `PYTHONSTARTUP` dropped from the stage environment. | `CHECK_CELL`, `INSTALL_CELL`, `run_stage`; validator markers | `test_naf_m5_*` (4 tests) |

Suggestions are not taken in this cycle.

## User-visible changes

- Section 1 has a new form field, `NEW_RUN_DIRECTORY` (off). Re-running Section 1, or a second Run all in the same kernel, keeps the run directory; the environment is reused (`environment_reused: True`) instead of downloading the CUDA torch wheels again.
- The environment lives at `<tmp>/nafnet_deblurring_env_<lock key>`.
- `prepare` prints `colour_conversions`; `dataset.json` gains `colour_conversions`; the sample-pairs CSV gains `source_conversion`.
- The BYOD upload dialog refuses zero or several files with a message.

## Verification (offline, not clean-runtime evidence)

- `build_notebook.py --check`: OK. `validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's lightweight dependencies and **no torch** (torch tests skip): 44 passed / 14 skipped before, **54 passed / 14 skipped** after.
- **Real input:** the `prepare` stage ran on the pinned scikit-image 0.26.0 photographs and reported six greyscale replications (`brick`, `grass`, `camera`, `gravel`, `text`, `clock_motion`), matching the review's P3. Dataset digest unchanged.
- No model stage ran: `huggingface.co` is unreachable from this container and torch is not installed. The timing and ratio numbers in the prose are quoted from the recorded Colab T4 run in `docs/release-verification.md`.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated notebook on a fresh T4 runtime (no restart), then a re-run of the export/reload cell.
2. The REL12 BYOD journey on a hosted runtime (one paired, one unpaired, one refused archive), including the upload dialog.
3. Maintainer decision: pin the converted `model.safetensors` digest (`STATUS.md` item 1) once a run confirms it is identical on CPU and GPU.
