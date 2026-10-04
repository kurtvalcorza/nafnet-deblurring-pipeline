"""Stage runner for the standalone NAFNet deblurring tutorial (NOTEBOOK_SPEC 2.2 §25.13 isolated-environment pattern).

The tutorial notebook carries this file verbatim (as ``tutorial_stages.py`` in its run directory, beside the carried
package under ``src/`` and the carried upstream architecture under ``third_party/``) and runs every stage with the
interpreter of an isolated, hash-locked environment::

    python -u tutorial_stages.py --root RUN_DIR --weights WEIGHTS_DIR --stage prepare

Nothing is installed into the notebook kernel. Each stage is a separate process, so a stage starts from files only:
the verified checkpoint and its converted safetensors under ``--weights``, the dataset digest recorded by ``prepare``
(the deterministic sample is rebuilt and must reproduce it), the baseline report, the exported artifact and the JSON
records of earlier stages. Learner-facing exports go to ``RUN_DIR/outputs``; hand-off state goes to ``RUN_DIR/state``.
On failure a stage writes ``RUN_DIR/state/<stage>.error.json`` with the exception type and message, which the notebook
re-raises in the kernel.

Stages: weights → prepare → baseline → adapt → evaluate → reload, plus the optional ``activity`` and ``byod``.
"""
# ruff: noqa: E501  -- the printed dictionaries are the learner-facing output; they are kept on one line each
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

STEM = "nafnet_deblurring"
MODEL_KEY = "nafnet-gopro-width32"
SPLIT_SEED = 0
REFERENCE_RECORDS = 2  # test records whose float outputs anchor the reload-equivalence check
RELOAD_TOLERANCE = 1e-5  # max |float output difference| accepted between the trained model and its reloaded artifact


# --------------------------------------------------------------------------------------------------
# run context and small helpers
# --------------------------------------------------------------------------------------------------


class Run:
    """Paths of one run: carried sources and state under ``root``, the checkpoint under ``weights``."""

    def __init__(self, root: Path, weights: Path, options: argparse.Namespace) -> None:
        self.root = root
        self.weights = weights
        self.options = options
        self.out = root / "outputs"
        self.state = root / "state"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state.mkdir(parents=True, exist_ok=True)

    @property
    def snapshot(self) -> Path:
        return self.weights / MODEL_KEY

    @property
    def arch_dir(self) -> Path:
        return self.root / "third_party" / "nafnet"

    def write_state(self, name: str, value: Any) -> Path:
        path = self.state / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path

    def read_state(self, name: str, needed_by: str) -> Any:
        path = self.state / name
        if not path.is_file():
            raise RuntimeError(f"{name} is missing: run the stage that writes it before '{needed_by}' (run the notebook from the top)")
        return json.loads(path.read_text(encoding="utf-8"))

    def write_output(self, name: str, value: Any) -> Path:
        path = self.out / name
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        return path


def rounded(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, Mapping):
        return {k: rounded(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [rounded(v, digits) for v in value]
    return value


def summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {"method": report["method"], "n": report["n"], "psnr_db": round(report["psnr"], 2), "ssim": round(report["ssim"], 4)}


def save_png(array: Any, path: Path) -> Path:
    from PIL import Image

    Image.fromarray(array).save(path)
    return path


def panel_sheet(rows: Sequence[Sequence[Any]], labels: Sequence[str], path: Path, *, tile: int = 192, row_titles: Sequence[str] | None = None) -> Path:
    """A labelled grid: one row per record, one column per method (uint8 RGB arrays)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=13)
    header, side = 22, 0 if row_titles is None else 120
    sheet = Image.new("RGB", (side + tile * len(labels), header + tile * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, label in enumerate(labels):
        draw.text((side + j * tile + 4, 4), label, fill="black", font=font)
    for i, row in enumerate(rows):
        if row_titles is not None:
            draw.text((4, header + i * tile + tile // 2), row_titles[i][:18], fill="black", font=font)
        for j, array in enumerate(row):
            if array is None:
                continue
            image = Image.fromarray(array)
            image.thumbnail((tile, tile))
            sheet.paste(image, (side + j * tile, header + i * tile))
    sheet.save(path)
    return path


def device() -> str:
    from nafnet_deblurring_pipeline.pipeline import default_device

    return default_device()


# --------------------------------------------------------------------------------------------------
# model and data factories (the CPU pre-flight test replaces these with tiny stand-ins)
# --------------------------------------------------------------------------------------------------


def stage_checkpoint(run: Run) -> dict[str, Any]:
    """Install the carried manifest, fetch the checkpoint if absent, verify it, convert it to safetensors."""
    from nafnet_deblurring_pipeline import pipeline as P

    carried = run.root / "weights" / MODEL_KEY / P.MANIFEST_NAME
    manifest = P.load_manifest(carried)
    run.snapshot.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(carried, run.snapshot / P.MANIFEST_NAME)
    print({"model_id": manifest["modelId"], "variant": manifest["variant"], "revision": manifest["revision"], "license": manifest["license"], "file": P.CHECKPOINT_FILE, "bytes": P.CHECKPOINT_BYTES, "sha256": P.CHECKPOINT_SHA256[:16] + "…"}, flush=True)
    fetched = P.fetch_checkpoint(run.snapshot, allow_download=True)
    print({"checkpoint_verified": True, "fetched_on_this_run": fetched["fetched"], "source": fetched["source"]}, flush=True)
    converted = P.convert_checkpoint(run.snapshot / P.CHECKPOINT_FILE, run.snapshot / P.BASE_SAFETENSORS, arch_dir=run.arch_dir, device=device())
    return {"checkpoint": fetched, "base": converted, "parameters": P.PARAMETER_COUNT, "distribution_source": manifest["distribution_source"], "upstream_source": manifest["upstream_source"]}


def load_base(run: Run, stage: str) -> Any:
    from nafnet_deblurring_pipeline import DeblurPipeline

    base = run.read_state("base.json", stage)
    return DeblurPipeline.from_base(Path(base["path"]), expected_sha256=base["sha256"], device=device(), arch_dir=run.arch_dir)


def load_artifact(run: Run, directory: Path) -> Any:
    from nafnet_deblurring_pipeline import DeblurPipeline

    return DeblurPipeline.from_artifact(directory, device=device(), arch_dir=run.arch_dir)


def load_sample(run: Run) -> dict[str, list[dict[str, Any]]]:
    from nafnet_deblurring_pipeline import build_sample_dataset

    return build_sample_dataset()


def load_new_data(run: Run) -> list[dict[str, Any]]:
    from nafnet_deblurring_pipeline import build_new_data

    return build_new_data()


def load_data(run: Run, stage: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Rebuild the deterministic sample and refuse if it differs from what `prepare` recorded."""
    from nafnet_deblurring_pipeline import dataset_manifest

    data = run.read_state("data.json", stage)
    splits = load_sample(run)
    digest = dataset_manifest(splits)["digest"]
    if digest != data["dataset_digest"]:
        raise RuntimeError(f"the sample changed since 'prepare' (digest {digest[:16]}… != {data['dataset_digest'][:16]}…); re-run from Section 4")
    return splits, data


# --------------------------------------------------------------------------------------------------
# shared building blocks (used by the canonical stages and by BYOD, DAT13)
# --------------------------------------------------------------------------------------------------


def run_baselines(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]], pretrained: Any) -> dict[str, Any]:
    """Classical baselines tuned on train and frozen, then the pretrained network, all scored on test."""
    from nafnet_deblurring_pipeline import classical_outputs, score, tune_unsharp, tune_wiener

    unsharp, wiener = tune_unsharp(train), tune_wiener(train)
    print({"unsharp_chosen_on_train": unsharp["chosen"], "wiener_oracle_chosen_on_train": None if wiener is None else wiener["chosen"]}, flush=True)
    outputs = classical_outputs(test, unsharp, wiener)
    started = time.perf_counter()
    outputs["pretrained"] = pretrained.restore([r["blurred"] for r in test])
    seconds = round(time.perf_counter() - started, 2)
    reports = {name: score(out, test, method=name) for name, out in outputs.items()}
    for report in reports.values():
        print(summary(report), flush=True)
    return {"tuning": {"unsharp": unsharp, "wiener_oracle": wiener}, "reports": reports, "outputs": outputs, "pretrained_seconds": seconds}


def adapt_model(pipe: Any, train: Sequence[Mapping[str, Any]], options: argparse.Namespace, scope: str) -> dict[str, Any]:
    patch = min(256, min(min(r["blurred"].shape[:2]) for r in train))
    patch -= patch % 16
    return pipe.finetune(train, steps=options.steps, lr=options.lr, batch_size=options.batch_size, patch=patch, scope=scope, seed=options.seed)


def reference_outputs(pipe: Any, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    return {r["id"]: np.asarray(pipe.restore_float(r["blurred"]), dtype=np.float32) for r in records[:REFERENCE_RECORDS]}


def reload_parity(pipe: Any, records: Sequence[Mapping[str, Any]], reference: Mapping[str, Any]) -> dict[str, Any]:
    """VER1-VER5: float outputs of a reloaded artifact against outputs recorded from the trained in-memory model."""
    import numpy as np

    by_id = {r["id"]: r for r in records}
    diffs, equal = [], []
    for record_id, expected in reference.items():
        got = pipe.restore_float(by_id[record_id]["blurred"])
        diffs.append(float(np.abs(got - expected).max()))
        equal.append(float((np.round(got * 255) == np.round(expected * 255)).mean()))
    parity = {"records_compared": len(diffs), "max_abs_float_diff": max(diffs), "uint8_equal_fraction": min(equal), "tolerance": RELOAD_TOLERANCE}
    if not parity["max_abs_float_diff"] <= RELOAD_TOLERANCE:
        raise RuntimeError(f"reloaded artifact differs from the trained model by {parity['max_abs_float_diff']:.3g} > {RELOAD_TOLERANCE}; the artifact does not reproduce the adapted model")
    return parity


def write_metrics_csv(path: Path, reports: Mapping[str, Mapping[str, Any]]) -> Path:
    methods = list(reports)
    ids = [row["id"] for row in next(iter(reports.values()))["per_record"]]
    rows = {m: {row["id"]: row for row in reports[m]["per_record"]} for m in methods}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "source", *[f"{m}_psnr_db" for m in methods], *[f"{m}_ssim" for m in methods]])
        for record_id in ids:
            first = rows[methods[0]][record_id]
            writer.writerow([record_id, first["source"], *[round(rows[m][record_id]["psnr"], 4) for m in methods], *[round(rows[m][record_id]["ssim"], 5) for m in methods]])
    return path


# --------------------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------------------


def stage_weights(run: Run) -> None:
    """Section 3: install the carried manifest, fetch and verify the pinned checkpoint, convert it to safetensors."""
    record = stage_checkpoint(run)
    base = record["base"]
    print({"converted_to": base["file"], "bytes": base["bytes"], "sha256": base["sha256"], "tensors": base["tensors"], "parameters": base["parameters"], "max_abs_output_diff_vs_pth": base["max_abs_output_diff"]}, flush=True)
    run.write_state("base.json", {"path": str(run.snapshot / base["file"]), "sha256": base["sha256"], "bytes": base["bytes"]})
    run.write_output("weights.json", record)


def stage_prepare(run: Run) -> None:
    """Section 4: build the pinned synthetic-blur sample, validate it, split it by photograph, probe the refusals."""
    import numpy as np

    from nafnet_deblurring_pipeline import INPUT_SCHEMA, SAMPLE_IMAGES, dataset_manifest, validate_records

    splits = load_sample(run)
    report = validate_records(splits["train"] + splits["test"])
    manifest = dataset_manifest(splits)
    print({"data_source": "scikit-image 0.26.0 sample photographs (public domain / CC0) + seeded synthetic motion blur", "train_pairs": len(splits["train"]), "test_pairs": len(splits["test"]), "crop_px": splits["train"][0]["blurred"].shape[0], "disjoint_sources": manifest["disjoint_sources"]}, flush=True)
    print({"train_sources": manifest["sources"]["train"], "test_sources": manifest["sources"]["test"]}, flush=True)
    lengths = [r["kernel"]["length_px"] for r in splits["train"] + splits["test"]]
    print({"blur_length_px_range": [min(lengths), max(lengths)], "noise_sigma": splits["train"][0]["noise_sigma"], "validation": {k: report[k] for k in ("records", "sources", "width_range", "height_range", "ceilings")}}, flush=True)
    print({"schema": INPUT_SCHEMA["record"], "validation_scope": INPUT_SCHEMA["validation"]}, flush=True)
    good = splits["train"][0]
    probes = {
        "pair sizes differ": [dict(good, id="probe-a", sharp=good["sharp"][:-8]), *splits["train"][1:4]],
        "image too small": [dict(good, id="probe-b", blurred=good["blurred"][:40, :40], sharp=good["sharp"][:40, :40]), *splits["train"][1:4]],
        "duplicate id": [good, dict(splits["train"][1], id=good["id"]), *splits["train"][2:4]],
        "single source": [r for r in splits["train"] if r["source"] == good["source"]][:4],
        "greyscale array": [dict(good, id="probe-c", blurred=good["blurred"][..., 0]), *splits["train"][1:4]],
    }
    refusals = {}
    for label, records in probes.items():
        try:
            validate_records(records)
            refusals[label] = "ACCEPTED (unexpected)"
        except ValueError as exc:
            refusals[label] = f"rejected: {exc}"
        print({"refusal_probe": label, "result": refusals[label]}, flush=True)
    if any(v.startswith("ACCEPTED") for v in refusals.values()):
        raise RuntimeError(f"a refusal probe was accepted: {refusals}")
    with (run.out / f"{STEM}_sample_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "split", "source", "blur_length_px", "blur_angle_deg", "noise_sigma", "window_xywh", "source_license"])
        for r in splits["train"] + splits["test"]:
            writer.writerow([r["id"], r["split"], r["source"], r["kernel"]["length_px"], r["kernel"]["angle_deg"], r["noise_sigma"], r.get("window"), SAMPLE_IMAGES.get(r["source"], {}).get("license", "unknown")])
    show = [splits["train"][0], splits["train"][len(splits["train"]) // 2], splits["test"][0], splits["test"][len(splits["test"]) // 2]]
    sheet = panel_sheet([[r["blurred"], r["sharp"], np.repeat(np.repeat((r["kernel_array"] / r["kernel_array"].max() * 255).astype(np.uint8)[..., None], 3, axis=2), 8, axis=0).repeat(8, axis=1)] for r in show], ["blurred input", "sharp reference", "blur kernel (x8)"], run.out / f"{STEM}_sample_pairs.png", row_titles=[f"{r['split']}: {r['id']}" for r in show])
    print({"sample_sheet": str(sheet), "pairs_csv": str(run.out / f"{STEM}_sample_pairs.csv")}, flush=True)
    run.write_state("data.json", {"source": "sample", "dataset_digest": manifest["digest"], "counts": manifest["counts"]})
    run.write_output("dataset.json", {"manifest": manifest, "validation": report, "refusal_probes": refusals})


def stage_baseline(run: Run) -> None:
    """Section 5: identity, unsharp masking and oracle Wiener (tuned on train), then the pretrained network, on test."""
    splits, _data = load_data(run, "baseline")
    pretrained = load_base(run, "baseline")
    result = run_baselines(splits["train"], splits["test"], pretrained)
    reports = result["reports"]
    from nafnet_deblurring_pipeline import METRIC_DEFINITIONS, paired_difference

    gain = paired_difference(reports["identity"], reports["pretrained"])
    print({"pretrained_vs_blurred_input": rounded(gain, 3)}, flush=True)
    show = splits["test"][::4]
    idx = [splits["test"].index(r) for r in show]
    methods = [m for m in ("identity", "unsharp", "wiener_oracle", "pretrained") if m in result["outputs"]]
    sheet = panel_sheet([[result["outputs"][m][i] for m in methods] + [splits["test"][i]["sharp"]] for i in idx], [*methods, "sharp"], run.out / f"{STEM}_baselines.png", row_titles=[splits["test"][i]["id"] for i in idx])
    print({"comparison_sheet": str(sheet), "pretrained_inference_seconds_for_test_split": result["pretrained_seconds"]}, flush=True)
    record = {"reports": reports, "tuning": result["tuning"], "pretrained_vs_identity": gain, "definitions": METRIC_DEFINITIONS, "split": "test", "selection_split": "train"}
    run.write_output(f"{STEM}_baseline_report.json", record)


def stage_adapt(run: Run) -> None:
    """Section 6: bounded fine-tuning from the verified base, in-memory reference outputs, artifact export."""
    import numpy as np

    from nafnet_deblurring_pipeline import pipeline as P

    splits, data = load_data(run, "adapt")
    pipe = load_base(run, "adapt")  # every run of this stage starts from the verified pretrained weights
    opts = run.options
    print({"adaptation": "gradient fine-tuning", "scope": opts.scope, "trainable_modules": list(P.TRAINABLE_SCOPES[opts.scope] or ("all",)), "steps": opts.steps, "lr": opts.lr, "batch_size": opts.batch_size, "seed": opts.seed, "device": pipe.device}, flush=True)
    adaptation = adapt_model(pipe, splits["train"], opts, opts.scope)
    print({k: adaptation[k] for k in ("trainable_parameters", "frozen_parameters", "total_parameters", "patch", "seconds")}, flush=True)
    reference = reference_outputs(pipe, splits["test"])
    np.savez(run.state / "reference_outputs.npz", **reference)
    artifact_dir = run.out / f"{STEM}_artifact"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    manifest = pipe.save_artifact(artifact_dir, metadata={"data": {"source": data["source"], "dataset_digest": data["dataset_digest"], "counts": data["counts"]}, "selection": "none: fixed steps chosen before training; no validation-based selection, the test split is never used for a decision"})
    print({"artifact": str(artifact_dir), "files": manifest["files"], "weights_bytes": manifest["weights"]["bytes"], "weights_sha256": manifest["weights"]["sha256"]}, flush=True)
    run.write_state("adapted.json", {"artifact": str(artifact_dir), "sha256": manifest["weights"]["sha256"], "reference_records": list(reference)})
    run.write_output(f"{STEM}_adaptation.json", adaptation)


def stage_evaluate(run: Run) -> None:
    """Section 7: a fresh process loads the exported artifact and scores it beside every baseline on the same test split."""
    from nafnet_deblurring_pipeline import METRIC_DEFINITIONS, paired_difference, score

    splits, _data = load_data(run, "evaluate")
    adapted_state = run.read_state("adapted.json", "evaluate")
    baseline = json.loads((run.out / f"{STEM}_baseline_report.json").read_text(encoding="utf-8"))
    pipe = load_artifact(run, Path(adapted_state["artifact"]))
    outputs = pipe.restore([r["blurred"] for r in splits["test"]])
    adapted = score(outputs, splits["test"], method="adapted")
    reports = {**baseline["reports"], "adapted": adapted}
    table = {name: {"psnr_db": round(r["psnr"], 2), "ssim": round(r["ssim"], 4)} for name, r in reports.items()}
    for name, row in table.items():
        print({"method": name, **row}, flush=True)
    gains = {"adapted_vs_pretrained": paired_difference(reports["pretrained"], adapted), "adapted_vs_identity": paired_difference(reports["identity"], adapted)}
    if "wiener_oracle" in reports:
        gains["adapted_vs_wiener_oracle"] = paired_difference(reports["wiener_oracle"], adapted)
    for key, gain in gains.items():
        print({key: rounded(gain, 3)}, flush=True)
    per_source = {}
    for source in sorted({r["source"] for r in splits["test"]}):
        rows = [(p, a) for p, a in zip(reports["pretrained"]["per_record"], adapted["per_record"], strict=True) if p["source"] == source]
        per_source[source] = {"n": len(rows), "pretrained_psnr_db": round(sum(p["psnr"] for p, _ in rows) / len(rows), 2), "adapted_psnr_db": round(sum(a["psnr"] for _, a in rows) / len(rows), 2)}
    print({"per_test_photograph": per_source}, flush=True)
    write_metrics_csv(run.out / f"{STEM}_test_metrics.csv", reports)
    from nafnet_deblurring_pipeline import classical_outputs

    tuning = baseline["tuning"]
    classical = classical_outputs(splits["test"], tuning["unsharp"], tuning["wiener_oracle"])
    pretrained_pipe = load_base(run, "evaluate")
    idx = list(range(0, len(splits["test"]), 4))
    pre = pretrained_pipe.restore([splits["test"][i]["blurred"] for i in idx])
    cols = [m for m in ("identity", "unsharp", "wiener_oracle") if m in classical]
    rows = [[classical[m][i] for m in cols] + [pre[k], outputs[i], splits["test"][i]["sharp"]] for k, i in enumerate(idx)]
    sheet = panel_sheet(rows, [*cols, "pretrained", "adapted", "sharp"], run.out / f"{STEM}_evaluation.png", row_titles=[splits["test"][i]["id"] for i in idx])
    print({"comparison_sheet": str(sheet), "metrics_csv": str(run.out / f"{STEM}_test_metrics.csv")}, flush=True)
    run.write_output(f"{STEM}_evaluation_report.json", {"split": "test (held out by photograph)", "estimation": "single held-out split, one seeded run; no dispersion estimate", "reports": reports, "table": table, "paired_gains": gains, "per_test_photograph": per_source, "definitions": METRIC_DEFINITIONS, "artifact_sha256": adapted_state["sha256"]})


def stage_reload(run: Run) -> None:
    """Section 8: a second fresh process reloads the artifact, checks equivalence, then deblurs two new images."""
    import numpy as np

    from nafnet_deblurring_pipeline import gradient_energy, psnr, ssim
    from nafnet_deblurring_pipeline.pipeline import CHECKPOINT_SHA256, MODEL_ID, MODEL_REVISION, MODEL_VARIANT, runtime_versions

    splits, data = load_data(run, "reload")
    adapted_state = run.read_state("adapted.json", "reload")
    reloaded = load_artifact(run, Path(adapted_state["artifact"]))
    print({"artifact_manifest_verified": True, "format": reloaded.manifest["format"], "base_checkpoint_sha256": reloaded.manifest["base"]["checkpoint_sha256"][:16] + "…", "weights_sha256": reloaded.manifest["weights"]["sha256"][:16] + "…"}, flush=True)
    reference = dict(np.load(run.state / "reference_outputs.npz"))
    parity = reload_parity(reloaded, splits["test"], reference)
    print({"reload_parity": rounded(parity, 8)}, flush=True)
    pretrained = load_base(run, "reload")
    rows, csv_rows = [], []
    for record in load_new_data(run):
        pre, ada = pretrained.restore([record["blurred"]])[0], reloaded.restore([record["blurred"]])[0]
        save_png(ada, run.out / f"{STEM}_new_{record['id']}_adapted.png")
        save_png(pre, run.out / f"{STEM}_new_{record['id']}_pretrained.png")
        row = {"id": record["id"], "kind": "paired (synthetic blur, fixed kernel)" if record["sharp"] is not None else "unpaired (real camera motion, no reference)", "size": list(record["blurred"].shape[:2])}
        if record["sharp"] is not None:
            for name, out in (("blurred_input", record["blurred"]), ("pretrained", pre), ("adapted", ada)):
                row[f"{name}_psnr_db"], row[f"{name}_ssim"] = round(psnr(out, record["sharp"]), 2), round(ssim(out, record["sharp"]), 4)
        else:
            base_energy = gradient_energy(record["blurred"])
            row["gradient_energy_ratio_pretrained"] = round(gradient_energy(pre) / base_energy, 3)
            row["gradient_energy_ratio_adapted"] = round(gradient_energy(ada) / base_energy, 3)
            row["note"] = "no sharp reference: no PSNR/SSIM; the gradient-energy ratio is a sharpness proxy that noise and ringing also raise, not a quality score"
        print(row, flush=True)
        csv_rows.append(row)
        rows.append([record["blurred"], pre, ada, record["sharp"]])
    sheet = panel_sheet(rows, ["blurred input", "pretrained", "adapted", "sharp (if known)"], run.out / f"{STEM}_new_images.png", tile=256, row_titles=[r["id"] for r in csv_rows])
    keys = sorted({k for r in csv_rows for k in r})
    with (run.out / f"{STEM}_new_image_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    evaluation = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    weights = json.loads((run.out / "weights.json").read_text(encoding="utf-8"))
    result = {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "variant": MODEL_VARIANT, "checkpoint_sha256": CHECKPOINT_SHA256, "base_safetensors_sha256": weights["base"]["sha256"], "checkpoint_source": weights["checkpoint"]["source"]},
        "runtime": runtime_versions(),
        "data": data,
        "adaptation": reloaded.adaptation,
        "artifact": {"dir": adapted_state["artifact"], "weights_sha256": adapted_state["sha256"], "format": reloaded.manifest["format"]},
        "evaluation": evaluation["table"],
        "paired_gains": evaluation["paired_gains"],
        "reload_parity": parity,
        "new_images": csv_rows,
        "evidence_class": "tutorial/sanity evidence on synthetic motion blur; not a benchmark",
    }
    run.write_output(f"{STEM}_result.json", result)
    print({"new_image_sheet": str(sheet)}, flush=True)
    print({"outputs": sorted(p.name for p in run.out.iterdir())}, flush=True)


def stage_activity(run: Run) -> None:
    """Optional: re-run the bounded fine-tuning from the verified base with another trainable scope; compare on test."""
    from nafnet_deblurring_pipeline import score

    splits, _data = load_data(run, "activity")
    default = json.loads((run.out / f"{STEM}_adaptation.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run.out / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    opts = run.options
    if opts.scope == default["scope"]:
        print({"note": f"the activity scope equals the default scope {default['scope']!r}; choose another to change one thing"}, flush=True)
    pipe = load_base(run, "activity")  # starts from the verified pretrained weights, never from the adapted model
    opts.steps, opts.lr, opts.batch_size, opts.seed = default["steps"], default["lr"], default["batch_size"], default["seed"]
    changed = adapt_model(pipe, splits["train"], opts, opts.scope)
    report = score(pipe.restore([r["blurred"] for r in splits["test"]]), splits["test"], method=f"adapted ({opts.scope})")
    comparison = {
        "pretrained": evaluation["table"]["pretrained"],
        f"adapted ({default['scope']}, default)": {**evaluation["table"]["adapted"], "trainable_parameters": default["trainable_parameters"], "seconds": default["seconds"], "first_logged_loss": default["history"][0]["loss"]},
        f"adapted ({opts.scope}, activity)": {"psnr_db": round(report["psnr"], 2), "ssim": round(report["ssim"], 4), "trainable_parameters": changed["trainable_parameters"], "seconds": changed["seconds"], "first_logged_loss": changed["history"][0]["loss"]},
    }
    for name, row in comparison.items():
        print({name: row}, flush=True)
    run.write_output(f"{STEM}_activity.json", {"changed": {"scope": [default["scope"], opts.scope]}, "held_fixed": {k: default[k] for k in ("steps", "lr", "batch_size", "seed", "patch")}, "comparison": comparison})


def stage_byod(run: Run) -> None:
    """Optional BYOD. Paired: validate → split → baselines → adapt → evaluate → export → reload → infer. Unpaired:
    validate → infer with the pretrained model (and the canonical adapted artifact when one exists)."""
    import numpy as np

    from nafnet_deblurring_pipeline import (
        METRIC_DEFINITIONS,
        dataset_manifest,
        gradient_energy,
        paired_difference,
        read_byod,
        score,
        split_by_source,
    )

    source = Path(run.options.byod)
    byod = read_byod(source)
    print({"byod_mode": byod["mode"], "records": len(byod["records"]), "validation": {k: byod["validation"][k] for k in ("records", "sources", "width_range", "height_range")}, "conversions": byod["conversions"][:10], "ignored_files": byod["ignored_files"][:10]}, flush=True)
    # one directory per archive, so a second BYOD run never overwrites or mixes with an earlier one
    out = run.out / "byod" / f"{byod['mode']}-{byod['archive_digest'][:12]}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    result_name = f"byod/{out.name}/byod_result.json"
    pretrained = load_base(run, "byod")
    result: dict[str, Any] = {"mode": byod["mode"], "archive_digest": byod["archive_digest"], "validation": byod["validation"], "conversions": byod["conversions"]}
    if byod["mode"] == "unpaired":
        adapted = None
        state = run.state / "adapted.json"
        if state.is_file():
            adapted = load_artifact(run, Path(json.loads(state.read_text(encoding="utf-8"))["artifact"]))
        rows = []
        for record in byod["records"]:
            row = {"id": record["id"], "size": list(record["blurred"].shape[:2])}
            energy = gradient_energy(record["blurred"])
            for name, pipe in (("pretrained", pretrained), ("adapted", adapted)):
                if pipe is None:
                    continue
                restored = pipe.restore([record["blurred"]])[0]
                save_png(restored, out / f"{record['id']}_{name}.png")
                row[f"gradient_energy_ratio_{name}"] = round(gradient_energy(restored) / energy, 3)
            print(row, flush=True)
            rows.append(row)
        result.update({"predictions": rows, "note": "unpaired images have no reference: no PSNR/SSIM is computed; the gradient-energy ratio is a sharpness proxy, not a quality score", "adapted_model": "canonical sample artifact" if adapted is not None else "none (run Section 6 first to include it)"})
        run.write_output(result_name, result)
        print({"byod_outputs": sorted(p.name for p in out.iterdir())}, flush=True)
        return
    splits = split_by_source(byod["records"], seed=SPLIT_SEED)
    manifest = dataset_manifest(splits)
    print({"split": manifest["counts"], "train_sources": manifest["sources"]["train"][:10], "test_sources": manifest["sources"]["test"][:10], "disjoint_sources": True, "note": "random split by image: assumes the images are independent (SPL3); pre-split folders are not supported"}, flush=True)
    base = run_baselines(splits["train"], splits["test"], pretrained)
    pipe = load_base(run, "byod")
    adaptation = adapt_model(pipe, splits["train"], run.options, run.options.scope)
    reference = reference_outputs(pipe, splits["test"])
    artifact_dir = out / "artifact"
    artifact = pipe.save_artifact(artifact_dir, metadata={"data": {"source": "byod", "dataset_digest": manifest["digest"], "archive_digest": byod["archive_digest"], "counts": manifest["counts"]}, "selection": "none: fixed steps; the test split is never used for a decision"})
    reloaded = load_artifact(run, artifact_dir)
    parity = reload_parity(reloaded, splits["test"], reference)
    outputs = reloaded.restore([r["blurred"] for r in splits["test"]])
    reports = {**base["reports"], "adapted": score(outputs, splits["test"], method="adapted")}
    print(summary(reports["adapted"]), flush=True)
    gains = {"adapted_vs_pretrained": paired_difference(reports["pretrained"], reports["adapted"]), "adapted_vs_identity": paired_difference(reports["identity"], reports["adapted"])}
    print({k: rounded(v, 3) for k, v in gains.items()}, flush=True)
    print({"reload_parity": rounded(parity, 8)}, flush=True)
    for record, restored in zip(splits["test"], outputs, strict=True):
        save_png(restored, out / f"{record['id']}_adapted.png")
    write_metrics_csv(out / "byod_test_metrics.csv", reports)
    result.update({"split": {k: [r["id"] for r in v] for k, v in splits.items()}, "dataset_digest": manifest["digest"], "tuning": base["tuning"], "adaptation": adaptation, "artifact": {"dir": str(artifact_dir), "weights_sha256": artifact["weights"]["sha256"]}, "reports": reports, "paired_gains": gains, "reload_parity": parity, "definitions": METRIC_DEFINITIONS})
    run.write_output(result_name, json.loads(json.dumps(result, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))))
    print({"byod_outputs": sorted(p.name for p in out.iterdir())}, flush=True)


STAGES = {
    "weights": stage_weights,
    "prepare": stage_prepare,
    "baseline": stage_baseline,
    "adapt": stage_adapt,
    "evaluate": stage_evaluate,
    "reload": stage_reload,
    "activity": stage_activity,
    "byod": stage_byod,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="run directory holding the carried sources")
    parser.add_argument("--weights", type=Path, required=True, help="directory holding the verified checkpoint")
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--byod", default="", help="byod: a zip or directory")
    parser.add_argument("--steps", type=int, default=300, help="adapt/byod: optimiser steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="adapt/byod: AdamW learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="adapt/byod: crops per optimiser step")
    parser.add_argument("--scope", default="decoder", help="adapt/activity/byod: trainable scope (decoder, decoder+middle, all)")
    parser.add_argument("--seed", type=int, default=0, help="adapt/byod: training seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    root = options.root.resolve()
    carried_src = root / "src"
    if carried_src.is_dir() and str(carried_src) not in sys.path:
        sys.path.insert(0, str(carried_src))
    run = Run(root, options.weights.resolve(), options)
    error_file = run.state / f"{options.stage}.error.json"
    error_file.unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        STAGES[options.stage](run)
    except Exception as exc:  # the notebook re-raises this message in the kernel
        traceback.print_exc()
        message = str(exc) or repr(exc)
        error_file.write_text(json.dumps({"stage": options.stage, "type": type(exc).__name__, "message": message}), encoding="utf-8")
        print(f"STAGE FAILED ({options.stage}): {type(exc).__name__}: {message}", flush=True)
        return 2
    print({"stage": options.stage, "status": "ok", "seconds": round(time.perf_counter() - started, 1)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
