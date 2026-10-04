"""Per-repository template for tools/build_notebook.py /3 (NOTEBOOK_SPEC 2.2 §4 standalone, §25.13 isolated environment).

The generator writes the infrastructure cells (runtime check, carrier, isolated install + stage runner, checkpoint
staging) from repository files; this template holds the learner-facing prose, the list of carried files and the
learner cells. Every learner cell calls ``run_stage(...)``: the carried ``tutorial_stages.py`` (``tools/`` in the
repository) runs one stage per process in an isolated, hash-locked environment, so nothing is installed into the
notebook kernel.

This template configures an E2E image-restoration workflow: the digest-pinned NAFNet-GoPro-width32 checkpoint is
fetched, verified, unpickled once with ``weights_only=True`` and converted to safetensors; a synthetic motion-blur
sample is built from public-domain photographs that ship inside a hash-locked wheel, validated and split by
photograph; classical baselines (tuned on the training split) and the pretrained network are scored with PSNR/SSIM on
the held-out split; the decoder half is fine-tuned for a bounded number of steps; the exported safetensors artifact is
scored in a fresh process, reloaded in a second fresh process with an equivalence check, and used on two new images
(one synthetic, one real camera-motion photograph without a reference).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "nafnet-deblurring-pipeline"
NOTEBOOK = "nafnet_deblurring_colab.ipynb"

BADGES = [
    ("GitHub", "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white", f"https://github.com/kurtvalcorza/{REPO}"),
    ("Open In Colab", "https://colab.research.google.com/assets/colab-badge.svg", f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/{NOTEBOOK}"),
    ("Upstream", "https://img.shields.io/badge/Upstream-megvii--research%2FNAFNet-181717?style=flat&logo=github&logoColor=white", "https://github.com/megvii-research/NAFNet"),
    ("License", "https://img.shields.io/badge/License-MIT%20%2B%20Apache--2.0-blue.svg", "https://github.com/megvii-research/NAFNet/blob/main/LICENSE"),
]

TEMPLATE = {
    "package": "nafnet_deblurring_pipeline",
    "repo_name": REPO,
    "stem": "nafnet_deblurring",
    "notebook_name": NOTEBOOK,
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh Linux runtime (a Colab or Kaggle **T4 GPU** is the documented runtime; a CPU-only runtime also "
        "completes, more slowly) builds an isolated Python environment from the carried hash-locked requirements (torch, numpy, pillow, "
        "safetensors, scikit-image and their dependencies) without touching the notebook kernel's own packages, then runs each stage below "
        "in its own process: it fetches the 68.7 MB NAFNet-GoPro-width32 checkpoint from a Hugging Face mirror and refuses it unless its size "
        "and SHA-256 equal the pinned values, unpickles it once with `weights_only=True` and converts it to safetensors; builds a sample of 60 "
        "blurred/sharp pairs from eleven public-domain photographs with seeded synthetic motion blur, validates it and splits it by photograph "
        "(48 training / 12 test pairs); scores the blurred input, unsharp masking, oracle Wiener deconvolution and the pretrained network on "
        "the test pairs with PSNR and SSIM; fine-tunes the decoder half of the network for 300 steps; scores the exported safetensors artifact "
        "in a fresh process beside every baseline; reloads it in a second fresh process and checks that it reproduces the trained model; and "
        "deblurs two new images. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog, no "
        "configuration edit and no runtime restart (NOTEBOOK_SPEC 2.2 §5). Its duration on a T4 has not been recorded yet; see the Prerequisites "
        "for what has been measured."
    ),
    "byod": (
        "After the canonical path completes, set `USE_BYOD = True` in Section 10 to run the same stages on your own images, supplied as a zip "
        "(or a directory already in the runtime, via `BYOD_PATH`). With `sharp/` and `blurred/` folders whose files share a name (paired data, "
        "at least four pairs from different scenes) the BYOD branch validates, splits by image, scores the same baselines, fine-tunes from the "
        "pretrained weights, evaluates, exports and reloads its own artifact and writes PNGs, a metrics CSV and a JSON record. With blurred "
        "images only (unpaired data) it validates them and deblurs them with the pretrained and the adapted model; no PSNR or SSIM is possible "
        "without a reference. The schema, ceilings and privacy guidance are stated in the Prerequisites and in Section 10; uploaded files stay "
        "inside this runtime. BYOD is optional and never part of the default path."
    ),
    "weights_key": "nafnet-gopro-width32",
    # Carried byte for byte (UTF-8 text, LF newlines) into the run directory and verified against CARRIED_HASHES.
    "carried": {
        "src/nafnet_deblurring_pipeline/__init__.py": "src/nafnet_deblurring_pipeline/__init__.py",
        "src/nafnet_deblurring_pipeline/pipeline.py": "src/nafnet_deblurring_pipeline/pipeline.py",
        "src/nafnet_deblurring_pipeline/samples.py": "src/nafnet_deblurring_pipeline/samples.py",
        "src/nafnet_deblurring_pipeline/metrics.py": "src/nafnet_deblurring_pipeline/metrics.py",
        "third_party/nafnet/basicsr/models/archs/NAFNet_arch.py": "third_party/nafnet/basicsr/models/archs/NAFNet_arch.py",
        "third_party/nafnet/basicsr/models/archs/arch_util.py": "third_party/nafnet/basicsr/models/archs/arch_util.py",
        "third_party/nafnet/basicsr/models/archs/local_arch.py": "third_party/nafnet/basicsr/models/archs/local_arch.py",
        "third_party/nafnet/LICENSE": "third_party/nafnet/LICENSE",
        "tutorial_stages.py": "tools/tutorial_stages.py",
        "requirements.txt": "tutorials/requirements-colab.lock.txt",
        "weights/nafnet-gopro-width32/dimer-base-manifest.json": "weights/nafnet-gopro-width32/dimer-base-manifest.json",
        "LICENSE": "LICENSE",
    },
    "stage_runner": "tutorial_stages.py",
    "lock": "requirements.txt",
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "disk_gib": {"weights": 0.2, "environment": 9},
    "runtime_modules": ["torch", "numpy", "safetensors"],
    "title": "NAFNet-GoPro-width32 — DIMER guided notebook: image deblurring and restoration fine-tuning (standalone)",
    "badges": BADGES,
    "capability": "single-image deblurring with a 17.1 M-parameter NAFNet (GoPro-trained, width 32), PSNR/SSIM evaluation against classical baselines, and bounded restoration fine-tuning to a new blur distribution",
    "intro": (
        "NAFNet (Chen et al., 2022) is a U-shaped convolutional network for image restoration built without nonlinear activation functions: "
        "each block uses layer normalisation, depthwise convolution, a *SimpleGate* (one half of the channels multiplies the other) and a "
        "simplified channel attention. The checkpoint used here was trained by the authors on the GoPro dataset, whose blurred images are "
        "averages of consecutive frames from a moving handheld camera. The network takes a blurred RGB image and returns a restored image of the "
        "same size; it predicts a residual that is added to its input.\n\n"
        "Two properties shape this notebook. **Restoration has a reference only when you make the blur yourself**, so the sample degrades "
        "sharp photographs with known synthetic motion blur and the metrics compare against the sharp originals; a real blurred photograph "
        "(Section 8) has no reference and gets no metric. **A model trained on one blur distribution meets another here**: GoPro blur comes "
        "from real camera motion, the sample's blur from straight-line kernels plus noise, and the bounded fine-tuning in Section 6 adapts the "
        "decoder half of the network to that new distribution. Every number is tutorial evidence on synthetic data, not a benchmark."
    ),
    "learning_objectives": (
        "by the end of this notebook you will be able to —\n\n"
        "1. **Explain** how a blurred/sharp pair is synthesised (motion kernel, noise, quantisation) and why the split is drawn by photograph "
        "rather than by crop (Section 4).\n"
        "2. **Diagnose** an invalid input from a validation refusal before any model runs (Section 4).\n"
        "3. **Interpret** PSNR and SSIM, and **compare** a deep network with the blurred input, a blind classical filter and a non-blind oracle "
        "(Section 5).\n"
        "4. **Identify** which parameters a bounded fine-tuning trains and which it freezes, and **explain** why the training loss is not "
        "evidence of quality (Section 6).\n"
        "5. **Compare** the adapted and the pretrained model on identical held-out pairs and **distinguish** a paired gain from a re-draw of "
        "randomness (Section 7).\n"
        "6. **Verify** that an exported artifact, reloaded from files, reproduces the trained model, and **explain** why a real blurred photograph "
        "gets no PSNR (Section 8).\n"
        "7. **Predict**, run and **explain** the effect of changing one thing, the trainable scope, in an optional activity (Section 9).\n"
        "8. **Apply** the same workflow to your own paired or unpaired images (Section 10) and **write** an evidence-based conclusion (Conclusion)."
    ),
    "exclusions": (
        "the other NAFNet checkpoints (SIDD denoising, REDS, width 64, NAFSSR stereo super-resolution), training from scratch, the GoPro "
        "benchmark itself (its numbers in the upstream README are not reproduced or quoted as notebook measurements), real-blur evaluation "
        "with ground truth, video deblurring, defocus or spatially varying blur models, perceptual metrics (LPIPS, FID) and human rating. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported **Linux x86_64** runtime — Google Colab or Kaggle with a **T4 GPU** is the documented runtime; a CPU-only runtime also completes. The kernel's own Python version does not matter: the notebook installs nothing into it and runs every stage with CPython 3.12.12 in an isolated environment built from {n_locked} hash-locked packages (torch 2.14.0, whose Linux wheel is the CUDA 13.0 build and also runs on a CPU). Everything runs in float32; cuDNN is set to deterministic kernels. About 9 GB of disk is needed for the isolated environment and 0.2 GB for the checkpoint and its converted copy.",
        "- **Measured timings (estimates for your runtime):** no hosted T4 run of this notebook has been recorded yet. On the 4-core CPU container used during development, measured while it was otherwise idle, one fine-tuning step at the defaults (batch 4, 256 × 256 crops, decoder scope) of a full-width network took about 2.3 s, so the 300-step fine-tuning would take roughly 12 minutes on a comparable CPU and the optional activity as long again; a T4 is expected to be much faster. Treat these as estimates, not measurements of your run.",
        "- **Knowledge:** basic Python and NumPy arrays (H × W × 3, uint8); what a convolution does; the idea of training by gradient descent and of a held-out test set. Image-restoration terms are explained where they are first used and collected in the glossary.",
        "- **Weights and trust boundary:** the upstream checkpoint is a PyTorch pickle (`.pth`). The notebook refuses it unless its size and SHA-256 equal the pinned values, reads it once with `torch.load(..., weights_only=True)` (tensors and plain containers only) and converts it to safetensors; every later stage loads the safetensors file. The pinned SHA-256 comes from third-party records of the upstream Google Drive file, not from the authors; `docs/WEIGHTS.md` in the repository explains that chain and its residual trust gap. No code from the Hugging Face mirror is executed: the architecture is the upstream source carried verbatim in Section 2. NAFNet is MIT-licensed; its BasicSR-derived parts are Apache-2.0.",
        "- **Data contract:** a record is `{{id, source, blurred, sharp}}` — two uint8 RGB arrays of identical size, each side 64..1024 px (larger images are refused, never resized silently); `source` groups the crops of one photograph so the split can keep them together. Greyscale is replicated to three channels and alpha is dropped, and both are reported. Validation is structural: it cannot tell whether a pair shows the same scene.",
        "- **Sample data:** eleven photographs that ship inside the hash-locked `scikit-image` 0.26.0 wheel (public domain or CC0 according to its data documentation), each pinned by size and SHA-256 and read as a file. `skimage` is never imported. Because they are well-known test images, overlap with the data behind other models cannot be ruled out; GoPro, the checkpoint's training set, consists of video frames and does not contain them as far as its description shows.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — photographs of identifiable people, medical images, licensed stock or client material are exactly that. The default path uploads nothing, and the BYOD branch keeps your files inside this runtime; nothing is sent to any service.",
    ],
    "guided": {
        "opening": [
            (
                "## How to use this notebook\n\n"
                "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab or Jupyter), run cells in order and read "
                "short Python, and who want to see how an image-restoration network is evaluated honestly and adapted to new data. No prior "
                "experience with deblurring is assumed: each term is explained where it is first needed, and the glossary below collects them. "
                "A T4 GPU is the documented runtime; a CPU-only runtime completes but the fine-tuning stages take longer. The **Prerequisites** "
                "give the details.\n\n"
                "**Running it.** In Colab, choose *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The default path needs no "
                "edit, no upload, no account, no token and no runtime restart. Sections 1–3 build an isolated environment from hash-locked "
                "packages and fetch the checkpoint, so they take the longest before any model runs; read ahead while they finish, or run the "
                "notebook one cell at a time with *Shift + Enter*.\n\n"
                "**Where the code runs.** The notebook kernel installs nothing and imports no model library. Each learner cell calls "
                "`run_stage('…')`, which runs one stage of the carried stage runner in its own process with the isolated environment's Python, "
                "streams what it prints, and stops the notebook with the stage's own error message if it fails. Stages hand results to each "
                "other only through files in the run directory — the verified checkpoint, the dataset digest, JSON records and the exported "
                "artifact — and memory is released when each stage ends.\n\n"
                "**Two kinds of cell.** *Learner cells* (Sections 4–10) are the machine-learning workflow; each runs one stage and prints compact "
                "dictionaries and a comparison image for you to read. *Infrastructure cells* (Sections 1–3: the runtime check, the carried code, "
                "the isolated install and the checkpoint staging) are collapsed and titled **Infrastructure**. You may run them without studying "
                "their implementation: they exist for reproducibility and provenance, not as prerequisite machine-learning knowledge. Open one "
                "with *Show code* if you are curious.\n\n"
                "**Form controls.** Some learner cells start with fields that Colab renders as a form: `STEPS`, `LEARNING_RATE`, `BATCH_SIZE` and "
                "`TRAINABLE_SCOPE` (Section 6); `RUN_ACTIVITY` and `ACTIVITY_SCOPE` (Section 9); and `USE_BYOD`, `BYOD_PATH` and `BYOD_STEPS` "
                "(Section 10). Leave them at their defaults for the first run: the notes and sample answers describe the default path.\n\n"
                "**Section tags.** Each numbered heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation practice]** — "
                "how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and packaging.\n\n"
                "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an expectation; "
                "after it, **What to notice** describes normal output and a collapsed **Check your reasoning** answer follows each checkpoint. "
                "Write your own answer first, then open it. Exact numbers vary with the runtime (GPU or CPU) and the library versions, so the "
                "notes describe the shape of a normal result rather than fixed values."
            ),
            (
                "## The task: Input → Model/System → Output\n\n"
                "| Stage | Input | Model / system | Output |\n"
                "|---|---|---|---|\n"
                "| **Restoration** | a blurred RGB image (uint8, H × W × 3) | NAFNet-GoPro-width32: intro convolution → 4 encoder levels → middle block → 4 decoder levels with skip connections → ending convolution, plus the input (a residual) | a restored RGB image of the same size |\n"
                "| **Adaptation** | blurred/sharp training pairs | random 256 × 256 crops through the network; the PSNR loss updates only the decoder half (upsampling layers, decoder blocks, ending convolution) | a safetensors artifact with a manifest |\n"
                "| **Evaluation** | held-out blurred/sharp pairs from photographs never used in training | the blurred input, unsharp masking, oracle Wiener deconvolution, the pretrained and the adapted network | PSNR (dB) and SSIM per pair and on average |\n\n"
                "The network never sees the blur kernel. Only the Wiener baseline is given the true kernel, as an oracle no real deblurrer has.\n\n"
                "## Roadmap\n\n"
                "| Section | Tag | What happens | What you read |\n"
                "|---|---|---|---|\n"
                "| 1. Check the runtime | [Engineering] | Linux, accelerator and disk checked; a fresh run directory | the accelerator |\n"
                "| 2. Carry the code, install the runtime | [Engineering] | carried files verified; an isolated hash-locked environment | versions |\n"
                "| 3. Pin, verify and convert the checkpoint | [Engineering] | 68.7 MB `.pth` checked, unpickled safely, converted | digests |\n"
                "| 4. Sample pairs, validation and split | [Evaluation practice] | 60 synthetic pairs validated and split by photograph | split, refusals |\n"
                "| 5. Baselines and the pretrained model | [Evaluation practice] | four methods scored on the test pairs | the baseline table |\n"
                "| 6. Bounded fine-tuning | [Concept] | the decoder half trained for 300 steps | trainable parameters, loss |\n"
                "| 7. Held-out comparison | [Evaluation practice] | adapted vs everything else on identical pairs | the principal result |\n"
                "| 8. Fresh reload and new images | [Engineering] | artifact reloaded in a new process; two unseen images | reload parity |\n"
                "| 9. Optional activity | [Concept] | change the trainable scope (off by default) | your comparison |\n"
                "| 10. Bring your own data | [Engineering] | the same stages on your images (off by default) | your results |\n"
                "| Troubleshooting | [Engineering] | common hosted-runtime failures | when something fails |\n"
                "| Interpretation and conclusion | [Evaluation practice] | limits and an evidence-based conclusion | your conclusion |\n\n"
                "**Fast path.** Short on time? Run all, then read Sections 5 and 7 and the conclusion: they carry the principal results. The "
                "canonical path ends with Section 8; Sections 9 and 10 change nothing unless you switch them on."
            ),
            (
                "<details>\n"
                "<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
                "| Term | Meaning in this notebook |\n"
                "|---|---|\n"
                "| **Deblurring / restoration** | Estimating the sharp image from a degraded one. |\n"
                "| **Blur kernel (point-spread function)** | The small image that describes how one point of light is smeared; a blurred image is the sharp image convolved with it. |\n"
                "| **Motion blur** | Blur from movement during exposure; here a straight line of length 9..21 px at a random angle. |\n"
                "| **Convolution** | Replacing each pixel by a weighted sum of its neighbours, the weights given by the kernel. |\n"
                "| **Noise, quantisation** | Random per-pixel error (sigma 0.01 of the range here) and rounding to whole 8-bit values. |\n"
                "| **Pair** | A blurred image and the sharp image it was made from. |\n"
                "| **Crop** | A 256 × 256 window cut from a photograph; several crops come from each photograph. |\n"
                "| **Split by photograph** | All crops of one photograph go to the same side of the train/test split, so test scenes are never seen in training. |\n"
                "| **PSNR** | Peak signal-to-noise ratio in dB: 10·log10(255² / mean squared error). Higher is better; +1 dB ≈ 21 % less squared error. |\n"
                "| **SSIM** | Structural similarity in −1..1, comparing local means, contrasts and correlations; 1 means identical structure. |\n"
                "| **Identity baseline** | Using the blurred input as the answer: the floor every method must beat. |\n"
                "| **Unsharp masking** | Adding back the difference between an image and a blurred copy of it; blind, cheap, does not undo motion blur. |\n"
                "| **Wiener deconvolution** | Inverting a *known* blur in the frequency domain while damping noise; here given the true kernel (an oracle). |\n"
                "| **NAFNet** | Nonlinear Activation Free Network: a U-shaped restoration network whose blocks use multiplication (SimpleGate) instead of activation functions. |\n"
                "| **Encoder / decoder** | The half of the U that shrinks the image into features / the half that expands them back to an image. |\n"
                "| **TLC (NAFNetLocal)** | Test-time local converter: replaces global average pooling by pooling over a window sized from the training crops, as in the upstream GoPro test configuration. Identical to global pooling for images up to 384 px. |\n"
                "| **Pretrained** | The authors' GoPro weights, unchanged. |\n"
                "| **Fine-tuning** | Continuing gradient training on new data; here only some parameters are updated (the others are *frozen*). |\n"
                "| **Trainable scope** | Which parts are updated: `decoder` (default), `decoder+middle` or `all`. |\n"
                "| **PSNR loss** | The upstream training loss: minus the PSNR of the batch, so lowering it raises training-batch PSNR. |\n"
                "| **Paired comparison** | Two methods scored on exactly the same inputs, so differences are not random re-draws. |\n"
                "| **Artifact** | The exported `model.safetensors` + `manifest.json`. |\n"
                "| **safetensors / pickle** | A tensor-only file format that cannot run code / Python's object format, which can. |\n"
                "| **Digest (SHA-256)** | A fingerprint of a file's bytes; a single changed byte changes it. |\n"
                "| **Hash-locked environment** | A separate Python environment built from a requirements file that pins every package to one version and one set of SHA-256 digests. |\n"
                "| **Stage** | One step of the workflow run as its own process by `run_stage`. |\n"
                "| **BYOD** | Bring Your Own Data: an optional switch to run the same workflow on your own images. |\n\n"
                "</details>"
            ),
        ],
    },
    "setup": [
        {
            "cell": "check",
            "md": (
                "## 1. Check the runtime · [Engineering]\n\n"
                "> **Infrastructure.** The code cells in Sections 1–3 are collapsed. You may run them without studying their implementation; they "
                "exist for reproducibility and provenance. The learning activities start in Section 4.\n\n"
                "**Input:** a fresh hosted runtime. **System:** checks that it is Linux x86_64 with enough free disk, reports whether a CUDA GPU is "
                "present, and creates a new run directory. **Output:** the accelerator and the directories this run will use. Each run writes to a "
                "new directory under `outputs/{stem}/`, so an earlier export cannot be mistaken for a current result. The verified checkpoint is "
                "kept in `weights/` and reused by a later run."
            ),
            "after": (
                "**Expected result:** one dictionary naming the accelerator (for example `Tesla T4, 15360 MiB`, or `none (CPU only)` with a note "
                "that fine-tuning will be slower), the kernel's Python version, the run directory, the weights directory, the isolated "
                "environment's directory and the free disk. If the cell stops with a platform or disk message, see **Troubleshooting**."
            ),
        },
        {
            "cell": "carrier",
            "md": (
                "## 2. Carry the code and install the locked runtime · [Engineering]\n\n"
                "> **Infrastructure.** The next two code cells are collapsed. The first **is** the code this notebook runs, carried so that the "
                "notebook works on its own; the second builds the environment every stage runs in.\n\n"
                "The first cell holds, as text, the files the workflow needs: the package's four modules under `src/nafnet_deblurring_pipeline/` "
                "(identity constants and checkpoint handling, the degradation model and validation, the metrics and baselines), the three upstream "
                "architecture files `NAFNet_arch.py`, `arch_util.py` and `local_arch.py` copied verbatim from `megvii-research/NAFNet` at "
                "`{MODEL_REVISION}` with the upstream licence, the stage runner `tutorial_stages.py`, the hash-locked `requirements.txt` "
                "({n_locked} packages), the checkpoint manifest and the repository licence. It writes each file into the run directory and checks "
                "its SHA-256 against `CARRIED_HASHES`, stopping on any mismatch; the package checks the three upstream files once more against "
                "their upstream digests before executing them. The repository's parity test (`tests/test_notebook_parity.py`) fails whenever the "
                "carried text and the repository diverge, so what runs here is what the repository tests. Nothing in this cell runs a model."
            ),
            "after": (
                "**Expected result:** `carried_files`, `verified: True`, and the repository revision the notebook was generated from.\n\n"
                "The next cell installs nothing into this notebook's kernel. It downloads one pinned file — the `uv` installer wheel, refused unless "
                "its size and SHA-256 match — creates a separate virtual environment with its own CPython 3.12.12, and installs `requirements.txt` "
                "into it with `--require-hashes --only-binary :all:`: every package must be the locked version, a prebuilt wheel, and match a "
                "locked digest. The hosted runtime's own packages are never replaced, which is why no restart is needed. The stage processes get "
                "no Hugging Face token and use a file-only plotting backend. The cell also defines `run_stage`, `load_record` and `show_image`, the "
                "three helpers the learner cells use."
            ),
        },
        {
            "cell": "install",
            "md": (
                "**Infrastructure: the isolated environment.** Installation messages from `uv` are normal and can take a few minutes (the torch "
                "wheel and its CUDA libraries are several GB). A failed download or a hash mismatch stops the cell; never remove a pin or a hash to "
                "get past one."
            ),
            "after": (
                "**Expected result:** one dictionary with the generating revision, the isolated environment's Python (3.12.12), the `torch`, "
                "`numpy` and `safetensors` versions, `'cuda': True` on a GPU runtime (`False` on a CPU-only runtime, with a note), the number of "
                "locked packages and the setup time."
            ),
        },
        {
            "cell": "weights",
            "md": (
                "## 3. Pin, verify and convert the checkpoint · [Engineering]\n\n"
                "> **Infrastructure.** The next code cell is collapsed. It downloads one 68.7 MB file and checks it; you may run it without "
                "studying its implementation.\n\n"
                "The identity is carried twice — the constants in the carried `pipeline.py` and the carried `dimer-base-manifest.json` — and the "
                "`weights` stage first checks that they agree. The authors publish `NAFNet-GoPro-width32.pth` on Google Drive and Baidu, which a "
                "hosted runtime cannot download reliably without interaction, so the stage fetches the same file name from a public Hugging Face "
                "mirror (a second mirror is tried only if the first fails) and accepts the bytes **only** if their size is 68,671,121 and their "
                "SHA-256 equals the pinned value; anything else is deleted and refused, never loaded. The verified pickle is then read once with "
                "`torch.load(..., weights_only=True)`, loaded strictly into the carried architecture (every tensor name and shape must match; "
                "17,111,907 parameters), saved as safetensors, reloaded from that file into a second network, and both networks are compared on a "
                "seeded input: the conversion is accepted only if the outputs are identical. Every later stage loads the safetensors file after "
                "checking its digest."
            ),
            "after": (
                "**What to notice:** the model identity and the pinned digest; `fetched_on_this_run` (`False` on a rerun, because a verified file "
                "is reused) and the mirror URL; then the converted file's size, SHA-256, tensor and parameter counts, and "
                "`max_abs_output_diff_vs_pth: 0.0`. A size or SHA-256 mismatch stops the cell with a message naming the file and both digests — "
                "see **Troubleshooting**, and never edit a digest to get past one."
            ),
        },
    ],
    "cells": [
        {
            "md": (
                "## 4. Sample pairs, validation and the split · [Evaluation practice]\n\n"
                "From here on, every code cell runs one stage of the carried runner with `run_stage`; its printed dictionaries appear under the "
                "cell. This cell runs the `prepare` stage.\n\n"
                "**How a pair is made.** Each sharp photograph is convolved with a **motion kernel** — an anti-aliased straight line 9..21 px long "
                "at a random angle — then Gaussian noise (sigma 0.01 of the 0..1 range) is added and the result is rounded to 8 bits. The whole "
                "photograph is blurred before a 256 × 256 crop is cut from the blurred and the sharp version at the same place, so the crop has no "
                "border artefact. Every kernel, noise draw and crop position comes from one seeded generator, so every run builds the same 60 "
                "pairs: 8 crops from each of six training photographs (48 pairs) and 4 from each of three test photographs (12 pairs).\n\n"
                "**Why split by photograph?** Crops of one photograph share content, lighting and texture. If some crops of the astronaut were in "
                "training and others in test, the test score would partly measure memory of the scene. `dataset_manifest` checks that no "
                "photograph appears on both sides and records a digest that every later stage re-checks.\n\n"
                "**Validation** checks each record's structure before any model sees it; a **refusal probe** is a deliberately broken input used to "
                "show that the check works.\n\n"
                "**Expected result:** 48 / 12 pairs, the training and test photographs listed separately, blur lengths between 9 and 21 px, five "
                "refusal probes each `rejected` with a message naming the record and the rule, and a sample sheet showing blurred inputs, sharp "
                "references and their kernels."
            ),
            "code": (
                "run_stage('prepare')\n"
                "show_image('{stem}_sample_pairs.png', 'Sample pairs: blurred input, sharp reference and the blur kernel')"
            ),
        },
        {
            "md": (
                "**What to notice:** `disjoint_sources: True`; the five refusal probes — a pair whose two images differ in size, a 40 px image, a "
                "duplicate id, four pairs from one photograph, and a greyscale array where RGB is required — each name the rule they broke. These "
                "are the same messages a bad BYOD input produces in Section 10. `outputs/{stem}_sample_pairs.csv` in the run directory lists every "
                "pair with its kernel and the licence of its photograph.\n\n"
                "**Checkpoint:** a classmate proposes shuffling all 60 crops and taking 12 at random for test, \"because that is how random splits "
                "work\". What would that do to the test score in Section 7, and why does validation refuse four pairs from one photograph?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Crops of one photograph are not independent: they share the scene, its colours and textures. A shuffled split would put crops of "
                "the same photograph on both sides, and the fine-tuned model could score well on test partly because it has seen that scene — the "
                "test score would overstate how it does on a new photograph. Splitting by photograph keeps the test scenes unseen, which is the "
                "question that matters. Four pairs from one photograph cannot be split that way at all — there would be no photograph left for one "
                "side — so validation refuses them before any model runs. Validation is structural only: it cannot check that a pair shows the "
                "same scene or that the blur is motion blur.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 5. Baselines and the pretrained model · [Evaluation practice]\n\n"
                "**Question tested:** before any adaptation, how much does the pretrained NAFNet improve these held-out pairs, compared with doing "
                "nothing, a blind classical filter and a classical method that is told the true blur?\n\n"
                "The `baseline` stage scores four methods on the 12 test pairs with the same two metrics.\n\n"
                "- **PSNR** compares pixel values: 10·log10(255² / MSE) in dB over all pixels and channels, as in the upstream GoPro test "
                "configuration (RGB, no border crop). +1 dB means about 21 % less mean squared error.\n"
                "- **SSIM** compares local structure — means, contrasts and correlations in 11 × 11 Gaussian windows — per RGB channel, averaged; 1 "
                "is identical. It can disagree with PSNR: a slightly shifted sharp edge loses PSNR but keeps much of its SSIM, and a smooth blurry "
                "image can have a decent PSNR and a poor SSIM. The upstream repository uses a 3-D SSIM variant, so these SSIM values are not "
                "comparable with its README table, and neither metric here reproduces that table.\n\n"
                "The methods: **identity** (the blurred input itself — the floor); **unsharp masking** (blind; its radius and strength are chosen on "
                "the *training* pairs and then frozen); **Wiener deconvolution with the true kernel** (an *oracle*: it is given the exact blur, which "
                "a real deblurrer never knows; its noise setting is chosen on the training pairs); and the **pretrained NAFNet** (blind: it sees only "
                "the blurred image). Nothing is tuned on the test pairs.\n\n"
                "**Predict before running:** rank the four methods by test PSNR. Will blind unsharp masking beat the blurred input? Will the blind "
                "network beat the oracle that knows the kernel?"
            ),
            "code": (
                "run_stage('baseline')\n"
                "show_image('{stem}_baselines.png', 'Test pairs: blurred input, unsharp, oracle Wiener, pretrained NAFNet and the sharp reference')"
            ),
        },
        {
            "md": (
                "**What to notice:** the settings chosen on the training pairs; one line per method with its mean test PSNR and SSIM; and "
                "`pretrained_vs_blurred_input`, the per-pair PSNR gain of the pretrained network over the blurred input with its smallest and "
                "largest value and how many of the 12 pairs improved. Look at the sheet as well as the numbers: ringing (ripples near edges) is "
                "typical of Wiener deconvolution and of over-strong sharpening.\n\n"
                "**Checkpoint:** unsharp masking usually stays at or below the blurred input in PSNR on this sample. Why can a filter that makes "
                "edges look crisper fail to improve PSNR, and why is the Wiener result called an *oracle* rather than a fair competitor?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Unsharp masking boosts the difference between an image and a blurred copy. It raises contrast at edges but does not move the "
                "light that motion blur smeared along the motion direction back to where it belongs, and it amplifies noise; PSNR counts every "
                "pixel's error, so the extra noise and overshoot can cost more than the edge contrast gains — which is why its strength had to be "
                "chosen on the training pairs and why it may still end below the identity on test. Wiener deconvolution inverts the blur in the "
                "frequency domain, but only because it is handed the exact kernel of each pair; a real photograph comes with no kernel, and "
                "estimating one is a hard problem of its own. It shows what a classical method can do with perfect knowledge, so a blind network "
                "that gets close to it, or beats it, is doing something useful; one that falls below it has room to improve. These numbers are 12 "
                "crops from 3 photographs: a tutorial measurement, not a benchmark.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 6. Bounded fine-tuning of the decoder · [Concept]\n\n"
                "The `adapt` stage always starts from the verified pretrained weights, so re-running this cell repeats the same experiment rather "
                "than continuing an earlier one. It updates only the parameters in `TRAINABLE_SCOPE` with gradient descent — this is ordinary "
                "fine-tuning of a parameter subset, not an adapter or PEFT method. With the default `decoder` scope the four upsampling layers, the "
                "four decoder blocks and the final convolution are trained (1,322,307 parameters, 7.7 % of 17,111,907); the intro convolution, the "
                "encoder (with its 28-block deepest level), the downsampling layers and the middle block stay frozen. Each step takes `BATCH_SIZE` "
                "random 256 × 256 crops of training pairs, each flipped or rotated at random, and minimises the upstream **PSNR loss** "
                "(10/ln 10 · mean log MSE, i.e. minus the batch PSNR) with AdamW (betas 0.9/0.9, no weight decay) at a fixed learning rate, in "
                "float32. The number of steps is fixed in advance: no model is selected on any split, so no validation split is needed and the test "
                "pairs are never used for a decision. The stage records reference outputs of the trained model while it is still in memory, then "
                "exports it as `outputs/{stem}_artifact/` (`model.safetensors` + `manifest.json`).\n\n"
                "**Predict before running:** will the training-batch PSNR rise smoothly from log line to log line? After 300 steps on 48 pairs, do "
                "you expect the held-out gain over the pretrained model (Section 7) to be large (several dB) or modest?"
            ),
            "code": (
                'STEPS = 300  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 4  # @param {{type:"integer"}}\n'
                'TRAINABLE_SCOPE = \'decoder\'  # @param ["decoder", "decoder+middle", "all"]\n\n'
                "run_stage('adapt', '--steps', STEPS, '--lr', LEARNING_RATE, '--batch-size', BATCH_SIZE, '--scope', TRAINABLE_SCOPE)"
            ),
        },
        {
            "md": (
                "**What to notice:** the configuration line; one log line every 25 steps with the mean PSNR loss and the equivalent training-batch "
                "PSNR in dB; the trainable, frozen and total parameter counts; the elapsed seconds; and the exported artifact with its SHA-256.\n\n"
                "**Checkpoint:** why is a rising training-batch PSNR not evidence that the model deblurs better?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "The training batches are crops of the six training photographs, which the model is being fitted to; a falling loss shows that the "
                "optimiser is working, not that the model generalises. It is also noisy: each log line averages 25 batches of different crops, "
                "kernels and orientations, and some crops are much harder than others, so the curve wobbles. Only the held-out comparison in "
                "Section 7, on photographs the model never trained on, speaks to quality — and even that only for this kind of synthetic blur. "
                "Freezing most of the network limits how far it can drift from the pretrained solution with only 48 pairs, which is a guard against "
                "overfitting, not a guarantee.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 7. Held-out evaluation: the paired comparison · [Evaluation practice]\n\n"
                "**Question tested:** on test pairs from photographs that played no part in training, does the adapted network restore better than "
                "the pretrained one and the classical baselines?\n\n"
                "The `evaluate` stage is a fresh process: it loads the adapted network from the exported files — the artifact you would ship, not "
                "the object that was trained — after checking the manifest and the weights' SHA-256, and scores it on exactly the 12 pairs and with "
                "exactly the metrics of Section 5. Because every method sees the same inputs, a difference between two methods is a *paired* "
                "difference, not a re-draw of random data. The stage prints the full table, the per-pair gains (mean, smallest, largest, number of "
                "pairs improved) and the mean PSNR per test photograph.\n\n"
                "**Predict before running:** will every one of the 12 pairs improve over the pretrained model, or only most? Which test photograph "
                "— the astronaut portrait, the rocket launch or the gravel texture — do you expect to gain least, and why?"
            ),
            "code": (
                "run_stage('evaluate')\n"
                "show_image('{stem}_evaluation.png', 'Test pairs: classical baselines, pretrained, adapted and the sharp reference')"
            ),
        },
        {
            "md": (
                "**What to notice:** a table with five rows (identity, unsharp, wiener_oracle, pretrained, adapted) and `adapted_vs_pretrained`, "
                "the principal result: its mean, its range and how many of the 12 pairs improved. Compare the pretrained and adapted columns of the "
                "sheet: differences are often subtle at this size, so look at fine texture and straight edges. The full record is "
                "`outputs/{stem}_evaluation_report.json` and the per-pair table `outputs/{stem}_test_metrics.csv` in the run directory.\n\n"
                "**Checkpoint:** write one sentence that reports the principal result honestly. What does it *not* tell you?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "A defensible sentence has the form: \"on 12 held-out 256-px crops from 3 photographs, with synthetic linear motion blur, "
                "fine-tuning the decoder for 300 steps changed mean PSNR from *a* dB (pretrained) to *b* dB, improving *k* of 12 pairs, against "
                "*c* dB for the blurred input and *d* dB for the oracle Wiener filter\". It does not tell you how the model behaves on real camera "
                "blur, defocus or other noise levels, because the adaptation targeted exactly this synthetic distribution — gains here can come with "
                "losses on GoPro-like blur, which this notebook does not measure. It is one seeded run with no spread estimate; another seed would "
                "move the numbers a little. A photograph dominated by fine random texture (gravel) is the hardest to restore and often gains least, "
                "but read your own per-photograph line rather than assuming it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 8. Fresh reload and inference on new images · [Engineering]\n\n"
                "The `reload` stage is a second fresh process. It verifies the artifact's manifest — format, model identity, the list of files "
                "(nothing more, nothing missing), the weights' size and SHA-256 and the upstream architecture digests — **before** reading the "
                "weights, rebuilds the network from the recorded architecture and loads them strictly. Nothing of the trained model survives in "
                "memory between processes, so this is a reload from files. It then compares the reloaded network's float outputs on two test pairs "
                "with the outputs the `adapt` stage recorded from the trained model while it was still in memory; the check passes only if the "
                "largest difference is at most 1e-5. *Loading* a file only shows that it parses; matching outputs show that the file reproduces "
                "the model.\n\n"
                "Then both the pretrained and the reloaded adapted network deblur two images that played no part in adaptation or evaluation: "
                "`text-synthetic`, a printed-text image blurred with one fixed kernel (15 px at 30°), which has a sharp reference and so gets PSNR "
                "and SSIM; and `clock-real-motion`, a real photograph of a wall clock taken while moving the camera, which has **no** sharp reference. "
                "For it the stage reports only a *gradient-energy ratio* (how much stronger the restored image's edges are than the input's) — a "
                "sharpness proxy that noise and ringing also raise, never a quality score. Finally it writes `outputs/{stem}_result.json` with the "
                "provenance, the runtime versions, the evaluation table and the reload check, and lists every file in `outputs/`.\n\n"
                "**Predict before running:** will the reloaded artifact reproduce the trained model's outputs exactly, approximately, or "
                "not at all? On the real clock photograph, which model — pretrained on real camera blur, or adapted to synthetic straight-line "
                "blur — do you expect to look better?\n\n"
                "**Expected result:** `artifact_manifest_verified: True`, a `reload_parity` dictionary whose `max_abs_float_diff` is at or near zero "
                "and within the tolerance, one line per new image, the listing of `outputs/`, and a sheet with the two images."
            ),
            "code": (
                "run_stage('reload')\n"
                "show_image('{stem}_new_images.png', 'New images: blurred input, pretrained, adapted and (if known) the sharp reference')"
            ),
        },
        {
            "md": (
                "**What to notice:** the text image's PSNR for the blurred input, the pretrained and the adapted network — one image, so a sanity "
                "check, not an evaluation — and the clock photograph, where you must judge by eye. The real clock blur was not made by a straight-"
                "line kernel; compare how the pretrained (trained on real camera blur) and the adapted network (tuned to synthetic lines) handle it. "
                "This is the end of the canonical path; everything it produced is in `outputs/`.\n\n"
                "**Checkpoint:** why is there no PSNR for the clock photograph, and why would a higher gradient-energy ratio not prove that one "
                "model restored it better?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "PSNR and SSIM compare an output with the true sharp image. For a real blurred photograph nobody has that image, so any full-"
                "reference number would be invented. Gradient energy only measures how strong the edges are: amplified noise, ringing or "
                "over-sharpened halos raise it as much as genuinely restored detail does, so a model can increase it while making the image worse. "
                "Judging real blur needs a reference capture (a tripod shot, or a paired dataset such as GoPro), a no-reference metric validated for "
                "the task, or human rating — none of which this notebook provides.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 9. Optional activity: change one thing — the trainable scope · [Concept]\n\n"
                "**Predict → Change one thing → Run → Observe → Explain.** This activity is off by default and changes nothing the canonical path "
                "produced: with `RUN_ACTIVITY = False` the next cell only prints how to switch it on. It runs the `activity` stage, which starts "
                "again from the verified pretrained weights (never from the adapted model), fine-tunes with exactly the steps, learning rate, batch "
                "size and seed of Section 6 but a different `ACTIVITY_SCOPE`, scores the result on the same 12 test pairs, and prints it beside the "
                "Section 6 result. It exports no artifact. On a CPU-only runtime it takes as long as Section 6 or longer.\n\n"
                "**Change one thing:** set `ACTIVITY_SCOPE` to `all` (all 17.1 M parameters) or `decoder+middle` (adds the middle block, 3.2 M "
                "trainable in total) and `RUN_ACTIVITY = True`, then run the cell.\n\n"
                "**Predict before running:** with sixteen times as many trainable parameters (`all`), will the held-out PSNR be higher, about the "
                "same, or lower than with the decoder alone? Will training take much longer? Write your prediction down."
            ),
            "code": (
                'RUN_ACTIVITY = False  # @param {{type:"boolean"}}\n'
                'ACTIVITY_SCOPE = \'all\'  # @param ["decoder", "decoder+middle", "all"]\n\n'
                'if RUN_ACTIVITY:\n'
                "    run_stage('activity', '--scope', ACTIVITY_SCOPE)\n"
                'else:\n'
                "    print({{'activity': 'skipped (optional)', 'to_run': 'set RUN_ACTIVITY = True and choose ACTIVITY_SCOPE, then run this cell'}})"
            ),
        },
        {
            "md": (
                "**Observe:** three rows — the pretrained model, the Section 6 adaptation and the activity's — each with test PSNR, SSIM, trainable "
                "parameters, training seconds and the first logged loss (which should be close for both adaptations, showing that both started from "
                "the same weights).\n\n"
                "**Explain:** did the result match your prediction? What does it suggest about how much of the network needs to change to adapt "
                "to a new blur distribution?\n\n"
                "<details>\n<summary>Check your reasoning (open after answering)</summary>\n\n"
                "Training every parameter gives the optimiser more freedom: on a distribution shift like this one it can fit the new blur more "
                "closely, so held-out PSNR often rises somewhat — but with only 48 training pairs from six photographs it can also start to fit "
                "those scenes, and the backward pass through the 28-block encoder costs more memory and time. The decoder-only scope keeps the "
                "encoder's features fixed and changes how they are turned back into an image, which is cheaper and changes the pretrained model less. "
                "Either outcome is an observation about one seeded run on 12 test pairs; a difference of a few hundredths of a dB is within what a "
                "different seed could produce. Because only the scope changed, any difference is caused by it.\n\n"
                "</details>"
            ),
        },
        {
            "md": (
                "## 10. Bring your own data (optional) · [Engineering]\n\n"
                "Off by default; with `USE_BYOD = False` the next cell only prints how to switch it on. The `byod` stage uses the same validation, "
                "metrics, baselines, fine-tuning, artifact format and reload check as the canonical path.\n\n"
                "**Input contract (read before choosing files).** A zip archive (or, with `BYOD_PATH`, a zip or a directory already in the runtime) "
                "containing either\n\n"
                "- **paired data:** a folder `sharp/` and a folder `blurred/`, with each pair stored under the same file name in both (for example "
                "`sharp/street_01.png` and `blurred/street_01.png`); at least 4 and at most 200 pairs, from at least 2 different scenes; or\n"
                "- **unpaired data:** blurred images only, in `blurred/` or at the top level of the zip; 1..200 images.\n\n"
                "Images are PNG, JPEG, BMP, TIFF or WebP, 8-bit, each side 64..1024 px (larger images are refused with a message — resize or crop "
                "them first); both images of a pair must have the same size; paired fine-tuning uses crops of up to 256 px, so the training images "
                "should be at least that large. One wrapping folder is tolerated; other folders, duplicate names, unmatched files, absolute or `..` "
                "paths and symbolic links are refused with a message naming the file. Members of the zip are read in memory; nothing is extracted "
                "to disk outside the run directory.\n\n"
                "**Paired mode** splits *by image* with a fixed seed (one in four images, at least one, goes to test — the split assumes your images "
                "are independent scenes), tunes the unsharp baseline on your training images, scores the identity, unsharp and pretrained baselines "
                "(no oracle Wiener: your kernels are unknown), fine-tunes from the pretrained weights for `BYOD_STEPS` steps with the Section 6 "
                "settings, exports its own artifact, reloads it with the equivalence check, and writes the artifact, the restored test images, "
                "`byod_test_metrics.csv` and `byod_result.json` to `outputs/byod/<mode>-<archive digest>/`, one directory per archive, so a second "
                "BYOD run never overwrites an earlier one. **Unpaired mode** deblurs every image with the pretrained model "
                "and, if Section 6 ran, the adapted sample model, and writes the PNGs and `byod_result.json`; it reports no PSNR or SSIM because there "
                "is no reference.\n\n"
                "**Privacy:** your files stay in this runtime and are sent nowhere. Do not upload confidential, personal or regulated images unless "
                "you are authorized to process them in this environment. Leave `BYOD_PATH` empty to get the Colab upload dialog; set it to a path to "
                "skip the dialog (it works outside Colab too)."
            ),
            "code": (
                'from pathlib import Path\n\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'BYOD_PATH = \'\'  # @param {{type:"string"}}\n'
                'BYOD_STEPS = 300  # @param {{type:"integer"}}\n\n'
                'if USE_BYOD:\n'
                '    if BYOD_PATH:\n'
                '        byod_path = Path(BYOD_PATH)\n'
                '    else:\n'
                '        from google.colab import files\n'
                '        uploaded = files.upload()\n'
                '        file_name, payload = next(iter(uploaded.items()))\n'
                "        byod_path = ROOT / 'byod_upload' / Path(file_name).name\n"
                '        byod_path.parent.mkdir(parents=True, exist_ok=True)\n'
                '        byod_path.write_bytes(payload)\n'
                "    run_stage('byod', '--byod', byod_path.resolve(), '--steps', BYOD_STEPS, '--lr', LEARNING_RATE, '--batch-size', BATCH_SIZE, '--scope', TRAINABLE_SCOPE)\n"
                'else:\n'
                "    print({{'byod': 'skipped (optional)', 'to_run': 'set USE_BYOD = True; optionally set BYOD_PATH to a zip or directory already in the runtime'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `byod_mode` (`paired` or `unpaired`), the validation summary and any colour conversions; for paired data the "
                "split, the baseline lines, the training log, the comparison table with `adapted_vs_pretrained`, and the reload check; for unpaired "
                "data one line per image with gradient-energy ratios. An invalid archive stops the cell with a `RuntimeError` that repeats the "
                "validator's message — for example a file without a partner, a pair whose sizes differ, or an image larger than 1024 px. With very "
                "few test images the metrics describe those images only."
            ),
        },
        {
            "md": (
                "## Troubleshooting · [Engineering]\n\n"
                "| Symptom | Likely cause | What to do |\n"
                "|---|---|---|\n"
                "| Section 1 prints `No CUDA GPU detected` | the runtime has no GPU | The notebook still completes on the CPU, more slowly. For the documented runtime choose *Runtime → Change runtime type → T4 GPU* and run all again from the top. If Colab offers no GPU, your quota may be exhausted. |\n"
                "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle, or a Linux x86_64 machine: the locked environment is built for manylinux x86_64 wheels. |\n"
                "| Section 1 stops with `Not enough free disk` | the isolated environment needs about 9 GB | Start a fresh runtime with more free disk. |\n"
                "| `Carried file integrity failure` in Section 2, or `carried upstream file … sha256 … != pinned` in a stage | a carried file was edited | Do not edit the infrastructure cells; open a fresh copy of the notebook from the repository. |\n"
                "| `uv 0.12.15 wheel size/hash mismatch`, or a `URLError` / timeout while downloading it | a network failure or an unexpected response from PyPI | Re-run the Section 2 install cell. Never replace the pinned URL or digest. |\n"
                "| `CalledProcessError` from `uv venv` or `uv pip install` (a hash mismatch, `Failed to download`, HTTP 5xx) | a transient PyPI or network failure | Re-run the Section 2 install cell: `uv` reuses what it already downloaded. If a hash mismatch repeats, stop and report it — never remove `--require-hashes`, a pin or a hash. |\n"
                "| `DeprecationWarning: 'saved_variables' is deprecated; use 'saved_tensors'` in a stage log | the carried upstream `arch_util.py` uses an older PyTorch name that still works in torch 2.14 | Nothing: the warning is harmless and the carried file is kept verbatim on purpose. |\n"
                "| `RuntimeError: Stage '…' failed (exit 2): …` | the stage raised an error; the text after the colon is the stage's own message, and its full log is printed above and kept in the run directory's `logs/` | Find the message in the rows below. A stage reads only files, so after fixing the cause you can re-run that cell and the cells after it. |\n"
                "| `no mirror delivered the pinned checkpoint bytes` | the mirrors were unreachable, or served different bytes | Re-run the Section 3 cell later. If the message shows a different SHA-256, the mirror changed: the notebook correctly refuses it; report it rather than editing the digest. |\n"
                "| `… is missing: run the stage that writes it before …` | a learner cell was run before an earlier stage | Run the notebook from the top, or re-run the earlier cells in order. |\n"
                "| `the sample changed since 'prepare'` | the sample photographs or the package changed after Section 4 | Re-run from Section 4. |\n"
                "| `CUDA out of memory` | `BATCH_SIZE` was raised, the `all` scope was chosen on a small GPU, or another program holds GPU memory | Lower `BATCH_SIZE` (for example to 2) and re-run that cell; each stage is its own process, so the failed stage's memory was released. |\n"
                "| `reloaded artifact differs from the trained model` | the artifact files were modified, or a nondeterministic kernel was used | Re-run Section 6 and Section 8; report it if it repeats. |\n"
                "| `ModuleNotFoundError: No module named 'google.colab'` with `USE_BYOD = True` | the upload dialog needs Google Colab | Set `BYOD_PATH` to a zip or directory already in the runtime, or use Colab. |\n"
                "| `BYOD: files without a partner …` or `… is not in sharp/ or blurred/` | the archive layout differs from the contract | Put pairs in `sharp/` and `blurred/` under identical file names; Section 10 gives an example. |\n"
                "| `… each side must be within 64..1024 px` or `blurred is … but sharp is …` | an image is too small or too large, or a pair's sizes differ | Resize or crop the images; both images of a pair must have identical size. |\n"
                "| `at least 2 distinct sources are needed` or `4..200 are required` | too few paired images | Provide at least four pairs from at least two scenes. |"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits · [Evaluation practice]\n\n"
        "The question this notebook can answer is narrow: does fine-tuning the decoder half of a GoPro-trained NAFNet for a few hundred steps on "
        "48 synthetic motion-blur pairs improve PSNR and SSIM on 12 pairs from three unseen photographs with the same kind of blur, compared with "
        "the pretrained model, the blurred input, a blind classical filter and a classical oracle? Your run answers it in Section 7 on identical "
        "inputs; read the direction and size of each difference there rather than from this text.\n\n"
        "The numbers are sample-sanity evidence. The blur is synthetic (straight-line kernels, Gaussian noise, no camera response curve, no "
        "compression), the test set is 12 crops from 3 photographs with no dispersion estimate, the oracle baseline knows what no real method "
        "knows, and PSNR/SSIM do not measure perceived quality. Adapting to synthetic lines can make the model worse on real camera blur — "
        "Section 8's clock photograph is one visual hint, not a measurement. None of this reproduces or tests the upstream GoPro results.\n\n"
        "Three things to carry to real data. **The reference defines the metric:** without a sharp image, PSNR and SSIM cannot be computed, so "
        "real deployments need paired captures or human rating. **Split by scene, not by crop:** crops of one photograph share content and leak "
        "across a random split. **The trust boundary is the digest:** the checkpoint is a pickle; it is safe to load here only because its exact "
        "bytes are pinned and read with `weights_only=True`, and the exported artifact is safetensors.\n\n"
        "Successful execution proves that the recorded repository revision's modules, the verbatim upstream architecture and the stage runner, "
        "carried in this standalone notebook and run in an isolated hash-locked environment, can fetch and digest-verify the pinned checkpoint, "
        "convert it to safetensors with identical outputs, build and validate the sample, score classical baselines and the pretrained network, "
        "execute a bounded fine-tuning, evaluate the exported artifact on held-out pairs, reload it in a fresh process with output equivalence and "
        "emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "production fitness, or restoration quality on real blur.\n\n"
        "## Conclude with evidence · [Evaluation practice]\n\n"
        "Complete this in your own words, using the numbers your run printed:\n\n"
        "> On [12 held-out 256-px crops from three photographs with synthetic linear motion blur / your BYOD test images], fine-tuning "
        "NAFNet-GoPro-width32 ([scope], [steps] steps, [trainable] of 17,111,907 parameters) changed mean test PSNR from [pretrained] dB to "
        "[adapted] dB and SSIM from [pretrained] to [adapted], improving [k] of [n] pairs. The blurred input scored [identity] dB, unsharp masking "
        "[unsharp] dB and the oracle Wiener filter, which was given the true kernels, [wiener] dB. The most important uncertainty is [for "
        "example: one seeded run on 12 crops; synthetic blur only; the hardest photograph]. These numbers do not show [behaviour on real camera "
        "blur / perceived quality / ...]. Next I would [specific next experiment].\n\n"
        "<details>\n<summary>Check your reasoning: what makes a conclusion strong? (open after writing yours)</summary>\n\n"
        "A strong conclusion names the data (how many held-out pairs, from how many photographs, which blur), reports the pretrained model and the "
        "baselines beside the adapted number, and says how many pairs improved rather than only the mean. It says what the oracle knows, keeps "
        "the training loss out of the evidence, and limits the claim to synthetic linear motion blur. It ends with a specific next step — several "
        "seeds, more test photographs, a real paired blur dataset, or checking the adapted model on GoPro-like blur for regressions. A weak "
        "conclusion says only that \"fine-tuning improved deblurring\".\n\n"
        "</details>\n\n"
        "**Transfer:** switch on BYOD (Section 10) with paired images from your own camera — for example a tripod shot and a handheld shot of the "
        "same still scene, aligned and cropped to the same size — and write the same conclusion for them. Before running, predict whether the "
        "adapted sample model or the pretrained model will do better on real camera blur, and why.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weights provenance and trust gap: https://github.com/kurtvalcorza/nafnet-deblurring-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream repository: https://github.com/megvii-research/NAFNet (revision `{MODEL_REVISION}`)\n"
        "- Checkpoint mirror: https://huggingface.co/nyanko7/nafnet-models (file `NAFNet-GoPro-width32.pth`, accepted only at the pinned SHA-256)\n"
        "- Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). Simple baselines for image restoration. ECCV 2022. https://doi.org/10.1007/978-3-031-20071-7_2\n"
        "- Chu, X., Chen, L., Chen, C., & Lu, X. (2022). Improving image restoration by revisiting global information aggregation (TLC). ECCV 2022. https://doi.org/10.1007/978-3-031-20071-7_4\n"
        "- Nah, S., Kim, T. H., & Lee, K. M. (2017). Deep multi-scale convolutional neural network for dynamic scene deblurring (GoPro dataset). CVPR 2017. https://doi.org/10.1109/CVPR.2017.35\n"
        "- Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE TIP, 13(4), 600–612. https://doi.org/10.1109/TIP.2003.819861\n"
        "- van der Walt, S., et al. (2014). scikit-image: Image processing in Python. PeerJ, 2, e453 (the sample photographs). https://doi.org/10.7717/peerj.453\n"
        "- uv (the installer that builds the isolated environment): https://docs.astral.sh/uv/\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.2 (in the ml-worker repository)\n"
    ),
}
