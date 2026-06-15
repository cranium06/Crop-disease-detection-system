# Implementation Notes — Chapter 3 Consistency Review

This note records a cross-check of the implementation (`config.py`,
`dataset_preparation.py`, `preprocessing.py`, `model.py`, `train.py`,
`evaluate.py`, `app.py`) against *Chapter 3* of
`Crop_Disease_Detection_Chapters_1_3_REVISED.docx` (Sections 3.3–3.9,
Tables 3.1–3.3), and the fixes applied as a result. Intended as a reference
for Chapter 4 write-up / supervisor review.

## 1. Augmentation order (fixed in code)

**File:** `preprocessing.py`, `augment()`

Table 3.2 lists the eight training-time augmentations in this order:
... Contrast Adjustment (6) → **Gaussian Noise Addition (7)** → **Random
Shear (8)**.

The implementation previously applied Shear (7) before Gaussian Noise (8) —
the opposite order — while the docstring claimed the steps followed Table
3.2. The two operations have been swapped so the code now applies Gaussian
Noise before Shear, matching Table 3.2 exactly. No parameter values changed
(σ = 0.01, shear ±0.2).

## 2. Pandas usage (fixed in code)

**Files:** `requirements.txt`, `dataset_preparation.py`

Section 3.9.3 states Pandas 1.5 is used "for dataset metadata management and
class distribution analysis," and `pandas==1.5.0` is pinned, but no script
previously imported pandas.

Added `summarize_class_distribution()` to `dataset_preparation.py`, which
builds a pandas `DataFrame` reproducing the Table 3.1 class distribution
(class index, class name, folder name, image count, percentage) and prints
it. It is called from `__main__` immediately after `build_file_list()`. The
pandas dependency in `requirements.txt` is now actually exercised by the
pipeline.

## 3. Loss function naming — Table 3.3 vs. `model.py` (documentation note only, no code change)

**Files:** `model.py` (`_compile_model`), Table 3.3, Section 3.7.1

Table 3.3 and Section 3.7.1 name the loss "Categorical Cross-Entropy."
`model.py` uses `tf.keras.losses.SparseCategoricalCrossentropy()` because
labels are stored as integer class indices (0–5) rather than one-hot
vectors throughout the pipeline (`dataset_preparation.py`,
`preprocessing.py`).

`SparseCategoricalCrossentropy(y_true=int, y_pred=softmax)` and
`CategoricalCrossentropy(y_true=one-hot, y_pred=softmax)` compute an
identical loss value for the same underlying label — they differ only in
the label encoding the loss function expects. No code change is needed.

**Suggested Chapter 3 wording (Table 3.3 / Section 3.7.1):**
> Loss function: Categorical Cross-Entropy, implemented via Keras'
> `SparseCategoricalCrossentropy` (mathematically equivalent to
> `CategoricalCrossentropy` for integer-encoded labels).

## 4. Fine-tuned layer count for ResNet-50 — Table 3.3 wording (documentation note only, no code change)

**Files:** `model.py` (`prepare_phase2`), `config.py` (`FINETUNE_LAYERS`),
Table 3.3, Section 3.6.3

Table 3.3 lists "Fine-tuned backbone layers: Top 30 of **MobileNetV2**."
`config.FINETUNE_LAYERS = 30` is applied generically by
`prepare_phase2(model, backbone_name)` to *both* MobileNetV2 and ResNet-50,
consistent with Section 3.6.3's framing that ResNet-50 "is trained under
identical conditions for comparative evaluation."

Because MobileNetV2 and ResNet-50 have different total layer counts, the
top 30 layers represent a different proportion of each backbone's depth.
This is an intentional methodological choice (same raw hyperparameter value
applied to both models for a controlled comparison), but Table 3.3's
wording implies it is MobileNetV2-specific.

A clarifying comment was added next to `FINETUNE_LAYERS` in `config.py`.

**Suggested Chapter 3 wording (Table 3.3):**
> Fine-tuned backbone layers: Top 30 layers (applied identically to
> MobileNetV2 and ResNet-50 for controlled comparison — Section 3.6.3).

Optionally, Chapter 4 could report the actual layer count for each backbone
(printed by `model.py`'s `_get_backbone()` as "Layers: N" at run time) to
make the resulting proportion explicit.

## 5. TensorFlow version bump (2.11 → 2.12) — fixed in code and docx

**Files:** `requirements.txt`, `preprocessing.py`, `train.py`,
`Crop_Disease_Detection_Chapters_1_3_REVISED.docx` (Section 3.9.2)

While setting up the local environment, `pip install -r requirements.txt`
failed with `ResolutionImpossible`:

```
streamlit 1.32.0 depends on protobuf<5 and >=3.20
tensorflow-intel 2.11.0 depends on protobuf<3.20 and >=3.9.2
```

TensorFlow 2.11 and Streamlit 1.32 require mutually exclusive `protobuf`
ranges and cannot be installed in the same environment. TensorFlow 2.12
relaxed its protobuf pin to `>=3.20.3`, which is compatible with Streamlit
1.32's requirement, so:

- `requirements.txt`: `tensorflow==2.11.0` → `tensorflow==2.12.0`.
- `preprocessing.py` (line 14) and `train.py` (line 21): updated the
  "Section 3.9.2 — TensorFlow 2.11" reference comments to "2.12". No
  functional code changes — the script logic targets the stable
  `tf.keras` / `tf.data` API surface unchanged between 2.11 and 2.12.
- **Section 3.9.2 of the docx** updated: "TensorFlow 2.11" → "TensorFlow
  2.12", with a clarifying parenthetical explaining that 2.12 is used in
  place of 2.11 specifically to resolve the protobuf conflict with
  Streamlit 1.32 (Section 3.9.4).

Bumping TensorFlow to 2.12 also tightened its `numpy` requirement to
`>=1.22,<1.24`, which conflicts with the previously pinned `numpy==1.24.0`.
Fixed by pinning `numpy==1.23.5` in `requirements.txt`, which satisfies
TensorFlow 2.12's range as well as the `numpy` requirements of pandas,
scikit-learn, opencv-python, matplotlib, seaborn, and streamlit. A note was
added to `requirements.txt` explaining the pin. Section 3.9.3 of the docx
specifies pandas/scikit-learn/etc. versions but does not pin an exact numpy
version, so no further docx change was needed for this part.

## 6. Windows console UnicodeEncodeError on `→`, `←`, `─` in `print()` (fixed in code)

**Files:** `dataset_preparation.py`, `train.py`, `evaluate.py`

While running `dataset_preparation.py` for the first time on Windows (after
downloading the PlantVillage tomato classes into `data/PlantVillage/`), the
script crashed with:

```
UnicodeEncodeError: 'charmap' codec can't encode character '→' in
position 14: character maps to <undefined>
```

Several `print()` statements use Unicode box-drawing and arrow characters
(`→`, `←`, `─`) for readability (e.g. `"Split saved → {path}"`,
`f"{'─' * 65}"`). Windows consoles default to the `cp1252` code page, which
encodes `—` and `×` but not `→`, `←`, or `─`, so `print()` raises instead of
writing the line.

Note: in `dataset_preparation.py` this crash happened *after* `np.savez()`
had already written the split file, so the first run actually succeeded in
producing `checkpoints/dataset_split.npz` despite the traceback — but
`train.py` and `evaluate.py` contain the same characters in `print()`
statements that run *before* their respective outputs are saved, so the
crash would have been more disruptive there.

Fixed by adding, at the top of each script's `if __name__ == "__main__":`
block:
```python
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
```
(`import sys` added to each file's imports.) This forces UTF-8 output
regardless of the host console's code page.

**Caveat discovered later (see item 9):** this `reconfigure()` approach only
works in `dataset_preparation.py`, which does not import TensorFlow. In
`train.py` and `evaluate.py`, importing TensorFlow resets stdout/stderr back
to `cp1252` *after* the reconfigure call (observed empirically — the exact
mechanism wasn't pinned down further), so the same fix does not hold for
those two scripts. Item 9 describes the fix actually used for `train.py` /
`evaluate.py`.

## 7. `_shear_numpy()` crash — `'float' object has no attribute 'astype'` (fixed in code)

**File:** `preprocessing.py`, `_shear_numpy()` (line 91)

Discovered during the Phase 1 smoke-test run (first batch that triggered the
shear augmentation branch):

```
AttributeError: 'float' object has no attribute 'astype'
```

```python
# Before:
shear = np.random.uniform(-config.AUG_SHEAR, config.AUG_SHEAR).astype(np.float32)
```

`np.random.uniform(low, high)` called with two scalar (non-array) bounds
returns a plain Python `float`, which has no `.astype()` method (that method
only exists on `numpy.ndarray`/`numpy.generic`). Fixed by removing the
`.astype(np.float32)` call — the plain `float` is accepted directly by the
OpenCV affine-transform call that uses `shear`:

```python
# After:
shear = np.random.uniform(-config.AUG_SHEAR, config.AUG_SHEAR)
```

## 8. `prepare_phase2()` crash — `ValueError: No such layer: mobilenetv2` (fixed in code)

**File:** `model.py`, `prepare_phase2()` (lines ~188–197)

Discovered during the Phase 1→Phase 2 transition of the smoke-test run:

```
ValueError: No such layer: mobilenetv2. Existing layers are:
['input_image', 'mobilenetv2_1.00_224', 'gap', 'dense_256', 'dropout_05',
'output_softmax']
```

```python
# Before:
backbone_name = backbone_name.lower()
base_model = model.get_layer(backbone_name)
```

`build_model()` passes `backbone_name` as `"mobilenetv2"` / `"resnet50"`, but
`tf.keras.applications.MobileNetV2()` / `ResNet50()` assign their *own*
auto-generated model names to the returned sub-model (e.g.
`"mobilenetv2_1.00_224"`, `"resnet50"` — and even `"resnet50"` is not
guaranteed if TF appends a uniquifying suffix on repeated builds), so
`model.get_layer("mobilenetv2")` never matches.

Fixed by locating the backbone sub-model by *type* instead of by name — in
this architecture it is the only nested `tf.keras.Model` layer (the head is
plain `GAP`/`Dense`/`Dropout` layers):

```python
# After:
base_model = next(
    layer for layer in model.layers if isinstance(layer, tf.keras.Model)
)
```

Re-validated with a targeted 2-step smoke test (`smoke_test_phase2.py`,
loading the Phase 1 checkpoint, calling `prepare_phase2()`, then running 2
train + 2 val steps) — passed: 30 backbone layers correctly unfrozen (124
remain frozen), recompiled at `LR=1e-05`, trainable params = 1,855,878, loss
decreased over the 2 steps.

## 9. `train.py` / `evaluate.py` `UnicodeEncodeError` on `→`/`←`/`─` despite `reconfigure()` (fixed in code)

**Files:** `train.py`, `evaluate.py`

During the smoke-test run, `train.py` crashed at the Phase 1→Phase 2
separator print — *even though* `sys.stdout.reconfigure(encoding="utf-8")`
runs at the top of `__main__` (and, in the smoke test, before any imports at
all):

```
File "train.py", line 102, in train
    print(f"\n{'─' * 50}")
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-51:
character maps to <undefined>
```

As noted in item 6, `reconfigure()` does not hold once TensorFlow is
imported in these two scripts — stdout/stderr end up back on `cp1252`,
which cannot encode `─` (U+2500), `→` (U+2192) or `←` (U+2190) (it *can*
encode `—` em-dash and `×`, which is why earlier prints in the same run
succeeded).

Rather than rely on `reconfigure()` for these two scripts, all `─`/`→`/`←`
characters in `print()` calls were replaced with plain ASCII:

| File | Before | After |
|------|--------|-------|
| `train.py` (×7 prints) | `f"{'─' * 50}"`, `"...— Two-Phase..."`, `"Best model saved → {path}"`, `"Training curves saved → {path}"` | `f"{'-' * 50}"`, `"...- Two-Phase..."`, `"Best model saved to {path}"`, `"Training curves saved to {path}"` |
| `evaluate.py` (×5 prints) | `f"{'─' * 65}"` (×2), `"...F1-Score ... ← primary metric..."`, `"...report saved → {path}"`, `"...matrix saved → {path}"` | `f"{'-' * 65}"` (×2), `"...F1-Score ... (primary metric...)"`, `"...report saved to {path}"`, `"...matrix saved to {path}"` |

`dataset_preparation.py` (no TensorFlow import) keeps the item-6
`reconfigure()` fix and its `→` characters unchanged — it was already
verified working. The `reconfigure()` calls in `train.py`/`evaluate.py`
`__main__` blocks were left in place (harmless, and still useful for any
future prints of TF's own Unicode warning text) but are no longer
load-bearing for the project's own `print()` statements.

## 10. Resume-from-checkpoint support in `train.py` (added — operational tooling, no methodology change)

**File:** `train.py`

No GPU is available on the development machine, so a full MobileNetV2 run is
estimated at ~37–48 h of CPU time (see "Smoke test validation" below). A CPU
training process does not survive the machine being shut down or slept, and in
practice the first attempt at the full run died at Phase 1 epoch 18/20 (memory
pressure + an interrupt). To make the long run survivable across sessions,
`train.py` was given resume-from-checkpoint support.

This is **execution plumbing only** — it does not change the model, the
hyperparameters, the augmentation, or the two-phase 20+30-epoch schedule
(Section 3.7.3). It is analogous to the (now-removed) smoke-test scaffolding.

**Behaviour:**
- By default training **auto-resumes**: re-invoking
  `python train.py --backbone mobilenetv2` continues from the last completed
  epoch. A new `--restart` flag forces a fresh run (and clears stale resume
  artefacts).
- After **every** epoch, a `_ResumeCheckpoint` callback saves the full model
  (weights + optimiser state) to `checkpoints/<backbone>_last.h5` and writes
  `checkpoints/<backbone>_train_state.json` recording the current phase, the
  number of completed epochs in each phase, and the accumulated
  loss/val_loss/accuracy/val_accuracy history (so the final two-phase training
  plot is correct even when assembled across several runs). The model is saved
  *before* the JSON so the state file never references an epoch the checkpoint
  lacks (worst case on a crash mid-save: one epoch is harmlessly redone).
- On resume the model is reloaded with `tf.keras.models.load_model(_last.h5)`,
  which restores layer trainability and the optimiser LR, so a run interrupted
  during Phase 2 resumes in Phase 2 form without re-calling `prepare_phase2()`.
  `net.fit(initial_epoch=...)` skips already-completed epochs.

**Untouched:** the methodological `checkpoints/<backbone>_best.h5`
(best-`val_loss`, written by `ModelCheckpoint`) — that remains the deliverable
checkpoint `evaluate.py` / `app.py` load. The `_last.h5` / `_train_state.json`
files are resume bookkeeping only.

**Limitation:** `EarlyStopping` / `ReduceLROnPlateau` internal counters are not
persisted across process restarts (Keras does not serialise them), so they
reset on each resume. Acceptable here — they rarely trigger within a single
20/30-epoch phase, and the best-`val_loss` checkpoint is preserved regardless.

**Validation:** a throwaway `resume_test.py` exercised the state machine with
the schedule shrunk to 2+2 epochs and the `tf.data` pipelines monkey-patched to
2 batches/epoch (so each "epoch" runs in seconds): a fresh `--restart` run, a
simulated mid-Phase-1 interruption that correctly resumed ("resuming at epoch
2") and ran through Phase 2, and an already-complete state that trained nothing
and only re-plotted — all passed, with the accumulated history reaching the
expected length (4) and `phase1_len` correctly marking the Phase 1→2 boundary.
The test file was removed after passing.

## 11. `app.py` deployment preprocessing did not match training (fixed in code)

**File:** `app.py`, `preprocess_image()`

`app.py`'s docstring states its preprocessing "mirrors the training pipeline
EXACTLY", but it resized uploads with **PIL `Image.LANCZOS`**, whereas the
training/eval pipeline (`preprocessing.py:load_and_resize`) resizes with
**`tf.image.resize` (bilinear)**. Different interpolation → a real train/serve
skew.

Found by a headless smoke test (`app_smoke.py`, since removed) that ran one
real test image per class through both the `app.py` path and the training
path and compared them: the per-pixel difference reached **0.67** in
ImageNet-normalised space, and on a borderline low-confidence Early-Blight
image (44.9%) the two paths produced **different** predicted classes
(app-path still happened to be correct, but the disagreement confirmed the
skew is large enough to flip predictions).

Fixed by resizing in `app.py` with `tf.image.resize(...)` (bilinear) on the
uint8 RGB array instead of PIL `LANCZOS`, so the deployment path now matches
`preprocessing.py` (bilinear resize → `/255` → ImageNet mean/std). Re-running
the smoke test after the fix cut the max per-pixel difference **from 0.67 to
0.07**. The ~0.07 residual is JPEG-decoder-level (PIL `Image.open` vs
`tf.io.decode_image` decode the same JPEG slightly differently — a few
intensity levels); it is inherent and negligible, so it was not chased
further. App-vs-training agreement stayed 5/6 — the one disagreement is the
same borderline Early-Blight image, now at **34% confidence**: its flip is
driven by the *undertrained epoch-12 model* being unsure, not by preprocessing
(a fully fine-tuned model is far less fragile). (`Image` is still imported in
`app.py` — used to open the uploaded file for display in `main()`.)

## 12. GPU training on Google Colab (`Crop_Disease_Colab_GPU.ipynb`)

**File added:** `Crop_Disease_Colab_GPU.ipynb` (no change to the `.py` pipeline)

Because the development machine has no GPU (~37–48 h CPU estimate), the actual
training run is performed on a free Colab **T4 GPU** (~1–2 h). The notebook runs
the existing `train.py` / `dataset_preparation.py` / `evaluate.py` **unchanged**;
only the runtime differs. Design points worth recording for Chapter 4:

- **Self-contained — the user uploads no data.** The dataset is re-fetched on
  Colab directly from the public `spMohanty/PlantVillage-Dataset` GitHub source
  (the same sparse blob-less clone of `raw/color/Tomato___*` used locally, then
  renamed triple→double underscore), so none of the ~180 MB comes off the
  user's machine. Only the 6 `.py` files are provided (via `files.upload()`,
  a few KB). Everything runs on Colab's local disk (`/content`); no Google
  Drive is required (an optional Drive-mount snippet is included for users who
  want disconnect-proof persistence, but it needs the user's OAuth consent).

- **Split is regenerated on Colab, not uploaded.** `dataset_split.npz` stores
  file *paths* built on Windows (`os.path.join`, backslash separators, relative
  to `DATASET_ROOT="data/PlantVillage"`), which are invalid on Colab's Linux.
  Since `StratifiedShuffleSplit` is seeded (42) and the file set is identical,
  re-running `dataset_preparation.py` on Colab reproduces the **exact same**
  70/15/15 partition with valid paths.

- **TensorFlow / Keras version on Colab.** The documented/local stack is
  TensorFlow 2.12 (Keras 2). Colab ships its own TF; if that is ≥ 2.16 it
  defaults to **Keras 3**, whose saved `.h5` does not load back in the local
  Keras 2 (used by `app.py` / `evaluate.py`). The notebook detects this and
  installs a matching `tf_keras` + sets `TF_USE_LEGACY_KERAS=1`, forcing the
  Keras-2 API for the run, so the downloaded `mobilenetv2_best.h5` loads
  locally. **Deviation to note in Chapter 4:** the production training run may
  execute on a TF minor version newer than 2.12 (kept on the Keras-2 API);
  this does not affect the architecture, hyperparameters, or methodology
  (Section 3.7), only the host TF build.

- **Disconnect handling.** Checkpoints live on Colab's local disk, so a
  disconnect means re-running (cheap at ~1–2 h on GPU); within a live session,
  re-running the training cell auto-resumes via the item-10 resume mechanism.
  The optional Drive-mount snippet makes checkpoints persist across
  disconnects for a fully resumable run.

## 13. Loading the Colab-trained model locally — `compile=False` (fixed in code)

**Files:** `evaluate.py`, `app.py`

After downloading the Colab-trained `mobilenetv2_best.h5` into the local
`checkpoints/`, loading it under the local TensorFlow 2.12 / Keras 2.12 failed:

```
TypeError: SparseCategoricalCrossentropy.__init__() got an unexpected
keyword argument 'fn'
```

The model was trained on Colab with a newer Keras (2.15/2.17, via the
item-12 `tf_keras` legacy mode). That newer Keras serialises the **loss**
config with an extra `fn` key which Keras 2.12 does not accept when it tries
to **recompile** the model on load. The architecture and weights are fully
compatible — only the optimiser/loss restoration trips.

`evaluate.py` and `app.py` only ever run **inference** (`model.predict`), so
the optimiser and loss are not needed. Fixed by loading with
`tf.keras.models.load_model(..., compile=False)` in both files, which skips
compilation and avoids the version-specific loss deserialisation entirely.
Verified locally: the model loads in TF 2.12 (2,587,462 params, 6-class
softmax) and predicts correctly. (`train.py`'s resume path still loads with
the default `compile=True` — correct, since it must continue training; that
only ever loads checkpoints written by the *same* environment.)

**Final test-set results** (held-out 1,941 images, full 20+30-epoch Colab run):
overall accuracy **96.65%**, macro-F1 **0.9551** (vs. 93.25% / 0.911 for the
earlier Phase-1-only baseline) — every class ≥ 0.89 F1, with Early Blight
(the smallest/weakest class) improving most under Phase 2 fine-tuning.

## 13. Deployment to Streamlit Community Cloud (added — packaging, no `.py` change)

**Files added:** `requirements.txt` (now slim), `requirements-train.txt`,
`.gitignore`, `README.md`; the project was made a git repo.

To let the supervisor/teammates use the app from a permanent public URL (laptop
off), `app.py` is deployed on Streamlit Community Cloud from a GitHub repo.

- **Requirements split.** `app.py` imports only `streamlit`, `tensorflow`,
  `numpy`, `PIL`, `matplotlib` (and `config.py`, stdlib-only). `requirements.txt`
  was slimmed to exactly those, using **`tensorflow-cpu==2.12.0`** (no GPU on
  Streamlit Cloud; smaller/faster install; 2.12.0 is the version proven to load
  `mobilenetv2_best.h5`). The full documented Section 3.9 stack (opencv, sklearn,
  pandas, seaborn, …) was preserved verbatim in **`requirements-train.txt`** for
  training/eval. **Python 3.11 must be selected** in the Streamlit Cloud deploy
  dialog (tensorflow-cpu 2.12.0 has no wheels for 3.12+).

- **What ships in the repo.** Code + `README.md` + `checkpoints/mobilenetv2_best.h5`
  (25.6 MB, under GitHub's 100 MB limit) + the result figures. **`.gitignore`
  excludes** `data/` (180 MB dataset), `.venv/`, `__pycache__/`, the thesis
  `*.docx`, scratch (`files.zip`, `pip_install.log`), and regenerable checkpoint
  artefacts (`dataset_split.npz`, `_last.h5`, `_train_state.json`). Staging was
  verified clean before the first commit.

- **Note:** the deployed app and its GitHub repo are public (anyone with the URL
  can use the app). The dataset is not redistributed (excluded from the repo).

## Summary of code changes

| # | File | Change |
|---|------|--------|
| 1 | `preprocessing.py` | Swapped Gaussian Noise / Shear order in `augment()` to match Table 3.2 |
| 2 | `dataset_preparation.py` | Added `summarize_class_distribution()` (pandas) + call in `__main__` |
| 3 | — | Documentation note only — see above |
| 4 | `config.py` | Added clarifying comment on `FINETUNE_LAYERS` |
| 5 | `requirements.txt`, `preprocessing.py`, `train.py`, docx Section 3.9.2 | TensorFlow 2.11 → 2.12 (protobuf/Streamlit conflict) + numpy 1.24.0 → 1.23.5 (TF 2.12 constraint) |
| 6 | `dataset_preparation.py` | Reconfigure stdout/stderr to UTF-8 in `__main__` to fix Windows `UnicodeEncodeError` on `→` (works here only — no TF import) |
| 7 | `preprocessing.py` | Fixed `_shear_numpy()` `AttributeError` — removed invalid `.astype()` on a Python `float` |
| 8 | `model.py` | Fixed `prepare_phase2()` `ValueError` — locate backbone sub-model by type, not by name |
| 9 | `train.py`, `evaluate.py` | Replaced `─`/`→`/`←` in `print()` calls with ASCII (`reconfigure()` doesn't survive TF import in these files) |
| 10 | `train.py` | Added resume-from-checkpoint (`_last.h5` + `_train_state.json`, auto-resume, `--restart`) so the long CPU run survives interruptions — no methodology change |
| 11 | `app.py` | Fixed deployment preprocessing: resize via `tf.image.resize` (bilinear) instead of PIL `LANCZOS`, to match the training pipeline exactly (max per-pixel skew 0.67 → 0.07) |
| 12 | `Crop_Disease_Colab_GPU.ipynb` (new) | Self-contained Colab T4-GPU runner (~1–2 h): fetches dataset from GitHub (no upload), regenerates split, pins Keras-2 API for `.h5` portability — no `.py` change |
| 13 | `evaluate.py`, `app.py` | Load model with `compile=False` (inference-only) — fixes a cross-Keras-version loss-deserialisation error when loading the Colab-trained `.h5` locally |

`Crop_Disease_Detection_Chapters_1_3_REVISED.docx` was updated for item 5
(Section 3.9.2, TensorFlow version). Items 3 and 4 remain suggested wording
tweaks for the author to apply to Table 3.3 during the next revision pass —
no other docx changes were made.

## Smoke test validation (1+1 epoch dry run, MobileNetV2, CPU)

Before committing to the full 20+30 epoch run (no GPU available — `tf.config.
list_physical_devices('GPU')` returns `[]`), the full `train.py` pipeline was
exercised end-to-end with `PHASE1_EPOCHS=1` / `PHASE2_EPOCHS=1` monkey-patched
in a throwaway script. This run surfaced and validated the fixes in items 7–9
above. Results:

- **Pipeline**: data loading, augmentation (incl. OpenCV shear), model build,
  callbacks, checkpoint save and Phase 1→2 transition all completed without
  errors after the fixes.
- **Phase 1** (1 epoch, backbone frozen, 329,478 trainable params):
  **1,485 s (~24.75 min)**, `val_loss=0.3907`, `val_accuracy=0.8716`.
- **Phase 2** (2 train + 2 val steps only, 1,855,878 trainable params,
  30 backbone layers unfrozen): **40 s for 4 steps** (~10–14 s/step) —
  loss decreased over the 2 steps (0.5226 → 0.5011), confirming gradient flow
  through the unfrozen backbone layers.

**Extrapolated full-run estimate (MobileNetV2, CPU-only)**, using
283 train-batches + 61 val-batches = 344 steps/epoch (batch size 32, 9,055
train / 1,940 val images):

| Phase | Steps/epoch | s/step | min/epoch | Epochs | Total |
|-------|------------:|-------:|----------:|-------:|------:|
| 1 (frozen)     | 344 | ~4.3  | ~25     | 20 | ~8.3 h |
| 2 (fine-tune)  | 344 | ~10–14| ~57–80  | 30 | ~28.5–40 h |
| **Total**      |     |       |         | 50 | **~37–48 h** |

This is a worst-case ceiling (assumes `EarlyStopping`/`ReduceLROnPlateau`
never trigger early). ResNet-50 (25.6M backbone params vs. MobileNetV2's
3.4M) is expected to be slower still, especially in Phase 2. These figures
were reported to the user so they can decide how to run the full training
(e.g. multi-day local background run vs. a GPU runtime such as Colab).

## Dataset acquisition (data/PlantVillage)

The 6 tomato classes were sourced from the public
`spMohanty/PlantVillage-Dataset` GitHub repository (`raw/color/Tomato___*`
folders), via a sparse/partial git clone to avoid downloading the full
~2.1 GB repo. Folders were copied into `data/PlantVillage/` and renamed from
the repo's triple-underscore convention (`Tomato___X`) to the
double-underscore convention in `config.CLASS_FOLDER_NAMES`
(`Tomato__X`). Per-class image counts match Table 3.1 exactly (12,936 total),
confirming this is the same source dataset the documentation was written
against.
