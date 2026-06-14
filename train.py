# =============================================================================
# train.py
#
# Two-phase training script for MobileNetV2 and ResNet-50.
#
# Usage:
#   python train.py --backbone mobilenetv2   # Primary model (Section 3.6.1)
#   python train.py --backbone resnet50      # Comparative model (Section 3.6.3)
#
# Prerequisites:
#   Run dataset_preparation.py first to generate config.SPLIT_FILE.
#
# Output:
#   Best model weights saved to checkpoints/<backbone>_best.h5
#   Training history plots saved to checkpoints/<backbone>_history.png
#
# Chapter 3 references:
#   Section 3.4.2  — dataset splits
#   Section 3.7.3  — two-phase training and hyperparameters (Table 3.3)
#   Section 3.7.4  — training callbacks
#   Section 3.9.2  — TensorFlow 2.12
# =============================================================================

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Colab / headless environments
import matplotlib.pyplot as plt
import tensorflow as tf

import config
import dataset_preparation as dp
import preprocessing as pp
import model as mdl


# ---------------------------------------------------------------------------
# Resume-from-checkpoint support  (operational tooling — NOT part of the
# Chapter 3 methodology; it does not change the model, hyperparameters, or
# the 20+30-epoch schedule, only lets an interrupted run be continued).
#
# On every epoch, the full model (weights + optimiser state) is written to
# `<backbone>_last.h5` and a small JSON state file records how far training
# has progressed. Re-invoking train.py then continues where it stopped. This
# is independent of the methodological `<backbone>_best.h5` (best-val_loss)
# checkpoint produced by ModelCheckpoint, which is left untouched.
# ---------------------------------------------------------------------------

def _resume_paths(ckpt_path: str):
    """Derive the `_last.h5` and `_train_state.json` paths from `_best.h5`."""
    if ckpt_path.endswith("_best.h5"):
        base = ckpt_path[: -len("_best.h5")]
    else:
        base = os.path.splitext(ckpt_path)[0]
    return base + "_last.h5", base + "_train_state.json"


def _load_state(state_path: str):
    """Return the saved training-state dict, or None if absent/unreadable."""
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(state_path, backbone, phase, done_p1, done_p2, hist):
    """Persist the training-state JSON (called after each epoch)."""
    state = {
        "backbone": backbone,
        "phase": phase,
        "completed_epochs_phase1": done_p1,
        "completed_epochs_phase2": done_p2,
        "history": hist,
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


class _ResumeCheckpoint(tf.keras.callbacks.Callback):
    """
    After every epoch, save the full model (weights + optimiser state) to
    `last_ckpt` and append the epoch's metrics to the accumulated `hist`
    dict, then write the resume-state JSON. The model is saved before the
    state file so the JSON never claims an epoch the checkpoint does not yet
    contain (worst case on a crash mid-save: one epoch is harmlessly redone).
    """

    def __init__(self, backbone, phase, hist, last_ckpt, state_path):
        super().__init__()
        self.backbone = backbone
        self.phase = phase
        self.hist = hist
        self.last_ckpt = last_ckpt
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
            self.hist[key].append(float(logs.get(key, float("nan"))))
        if self.phase == 1:
            self.hist["phase1_len"] = len(self.hist["loss"])

        completed = epoch + 1  # `epoch` is the absolute (initial_epoch-based) index
        self.model.save(self.last_ckpt)
        _save_state(
            self.state_path, self.backbone, self.phase,
            done_p1=(completed if self.phase == 1 else config.PHASE1_EPOCHS),
            done_p2=(completed if self.phase == 2 else 0),
            hist=self.hist,
        )


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------

def train(backbone_name: str, resume: bool = True) -> None:
    """
    Execute the full two-phase training pipeline for one backbone.

    Phase 1 (Section 3.7.3):
        Backbone fully frozen. Only the classification head is trained.
        Epochs: 20  |  LR: 0.001  |  Batch: 32

    Phase 2 (Section 3.7.3):
        Top 30 backbone layers + head unfrozen.
        Epochs: 30  |  LR: 1×10⁻⁵  |  Batch: 32

    Args:
        resume: If True (default) and a `<backbone>_train_state.json` from a
                previous interrupted run exists, continue from where it
                stopped. If False, start fresh and discard any saved state.
    """
    backbone_name = backbone_name.lower()
    ckpt_path = (config.MOBILENET_CKPT if backbone_name == "mobilenetv2"
                 else config.RESNET_CKPT)
    last_ckpt, state_path = _resume_paths(ckpt_path)

    print("=" * 65)
    print(f"Training: {backbone_name.upper()}")
    print(f"Chapter 3, Section 3.7 - Two-Phase Training Procedure")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load split  (Section 3.4.2)
    # ------------------------------------------------------------------
    if not os.path.exists(config.SPLIT_FILE):
        raise FileNotFoundError(
            f"Split file not found: {config.SPLIT_FILE}\n"
            "Run dataset_preparation.py first."
        )
    train_f, train_l, val_f, val_l, _, _ = dp.load_split(config.SPLIT_FILE)
    print(f"\nLoaded split from {config.SPLIT_FILE}")
    print(f"  Train: {len(train_f)} images  |  Val: {len(val_f)} images")

    # ------------------------------------------------------------------
    # 2. Compute class weights  (Section 3.7.1)
    # ------------------------------------------------------------------
    class_weights = dp.compute_class_weights(train_l)

    # ------------------------------------------------------------------
    # 3. Build tf.data pipelines  (Section 3.5)
    # ------------------------------------------------------------------
    print("\nBuilding tf.data pipelines...")
    train_ds = pp.build_dataset(train_f, train_l, augment_data=True,  shuffle=True)
    val_ds   = pp.build_dataset(val_f,   val_l,   augment_data=False, shuffle=False)

    # Verify pipeline shapes and normalisation range
    pp.verify_pipeline(train_ds)

    # ------------------------------------------------------------------
    # 4. Build a fresh model, or resume from a previous interrupted run
    #    (Section 3.6 for the fresh build)
    # ------------------------------------------------------------------
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    if not resume:
        # Discard any stale resume artefacts so this is a clean restart.
        for p in (last_ckpt, state_path):
            if os.path.exists(p):
                os.remove(p)

    state = _load_state(state_path) if resume else None
    if state is not None and state.get("backbone") != backbone_name:
        print(f"\n  [resume] Saved state is for '{state.get('backbone')}', "
              f"not '{backbone_name}' — starting fresh.")
        state = None

    if state is None:
        print(f"\nBuilding {backbone_name.upper()} model (fresh start)...")
        net = mdl.build_model(backbone_name)
        phase, done_p1, done_p2 = 1, 0, 0
        hist = {"loss": [], "val_loss": [], "accuracy": [],
                "val_accuracy": [], "phase1_len": 0}
    else:
        phase   = state["phase"]
        done_p1 = state["completed_epochs_phase1"]
        done_p2 = state["completed_epochs_phase2"]
        hist    = state["history"]
        print(f"\n[resume] Continuing from saved state: phase {phase}, "
              f"Phase 1 {done_p1}/{config.PHASE1_EPOCHS}, "
              f"Phase 2 {done_p2}/{config.PHASE2_EPOCHS} epochs done.")
        print(f"[resume] Loading model (weights + optimiser) from {last_ckpt} ...")
        net = tf.keras.models.load_model(last_ckpt)

    # ------------------------------------------------------------------
    # 5. Phase 1 — Feature Extraction  (Section 3.7.3)
    #    Backbone frozen, head only.
    #    Epochs: 20  |  LR: 0.001  |  Batch size: 32
    # ------------------------------------------------------------------
    if phase == 1 and done_p1 < config.PHASE1_EPOCHS:
        print(f"\n{'-' * 50}")
        print(f"Phase 1 - Feature Extraction"
              f"{f'  (resuming at epoch {done_p1 + 1})' if done_p1 else ''}")
        print(f"  Epochs : {config.PHASE1_EPOCHS}")
        print(f"  LR     : {config.PHASE1_LR}")
        print(f"  Batch  : {config.BATCH_SIZE}")
        print(f"{'-' * 50}")

        callbacks_p1 = mdl.get_callbacks(ckpt_path) + [
            _ResumeCheckpoint(backbone_name, 1, hist, last_ckpt, state_path)
        ]
        net.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=done_p1,
            epochs=config.PHASE1_EPOCHS,
            class_weight=class_weights,   # Inverse-frequency weighting — Section 3.7.1
            callbacks=callbacks_p1,
            verbose=1,
        )
        done_p1 = config.PHASE1_EPOCHS
        print(f"\nPhase 1 complete.")
        if hist["val_loss"]:
            print(f"  Best Phase 1 val_loss     : {min(hist['val_loss']):.4f}")
            print(f"  Best Phase 1 val_accuracy : {max(hist['val_accuracy']):.4f}")

    # ------------------------------------------------------------------
    # 6. Transition to Phase 2 configuration  (Section 3.7.3)
    #    Only when the in-memory model is still in Phase 1 form. When resuming
    #    directly into Phase 2 the loaded `_last.h5` already carries the
    #    unfrozen layers + LR=1e-5, so this is skipped to avoid re-preparing.
    # ------------------------------------------------------------------
    if phase == 1:
        net = mdl.prepare_phase2(net, backbone_name)
        phase, done_p2 = 2, 0
        # Persist the transition immediately so a crash before the first
        # Phase 2 epoch still resumes into Phase 2 (not back into Phase 1).
        net.save(last_ckpt)
        _save_state(state_path, backbone_name, phase, config.PHASE1_EPOCHS, 0, hist)

    # ------------------------------------------------------------------
    # 7. Phase 2 — Fine-Tuning  (Section 3.7.3)
    #    Top 30 backbone layers + head unfrozen.
    #    Epochs: 30  |  LR: 1×10⁻⁵  |  Batch size: 32
    # ------------------------------------------------------------------
    if done_p2 < config.PHASE2_EPOCHS:
        print(f"\n{'-' * 50}")
        print(f"Phase 2 - Fine-Tuning"
              f"{f'  (resuming at epoch {done_p2 + 1})' if done_p2 else ''}")
        print(f"  Epochs               : {config.PHASE2_EPOCHS}")
        print(f"  LR                   : {config.PHASE2_LR}")
        print(f"  Layers unfrozen      : top {config.FINETUNE_LAYERS} of backbone")
        print(f"  Batch                : {config.BATCH_SIZE}")
        print(f"{'-' * 50}")

        # Fresh callbacks — reset patience counters for Phase 2
        callbacks_p2 = mdl.get_callbacks(ckpt_path) + [
            _ResumeCheckpoint(backbone_name, 2, hist, last_ckpt, state_path)
        ]
        net.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=done_p2,
            epochs=config.PHASE2_EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks_p2,
            verbose=1,
        )
        print(f"\nPhase 2 complete.")

    # ------------------------------------------------------------------
    # 8. Report + plot full (both-phase) training history
    # ------------------------------------------------------------------
    if hist["val_loss"]:
        print(f"  Best overall val_loss     : {min(hist['val_loss']):.4f}")
        print(f"  Best overall val_accuracy : {max(hist['val_accuracy']):.4f}")
    print(f"\nBest model saved to {ckpt_path}")

    plot_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{backbone_name}_training_history.png"
    )
    _plot_history(hist, backbone_name, plot_path)
    print(f"Training curves saved to {plot_path}")


# ---------------------------------------------------------------------------
# Training history plot
# ---------------------------------------------------------------------------

def _plot_history(hist: dict, backbone_name: str, save_path: str) -> None:
    """
    Plot training/validation loss and accuracy across both phases.
    Phase 1 and Phase 2 are shown on a continuous x-axis with a separator.
    Matplotlib 3.7  (Section 3.9.3).

    `hist` is the accumulated history dict maintained across (possibly
    resumed) runs: continuous lists for loss/val_loss/accuracy/val_accuracy
    plus `phase1_len`, the epoch index at which Phase 1 ended.
    """
    train_loss = hist["loss"]
    val_loss   = hist["val_loss"]
    train_acc  = hist["accuracy"]
    val_acc    = hist["val_accuracy"]

    if not train_loss:
        print("  (no epochs recorded — skipping training-history plot)")
        return

    phase1_end = hist.get("phase1_len") or 0
    phase2_len = len(train_loss) - phase1_end
    epochs     = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"{backbone_name.upper()} — Training History\n"
        f"Chapter 3, Section 3.7.3 — Two-Phase Training",
        fontsize=13
    )

    for ax, (train_metric, val_metric, ylabel) in zip(
        axes,
        [(train_loss, val_loss, "Categorical Cross-Entropy Loss"),
         (train_acc,  val_acc,  "Accuracy")]
    ):
        ax.plot(epochs, train_metric, label="Training",   color="#1f77b4")
        ax.plot(epochs, val_metric,   label="Validation", color="#ff7f0e")
        ax.axvline(x=phase1_end + 0.5, color="grey",
                   linestyle="--", linewidth=1.2, label="Phase 1 → 2")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        # Annotate phase labels
        ax.text(phase1_end / 2, ax.get_ylim()[1] * 0.97,
                "Phase 1\n(frozen)", ha="center", va="top",
                fontsize=9, color="grey")
        ax.text(phase1_end + phase2_len / 2,
                ax.get_ylim()[1] * 0.97,
                "Phase 2\n(fine-tune)", ha="center", va="top",
                fontsize=9, color="grey")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure Unicode box-drawing/arrow characters in the prints below don't
    # crash on Windows consoles, whose default code page (cp1252) can't
    # encode them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Train Crop Disease Detection model — Chapter 3, Section 3.7"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=mdl.SUPPORTED_BACKBONES,
        help=(
            "Backbone architecture to train.\n"
            "  mobilenetv2 — primary model (Section 3.6.1)\n"
            "  resnet50    — comparative model (Section 3.6.3)"
        ),
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore any saved resume state and train from scratch. "
            "By default, training auto-resumes from the last completed epoch "
            "if a previous run was interrupted."
        ),
    )
    args = parser.parse_args()

    train(args.backbone, resume=not args.restart)

    print("\ntrain.py complete. Run evaluate.py to assess test-set performance.")
