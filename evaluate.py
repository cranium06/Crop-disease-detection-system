# =============================================================================
# evaluate.py
#
# Evaluates a trained model on the held-out test set (15% / 1,941 images).
# Computes and reports all metrics documented in Chapter 3, Section 3.8.
#
# Usage:
#   python evaluate.py --backbone mobilenetv2
#   python evaluate.py --backbone resnet50
#
# Output (saved to checkpoints/):
#   <backbone>_classification_report.txt
#   <backbone>_confusion_matrix.png
#
# IMPORTANT: The test set is loaded from config.SPLIT_FILE (the same split
# saved during dataset_preparation.py). It must never be touched during
# model development — only evaluated once per finalised model, as specified
# in Section 3.4.2: "held out entirely for final unbiased evaluation."
#
# Chapter 3 references:
#   Section 3.4.2  — test set isolation
#   Section 3.8    — evaluation metrics
#   Section 3.8.1  — overall accuracy
#   Section 3.8.2  — precision
#   Section 3.8.3  — recall
#   Section 3.8.4  — F1-score (macro-averaged = primary metric)
#   Section 3.8.5  — confusion matrix (6×6)
#   Section 3.9.3  — scikit-learn 1.2, Matplotlib 3.7, Seaborn 0.12
# =============================================================================

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import tensorflow as tf

import config
import dataset_preparation as dp
import preprocessing as pp
import model as mdl


# ---------------------------------------------------------------------------
# Evaluation orchestrator
# ---------------------------------------------------------------------------

def evaluate(backbone_name: str) -> None:
    """
    Load the best checkpoint and evaluate on the held-out test set.
    Reports all metrics from Section 3.8.
    """
    backbone_name = backbone_name.lower()
    ckpt_path     = (config.MOBILENET_CKPT if backbone_name == "mobilenetv2"
                     else config.RESNET_CKPT)

    print("=" * 65)
    print(f"Evaluation: {backbone_name.upper()}")
    print("Chapter 3, Section 3.8 — Evaluation Metrics")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load test split  (Section 3.4.2)
    # ------------------------------------------------------------------
    if not os.path.exists(config.SPLIT_FILE):
        raise FileNotFoundError(
            f"Split file not found: {config.SPLIT_FILE}\n"
            "Run dataset_preparation.py first."
        )
    _, _, _, _, test_files, test_labels = dp.load_split(config.SPLIT_FILE)
    print(f"\nTest set loaded: {len(test_files)} images  "
          f"(documented: 1,941 — Section 3.4.2)")

    # ------------------------------------------------------------------
    # 2. Build test dataset — NO augmentation  (Section 3.5.3)
    # ------------------------------------------------------------------
    test_ds = pp.build_dataset(
        test_files, test_labels,
        augment_data=False,
        shuffle=False         # Preserve order for label alignment
    )

    # ------------------------------------------------------------------
    # 3. Load best model checkpoint
    # ------------------------------------------------------------------
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run train.py --backbone {backbone_name} first."
        )
    print(f"\nLoading checkpoint: {ckpt_path}")
    # compile=False: only inference (model.predict) is needed here, so the
    # optimiser/loss are not required. This also avoids a cross-Keras-version
    # deserialisation error — a model trained on Colab (Keras 2.15/2.17)
    # serialises the loss with a 'fn' key that local Keras 2.12 rejects when
    # recompiling. Skipping compilation sidesteps it entirely.
    net = tf.keras.models.load_model(ckpt_path, compile=False)
    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # 4. Generate predictions
    # ------------------------------------------------------------------
    print("\nRunning inference on test set...")
    y_pred_proba = net.predict(test_ds, verbose=1)          # [N, 6] probabilities
    y_pred       = np.argmax(y_pred_proba, axis=1)          # Predicted class indices
    y_true       = np.array(test_labels)                    # Ground-truth labels

    # ------------------------------------------------------------------
    # 5. Compute all documented metrics  (Section 3.8)
    # ------------------------------------------------------------------
    overall_acc  = accuracy_score(y_true, y_pred)
    macro_f1     = f1_score(y_true, y_pred, average="macro")
    report_str   = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    cm           = confusion_matrix(y_true, y_pred)

    # ------------------------------------------------------------------
    # 6. Print results
    # ------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 65}")
    print(f"  Backbone           : {backbone_name.upper()}")
    print(f"  Checkpoint         : {ckpt_path}")
    print(f"  Test set size      : {len(y_true)} images")
    print()
    # Section 3.8.1 — Overall accuracy
    print(f"  Overall Accuracy   : {overall_acc * 100:.2f}%")
    # Section 3.8.4 — Macro-averaged F1 (primary summary metric)
    print(f"  Macro F1-Score     : {macro_f1:.4f}  (primary metric, Section 3.8.4)")
    print()
    # Section 3.8.2 / 3.8.3 / 3.8.4 — Per-class precision, recall, F1
    print("  Per-class Classification Report:")
    print(f"{'-' * 65}")
    print(report_str)
    print(f"{'-' * 65}")

    # ------------------------------------------------------------------
    # 7. Save classification report to file
    # ------------------------------------------------------------------
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    report_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{backbone_name}_classification_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(f"Backbone: {backbone_name.upper()}\n")
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Test set size: {len(y_true)} images\n\n")
        f.write(f"Overall Accuracy : {overall_acc * 100:.2f}%\n")
        f.write(f"Macro F1-Score   : {macro_f1:.4f}\n\n")
        f.write("Per-class Classification Report:\n")
        f.write(report_str)
    print(f"Classification report saved to {report_path}")

    # ------------------------------------------------------------------
    # 8. Plot and save confusion matrix  (Section 3.8.5)
    # ------------------------------------------------------------------
    cm_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{backbone_name}_confusion_matrix.png"
    )
    _plot_confusion_matrix(cm, backbone_name, cm_path)
    print(f"Confusion matrix saved to {cm_path}")


# ---------------------------------------------------------------------------
# Confusion matrix visualisation  (Section 3.8.5)
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, backbone_name: str,
                            save_path: str) -> None:
    """
    Plot the 6×6 confusion matrix as a Seaborn heatmap.

    Section 3.8.5: "A 6×6 confusion matrix is computed on the test set to
    provide a granular visualisation of classification behaviour across all
    class pairs, identifying specific disease categories prone to
    misclassification."

    Uses:
        Seaborn 0.12 (Section 3.9.3)
        Matplotlib 3.7 (Section 3.9.3)
    """
    # Compute row-normalised (recall-normalised) matrix for visual clarity
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # Short class labels for readability on the axes
    short_labels = [
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "TYLCV",
        "Bacterial\nSpot",
        "Healthy",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{backbone_name.upper()} — Confusion Matrix (Test Set)\n"
        f"Chapter 3, Section 3.8.5",
        fontsize=13
    )

    # Left: raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].tick_params(axis="x", rotation=30)

    # Right: row-normalised (proportion of each true class correctly predicted)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        ax=axes[1], linewidths=0.5, vmin=0.0, vmax=1.0,
        cbar_kws={"shrink": 0.8}
    )
    axes[1].set_title("Row-Normalised (Recall per Class)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [6×6 confusion matrix — raw counts + row-normalised]")


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
        description="Evaluate Crop Disease Detection model — Chapter 3, Section 3.8"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=mdl.SUPPORTED_BACKBONES,
        help="Backbone architecture to evaluate.",
    )
    args = parser.parse_args()

    evaluate(args.backbone)

    print("\nevaluate.py complete.")
    print("Results are ready for inclusion in Chapter 4.")
