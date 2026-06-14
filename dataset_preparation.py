# =============================================================================
# dataset_preparation.py
#
# Responsibilities:
#   1. Walk the PlantVillage directory and build (filepath, label) lists.
#   2. Apply stratified 70/15/15 split via StratifiedShuffleSplit.
#   3. Compute inverse-frequency class weights for training.
#   4. Persist the split to disk for exact reproducibility.
#
# Chapter 3 references:
#   Section 3.4.1 — dataset composition and class distribution (Table 3.1)
#   Section 3.4.2 — stratified splitting via StratifiedShuffleSplit
#   Section 3.7.1 — inverse-frequency class weighting
#   Section 3.9.3 — scikit-learn 1.2
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import config


# ---------------------------------------------------------------------------
# Step 1 — Build file list
# ---------------------------------------------------------------------------

def build_file_list(dataset_root: str) -> tuple:
    """
    Walk `dataset_root` and collect every JPEG/PNG image path alongside its
    integer class label.  Expected structure:
        dataset_root/CLASS_FOLDER_NAME/image.jpg

    Class folder names and their index order are fixed in config.CLASS_FOLDER_NAMES.
    The index order must be preserved throughout the pipeline.

    Chapter 3, Section 3.4.1, Table 3.1.
    """
    filepaths, labels = [], []

    for class_idx, folder_name in enumerate(config.CLASS_FOLDER_NAMES):
        class_dir = os.path.join(dataset_root, folder_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"\n[dataset_preparation] Directory not found: {class_dir}\n"
                "Check that DATASET_ROOT and CLASS_FOLDER_NAMES in config.py "
                "match your local PlantVillage directory layout."
            )
        found = 0
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(class_idx)
                found += 1
        print(f"  [{class_idx}] {config.CLASS_NAMES[class_idx]:<45} {found:>5} images")

    total = len(filepaths)
    print(f"\n  Total images found: {total}  (documented: 12,936 — Section 3.4.1)")
    if total != 12_936:
        print(f"  WARNING: Expected 12,936 images but found {total}. "
              "Verify your PlantVillage subset matches Table 3.1.")

    return filepaths, labels


# ---------------------------------------------------------------------------
# Step 1b — Class distribution summary (pandas)
# ---------------------------------------------------------------------------

def summarize_class_distribution(labels: list) -> pd.DataFrame:
    """
    Build a pandas DataFrame summarising image counts and proportions per
    class, reproducing the Table 3.1 dataset composition (Section 3.4.1)
    for dataset metadata management and class distribution analysis
    (Section 3.9.3).
    """
    labels_arr = np.array(labels)
    total = len(labels_arr)
    counts = [int((labels_arr == i).sum()) for i in range(config.NUM_CLASSES)]

    df = pd.DataFrame({
        "class_index": range(config.NUM_CLASSES),
        "class_name": config.CLASS_NAMES,
        "folder_name": config.CLASS_FOLDER_NAMES,
        "image_count": counts,
        "percentage": [f"{100.0 * c / total:.1f}%" for c in counts],
    })

    print("\nClass distribution summary (pandas — Table 3.1, Section 3.9.3):")
    print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Step 2 — Stratified 70 / 15 / 15 split
# ---------------------------------------------------------------------------

def stratified_split(filepaths: list, labels: list) -> tuple:
    """
    Perform a reproducible stratified split into train / val / test sets.

    Method (Section 3.4.2):
      Pass 1: StratifiedShuffleSplit → 70% train, 30% temp
      Pass 2: StratifiedShuffleSplit → 50% of temp = 15% val, 50% of temp = 15% test

    Stratification ensures proportional class representation in every subset,
    which is critical given the TYLCV / Leaf Mold imbalance (41.4% vs 7.4%).

    random_state = config.RANDOM_SEED (= 42; assumption — see config.py).
    """
    fps    = np.array(filepaths)
    lbls   = np.array(labels)

    # Pass 1: train (70%) vs temp (30%)
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.VAL_RATIO + config.TEST_RATIO,  # 0.30
        random_state=config.RANDOM_SEED
    )
    train_idx, temp_idx = next(sss1.split(fps, lbls))

    # Pass 2: val (50% of temp = 15%) vs test (50% of temp = 15%)
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=config.RANDOM_SEED
    )
    val_rel_idx, test_rel_idx = next(sss2.split(fps[temp_idx], lbls[temp_idx]))
    val_idx  = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_files  = fps[train_idx].tolist()
    train_labels = lbls[train_idx].tolist()
    val_files    = fps[val_idx].tolist()
    val_labels   = lbls[val_idx].tolist()
    test_files   = fps[test_idx].tolist()
    test_labels  = lbls[test_idx].tolist()

    print(f"\nDataset split  (random_state={config.RANDOM_SEED}):")
    print(f"  Train : {len(train_files):>5}  (documented target: ~9,055 — Section 3.4.2)")
    print(f"  Val   : {len(val_files):>5}  (documented target: ~1,940 — Section 3.4.2)")
    print(f"  Test  : {len(test_files):>5}  (documented target: ~1,941 — Section 3.4.2)")

    # Verify class proportions are maintained in each subset
    _verify_stratification(train_labels, "Train")
    _verify_stratification(val_labels,   "Val  ")
    _verify_stratification(test_labels,  "Test ")

    return train_files, train_labels, val_files, val_labels, test_files, test_labels


def _verify_stratification(labels: list, split_name: str) -> None:
    """Print per-class counts to confirm stratification was applied correctly."""
    labels_arr  = np.array(labels)
    total       = len(labels_arr)
    print(f"\n  {split_name} class distribution:")
    for i, name in enumerate(config.CLASS_NAMES):
        count = int((labels_arr == i).sum())
        pct   = 100.0 * count / total
        print(f"    [{i}] {name:<45} {count:>4}  ({pct:5.1f}%)")


# ---------------------------------------------------------------------------
# Step 3 — Inverse-frequency class weights
# ---------------------------------------------------------------------------

def compute_class_weights(train_labels: list) -> dict:
    """
    Compute class weights inversely proportional to class frequency.

    Applied to the loss function during training to address the pronounced
    imbalance in the dataset (TYLCV 41.4% vs Leaf Mold 7.4%, Table 3.1).

    Chapter 3, Section 3.7.1 — "Class weights inversely proportional to class
    frequencies are computed and applied via TensorFlow/Keras's class_weight
    parameter to offset the dataset imbalance."

    Returns dict {class_index: weight} for Keras class_weight argument.
    """
    labels_arr = np.array(train_labels)
    classes    = np.arange(config.NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr
    )
    weight_dict = {int(i): float(w) for i, w in enumerate(weights)}

    print("\nClass weights  (inverse-frequency — Section 3.7.1):")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"  [{i}] {name:<45}  weight = {weight_dict[i]:.4f}")

    return weight_dict


# ---------------------------------------------------------------------------
# Step 4 — Persist / load split
# ---------------------------------------------------------------------------

def save_split(train_files, train_labels, val_files, val_labels,
               test_files, test_labels, path: str) -> None:
    """
    Persist the split to an .npz file.  This guarantees that every
    subsequent training or evaluation run uses the exact same partitioning,
    satisfying the reproducibility requirement of Section 3.4.2.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        train_files=train_files, train_labels=train_labels,
        val_files=val_files,     val_labels=val_labels,
        test_files=test_files,   test_labels=test_labels,
    )
    print(f"\nSplit saved → {path}")


def load_split(path: str) -> tuple:
    """Load a previously saved split from disk."""
    data = np.load(path, allow_pickle=True)
    return (
        data["train_files"].tolist(), data["train_labels"].tolist(),
        data["val_files"].tolist(),   data["val_labels"].tolist(),
        data["test_files"].tolist(),  data["test_labels"].tolist(),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Ensure Unicode box-drawing/arrow characters in the prints below don't
    # crash on Windows consoles, whose default code page (cp1252) can't
    # encode them.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 65)
    print("dataset_preparation.py")
    print("Chapter 3, Sections 3.4.1 / 3.4.2 / 3.7.1")
    print("=" * 65)

    print(f"\nScanning: {config.DATASET_ROOT}\n")
    fps, lbls = build_file_list(config.DATASET_ROOT)

    summarize_class_distribution(lbls)

    train_f, train_l, val_f, val_l, test_f, test_l = stratified_split(fps, lbls)

    class_weights = compute_class_weights(train_l)

    save_split(train_f, train_l, val_f, val_l, test_f, test_l, config.SPLIT_FILE)

    print("\ndataset_preparation.py complete.")
    print(f"Split file written to: {config.SPLIT_FILE}")
    print("Run train.py next.")
