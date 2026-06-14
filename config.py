# =============================================================================
# config.py  —  Single source of truth for all pipeline constants
#
# Every numeric value here is traced directly to the approved project
# documentation (Chapters 1–3). Do NOT modify any value without first
# updating the corresponding section in the approved chapter document.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# Dataset  (Chapter 3, Section 3.4)
# -----------------------------------------------------------------------------

DATASET_ROOT = "data/PlantVillage"

# PlantVillage sub-folder names for the 6 tomato classes.
# NOTE: Exact folder names differ between PlantVillage download sources.
# Adjust CLASS_FOLDER_NAMES to match your local directory if needed.
# The index order here is FIXED and must be identical at every pipeline stage:
#   preprocessing → training → Streamlit inference output.
CLASS_FOLDER_NAMES = [
    "Tomato__Early_blight",                     # 1,000 images  (Table 3.1)
    "Tomato__Late_blight",                       # 1,909 images  (Table 3.1)
    "Tomato__Leaf_Mold",                         # 952 images    (Table 3.1)
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",     # 5,357 images  (Table 3.1)
    "Tomato__Bacterial_spot",                    # 2,127 images  (Table 3.1)
    "Tomato__healthy",                           # 1,591 images  (Table 3.1)
]

# Human-readable class names — index-aligned with CLASS_FOLDER_NAMES
CLASS_NAMES = [
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Yellow Leaf Curl Virus (TYLCV)",
    "Tomato Bacterial Spot",
    "Tomato Healthy",
]

NUM_CLASSES = 6  # Section 3.4.1

# Dataset split ratios (Section 3.4.2)
TRAIN_RATIO = 0.70   # → 9,055 images
VAL_RATIO   = 0.15   # → 1,940 images
TEST_RATIO  = 0.15   # → 1,941 images

# Random seed — not specified in approved documentation.
# ASSUMPTION: seed=42 (standard academic ML default).
# Document this assumption in Chapter 4 if deviating from it.
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Preprocessing  (Chapter 3, Section 3.5)
# -----------------------------------------------------------------------------

IMAGE_HEIGHT   = 224   # Section 3.5.1
IMAGE_WIDTH    = 224   # Section 3.5.1
IMAGE_CHANNELS = 3     # RGB

# Pixel normalisation (Section 3.5.2):
#   Step 1 — divide raw [0,255] by 255.0  →  [0.0, 1.0]
#   Step 2 — apply ImageNet channel-wise mean/std
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # Section 3.5.2
IMAGENET_STD  = [0.229, 0.224, 0.225]   # Section 3.5.2

# Augmentation parameters (Table 3.2, Section 3.5.3)
# Applied to TRAINING set only; validation and test sets receive no augmentation.
AUG_ROTATION_FACTOR = 30 / 360    # ±30° expressed as fraction of full 2π rotation
AUG_ZOOM_RANGE      = (-0.2, 0.0) # 80–100% of original (negative = zoom in)
AUG_BRIGHTNESS      = 0.2         # ±20% brightness delta
AUG_CONTRAST_LOWER  = 0.8         # lower bound of contrast factor  (1 − 0.2)
AUG_CONTRAST_UPPER  = 1.2         # upper bound of contrast factor  (1 + 0.2)
AUG_GAUSS_SIGMA     = 0.01        # Gaussian noise σ (applied in [0,1] domain)
AUG_SHEAR           = 0.2         # Shear intensity ±0.2

# -----------------------------------------------------------------------------
# Model architecture  (Chapter 3, Section 3.6)
# -----------------------------------------------------------------------------

DENSE_UNITS     = 256    # Section 3.6.2 — dense hidden layer units
L2_LAMBDA       = 1e-4   # Section 3.6.2 — L2 kernel regularisation factor (λ)
DROPOUT_RATE    = 0.5    # Section 3.6.2 — dropout rate
FINETUNE_LAYERS = 30     # Section 3.6.1 — top N backbone layers unfrozen in Phase 2
                         # Applied identically to MobileNetV2 and ResNet-50, per the
                         # "identical conditions" comparison in Section 3.6.3.
                         # See IMPLEMENTATION_NOTES.md for the Table 3.3 wording note.

# -----------------------------------------------------------------------------
# Training  (Chapter 3, Section 3.7 / Table 3.3)
# -----------------------------------------------------------------------------

BATCH_SIZE = 32   # Table 3.3

# Phase 1 — frozen backbone feature extraction (Section 3.7.3)
PHASE1_EPOCHS = 20
PHASE1_LR     = 0.001

# Phase 2 — fine-tuning top FINETUNE_LAYERS backbone layers (Section 3.7.3)
PHASE2_EPOCHS = 30
PHASE2_LR     = 1e-5

# Adam optimiser hyperparameters (Section 3.7.2)
ADAM_BETA1   = 0.9
ADAM_BETA2   = 0.999
ADAM_EPSILON = 1e-7

# Callback settings (Section 3.7.4)
EARLY_STOP_PATIENCE = 10    # EarlyStopping patience
REDUCE_LR_PATIENCE  = 5     # ReduceLROnPlateau patience
REDUCE_LR_FACTOR    = 0.2   # ReduceLROnPlateau reduction factor
REDUCE_LR_MIN_LR    = 1e-7  # ReduceLROnPlateau minimum learning rate

# -----------------------------------------------------------------------------
# Output paths
# -----------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints"
SPLIT_FILE     = os.path.join(CHECKPOINT_DIR, "dataset_split.npz")
MOBILENET_CKPT = os.path.join(CHECKPOINT_DIR, "mobilenetv2_best.h5")
RESNET_CKPT    = os.path.join(CHECKPOINT_DIR, "resnet50_best.h5")

# -----------------------------------------------------------------------------
# Disease management recommendations  (Chapter 3, Section 3.3)
# Displayed in the Streamlit application alongside the classification result.
# -----------------------------------------------------------------------------

MANAGEMENT = {
    "Tomato Early Blight": (
        "**Immediate action:** Remove and destroy all visibly infected lower leaves. "
        "Apply a registered fungicide (e.g., chlorothalonil or mancozeb) at 7–10 day intervals. "
        "Avoid overhead irrigation — water only at the base of the plant. "
        "Practise crop rotation: avoid planting tomato in the same bed for at least two seasons."
    ),
    "Tomato Late Blight": (
        "**Urgent — act immediately.** Late Blight can destroy an entire crop within days. "
        "Remove and bag all infected material; do not compost. "
        "Apply a systemic fungicide containing metalaxyl or cymoxanil without delay. "
        "Improve air circulation, avoid wet foliage, and monitor remaining plants daily."
    ),
    "Tomato Leaf Mold": (
        "Reduce humidity and ensure adequate ventilation between plants. "
        "Remove and destroy infected leaves. "
        "Apply a fungicide effective against Passalora fulva (e.g., copper-based products). "
        "Increase plant spacing to improve airflow through the canopy."
    ),
    "Tomato Yellow Leaf Curl Virus (TYLCV)": (
        "**No cure exists** — management focuses on prevention and vector control. "
        "Remove and destroy all infected plants immediately to limit spread. "
        "Apply insecticide to control the whitefly (Bemisia tabaci) vector population. "
        "Deploy reflective mulches and yellow sticky traps to deter whitefly. "
        "Plant TYLCV-resistant varieties in future seasons where available."
    ),
    "Tomato Bacterial Spot": (
        "Remove infected material promptly; avoid handling plants when wet. "
        "Apply copper-based bactericide as a protective spray at 7-day intervals. "
        "Use only certified disease-free seed, or hot-water treat seed before planting. "
        "Disinfect tools between plants and avoid overhead irrigation."
    ),
    "Tomato Healthy": (
        "No disease detected — your tomato plant appears healthy. "
        "Continue weekly monitoring: inspect both leaf surfaces for early symptoms. "
        "Maintain good cultural practices: balanced fertilisation, base-level irrigation, "
        "and adequate plant spacing to minimise humidity and promote airflow."
    ),
}
