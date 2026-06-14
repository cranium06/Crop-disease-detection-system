# =============================================================================
# preprocessing.py
#
# Builds tf.data.Dataset pipelines for train, validation, and test splits.
#
# Pipeline stages:
#   All sets:   load image → resize 224×224 → float [0,1] → ImageNet normalise
#   Train only: + 8 augmentation transforms from Table 3.2
#
# Chapter 3 references:
#   Section 3.5.1  — image resizing (224×224)
#   Section 3.5.2  — pixel normalisation ([0,1] then ImageNet mean/std)
#   Section 3.5.3  — data augmentation (Table 3.2, training set only)
#   Section 3.9.2  — TensorFlow 2.12 / tf.data
#   Section 3.9.3  — OpenCV 4.7 (used for shear transform)
# =============================================================================

import cv2
import numpy as np
import tensorflow as tf
import config

# ---------------------------------------------------------------------------
# ImageNet normalisation constants  (Section 3.5.2)
# ---------------------------------------------------------------------------
_MEAN = tf.constant(config.IMAGENET_MEAN, dtype=tf.float32)  # [0.485, 0.456, 0.406]
_STD  = tf.constant(config.IMAGENET_STD,  dtype=tf.float32)  # [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Keras augmentation layers — instantiated once at module level
# (expand_dims / squeeze batch dim required when calling inside Dataset.map)
# ---------------------------------------------------------------------------
# Rotation ±30°  (Table 3.2) — factor = fraction of full 2π rotation
_rot_layer = tf.keras.layers.RandomRotation(
    factor=config.AUG_ROTATION_FACTOR,   # 30/360 ≈ 0.0833
    fill_mode="reflect",
    seed=config.RANDOM_SEED
)
# Zoom 80–100% of original  (Table 3.2) — negative factor = zoom in
_zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=config.AUG_ZOOM_RANGE,  # (-0.2, 0.0)
    width_factor=config.AUG_ZOOM_RANGE,
    fill_mode="reflect",
    seed=config.RANDOM_SEED
)


# ---------------------------------------------------------------------------
# Core load + resize
# ---------------------------------------------------------------------------

def load_and_resize(filepath: tf.Tensor, label: tf.Tensor):
    """
    Read image file from disk, decode JPEG/PNG, resize to 224×224.
    Section 3.5.1 — all images resized to 224×224 pixels.
    """
    raw   = tf.io.read_file(filepath)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH])
    image = tf.cast(image, tf.float32)
    return image, label


# ---------------------------------------------------------------------------
# Normalisation  (Section 3.5.2)
# ---------------------------------------------------------------------------

def normalise(image: tf.Tensor, label: tf.Tensor):
    """
    Two-step normalisation (Section 3.5.2):
      1. Divide by 255.0  →  float range [0.0, 1.0]
      2. Subtract ImageNet channel mean, divide by ImageNet channel std
    """
    image = image / 255.0           # Step 1: [0, 255] → [0.0, 1.0]
    image = (image - _MEAN) / _STD  # Step 2: ImageNet channel normalisation
    return image, label


# ---------------------------------------------------------------------------
# Shear transform via OpenCV  (Table 3.2 — Random Shear ±0.2)
# Implemented with OpenCV 4.7 (Section 3.9.3) via tf.numpy_function.
# Applied in [0.0, 1.0] pixel domain before ImageNet normalisation.
# ---------------------------------------------------------------------------

def _shear_numpy(image_np: np.ndarray) -> np.ndarray:
    """
    OpenCV affine shear. image_np is float32 in [0.0, 1.0], shape [H, W, 3].
    Shear magnitude drawn uniformly from [−0.2, +0.2]  (Table 3.2).
    BORDER_REFLECT_101 avoids hard border artefacts at image edges.
    """
    shear = np.random.uniform(-config.AUG_SHEAR, config.AUG_SHEAR)
    h, w  = image_np.shape[:2]
    M     = np.float32([[1.0, shear, 0.0],
                        [0.0, 1.0,   0.0]])
    return cv2.warpAffine(
        image_np, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def _apply_shear(image: tf.Tensor) -> tf.Tensor:
    """TF wrapper around _shear_numpy using tf.numpy_function."""
    img = tf.numpy_function(_shear_numpy, [image], tf.float32)
    img.set_shape([config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS])
    return img


# ---------------------------------------------------------------------------
# Full augmentation function — applied to TRAINING set only
# (Section 3.5.3: "All augmentations are applied exclusively to the
#  training set and not to the validation or test sets.")
# ---------------------------------------------------------------------------

def augment(image: tf.Tensor, label: tf.Tensor):
    """
    Apply all 8 augmentation transforms from Table 3.2 (Section 3.5.3).
    Input image must be float32 in [0.0, 1.0] (before ImageNet normalisation).
    Operations are applied in the order listed in Table 3.2.
    """
    # 1. Random Horizontal Flip  p = 0.5  (Table 3.2)
    image = tf.image.random_flip_left_right(image, seed=None)

    # 2. Random Vertical Flip  p = 0.5  (Table 3.2)
    image = tf.image.random_flip_up_down(image, seed=None)

    # 3. Random Rotation  ±30°  (Table 3.2)
    #    Keras layer requires a batch dimension; squeeze it back after.
    image = tf.squeeze(
        _rot_layer(tf.expand_dims(image, 0), training=True), axis=0
    )

    # 4. Random Zoom / Crop  80–100% of original  (Table 3.2)
    image = tf.squeeze(
        _zoom_layer(tf.expand_dims(image, 0), training=True), axis=0
    )

    # 5. Brightness Adjustment  ±20%  (Table 3.2)
    image = tf.image.random_brightness(image, max_delta=config.AUG_BRIGHTNESS)

    # 6. Contrast Adjustment  ±20%  (Table 3.2)
    image = tf.image.random_contrast(
        image,
        lower=config.AUG_CONTRAST_LOWER,  # 0.8
        upper=config.AUG_CONTRAST_UPPER   # 1.2
    )

    # Clip to [0, 1] before noise and shear to ensure valid input range
    image = tf.clip_by_value(image, 0.0, 1.0)

    # 7. Gaussian Noise  σ = 0.01  (Table 3.2)
    noise = tf.random.normal(
        shape=[config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS],
        mean=0.0,
        stddev=config.AUG_GAUSS_SIGMA
    )
    image = image + noise

    # 8. Random Shear  ±0.2  (Table 3.2)  — via OpenCV tf.numpy_function
    image = _apply_shear(image)

    # Final clip: ensure augmented image stays in [0.0, 1.0] before normalisation
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


# ---------------------------------------------------------------------------
# Public API — build_dataset
# ---------------------------------------------------------------------------

def build_dataset(
    filepaths: list,
    labels: list,
    *,
    augment_data: bool,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """
    Build a fully-configured tf.data.Dataset for one data split.

    Args:
        filepaths:    List of image file paths.
        labels:       List of integer class labels (same length as filepaths).
        augment_data: True  → training pipeline (load + resize + augment + normalise)
                      False → val/test pipeline  (load + resize + normalise only)
                      Section 3.5.3: augmentation applied to TRAINING set only.
        batch_size:   Defaults to config.BATCH_SIZE (32 — Table 3.3).
        shuffle:      Shuffle before batching; should be True for training.

    Returns:
        tf.data.Dataset yielding (image, label) batches.
        Images are float32 tensors of shape [batch, 224, 224, 3], ImageNet-normalised.
        Labels are int32 tensors of shape [batch].
    """
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=config.RANDOM_SEED,
                        reshuffle_each_iteration=True)

    # Load and resize every image (all splits)
    ds = ds.map(load_and_resize, num_parallel_calls=AUTOTUNE)

    if augment_data:
        # Training: scale to [0,1] → augment → normalise
        ds = ds.map(
            lambda img, lbl: (img / 255.0, lbl),
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
        # ImageNet normalisation applied AFTER augmentation
        ds = ds.map(
            lambda img, lbl: ((img - _MEAN) / _STD, lbl),
            num_parallel_calls=AUTOTUNE
        )
    else:
        # Validation / test: scale + normalise only (no augmentation)
        ds = ds.map(normalise, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Pipeline sanity check
# ---------------------------------------------------------------------------

def verify_pipeline(train_ds: tf.data.Dataset) -> None:
    """
    Trace one batch through the pipeline to confirm shapes and value ranges.
    Run this after building datasets to catch normalisation/shape errors early.
    """
    for images, labels in train_ds.take(1):
        print("\nPipeline verification (one training batch):")
        print(f"  Image batch shape : {images.shape}  (expected [32, 224, 224, 3])")
        print(f"  Label batch shape : {labels.shape}  (expected [32])")
        print(f"  Pixel value min   : {float(tf.reduce_min(images)):.4f}")
        print(f"  Pixel value max   : {float(tf.reduce_max(images)):.4f}")
        print(f"  Pixel value mean  : {float(tf.reduce_mean(images)):.4f}")
        print("  (ImageNet-normalised values typically span approx [-2.1, 2.6])")
        unique_labels = sorted(set(labels.numpy().tolist()))
        print(f"  Unique labels in batch: {unique_labels}")
