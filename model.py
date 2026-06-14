# =============================================================================
# model.py
#
# Builds the CNN classification models documented in Chapter 3, Section 3.6.
#
# Two models are constructed under identical conditions:
#   - MobileNetV2  (primary model — Section 3.6.1)
#   - ResNet-50    (comparative model — Section 3.6.3)
#
# Classification head (Section 3.6.2), applied identically to both backbones:
#   GlobalAveragePooling2D
#   → Dense(256, ReLU, L2 λ=1e-4)
#   → Dropout(0.5)
#   → Dense(6, Softmax)
#
# Training phases (Section 3.7.3):
#   Phase 1: backbone fully frozen, head trained (LR = 0.001, 20 epochs)
#   Phase 2: top 30 backbone layers + head unfrozen (LR = 1e-5, 30 epochs)
# =============================================================================

import tensorflow as tf
import config


# ---------------------------------------------------------------------------
# Supported backbone names
# ---------------------------------------------------------------------------
SUPPORTED_BACKBONES = ("mobilenetv2", "resnet50")


# ---------------------------------------------------------------------------
# Internal helper — load the pretrained backbone
# ---------------------------------------------------------------------------

def _get_backbone(backbone_name: str) -> tf.keras.Model:
    """
    Load ImageNet-pretrained backbone with the top classification layer removed.

    MobileNetV2 (Section 3.6.1):
        72.0% top-1 ImageNet accuracy, 3.4M parameters.
        Inverted residual blocks with depthwise separable convolutions.
        Output feature map: 7×7×1280 before GlobalAveragePooling.

    ResNet-50 (Section 3.6.3):
        75.2% top-1 ImageNet accuracy, 25.6M parameters.
        50-layer bottleneck residual network.
    """
    kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
    )

    if backbone_name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(**kwargs)
    elif backbone_name == "resnet50":
        base = tf.keras.applications.ResNet50(**kwargs)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Supported: {SUPPORTED_BACKBONES}"
        )

    print(f"\n  Backbone : {backbone_name.upper()}")
    print(f"  Layers   : {len(base.layers)}")
    print(f"  Parameters (backbone): {base.count_params():,}")
    return base


# ---------------------------------------------------------------------------
# Custom classification head  (Section 3.6.2)
# ---------------------------------------------------------------------------

def _build_head(backbone_output: tf.Tensor) -> tf.Tensor:
    """
    Append the custom classification head to the backbone output tensor.

    Architecture (Section 3.6.2):
      (i)  GlobalAveragePooling2D
              Reduces 7×7×1280 (MobileNetV2) or 7×7×2048 (ResNet-50) to a
              flat feature vector. Provides spatial translation invariance.
      (ii) Dense(256, ReLU)  with L2 kernel regularisation (λ = 1×10⁻⁴)
              Learns disease-discriminating feature combinations.
      (iii) Dropout(0.5)
              Randomly zeroes 50% of activations during training to prevent
              co-adaptation and reduce overfitting.
      (iv) Dense(6, Softmax)
              Produces probability distribution over the 6 classes.
    """
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(backbone_output)

    # Section 3.6.2 — Dense(256, ReLU, L2 λ=1e-4)
    x = tf.keras.layers.Dense(
        units=config.DENSE_UNITS,          # 256
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.L2_LAMBDA),  # 1e-4
        name="dense_256"
    )(x)

    # Section 3.6.2 — Dropout(0.5)
    x = tf.keras.layers.Dropout(
        rate=config.DROPOUT_RATE,          # 0.5
        name="dropout_05"
    )(x)

    # Section 3.6.2 — Dense(6, Softmax)
    output = tf.keras.layers.Dense(
        units=config.NUM_CLASSES,          # 6
        activation="softmax",
        name="output_softmax"
    )(x)

    return output


# ---------------------------------------------------------------------------
# Public API — build_model
# ---------------------------------------------------------------------------

def build_model(backbone_name: str) -> tf.keras.Model:
    """
    Construct the full model (backbone + head) with the backbone fully frozen.
    Ready for Phase 1 training.

    Args:
        backbone_name: "mobilenetv2" or "resnet50"

    Returns:
        Compiled tf.keras.Model with backbone frozen (Phase 1 configuration).
        Optimiser: Adam (β₁=0.9, β₂=0.999, ε=1e-7)  — Section 3.7.2
        Loss:      Categorical cross-entropy            — Section 3.7.1
        Metrics:   accuracy
    """
    backbone_name = backbone_name.lower()
    if backbone_name not in SUPPORTED_BACKBONES:
        raise ValueError(f"backbone_name must be one of {SUPPORTED_BACKBONES}")

    # Load pretrained backbone
    base_model = _get_backbone(backbone_name)

    # Phase 1: freeze the entire backbone (Section 3.7.3)
    base_model.trainable = False
    frozen_count = sum(1 for l in base_model.layers if not l.trainable)
    print(f"  Phase 1: backbone fully frozen ({frozen_count} layers).")

    # Build model graph
    inputs = tf.keras.Input(
        shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        name="input_image"
    )
    features = base_model(inputs, training=False)  # training=False keeps BN frozen
    outputs  = _build_head(features)
    model    = tf.keras.Model(inputs=inputs, outputs=outputs,
                              name=f"{backbone_name}_phase1")

    # Compile for Phase 1  (Section 3.7.2, 3.7.1)
    _compile_model(model, lr=config.PHASE1_LR)

    total_params    = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable (Phase 1) : {trainable_params:,}  (head only)")

    return model


# ---------------------------------------------------------------------------
# Phase 2 — unfreeze top N backbone layers for fine-tuning
# ---------------------------------------------------------------------------

def prepare_phase2(model: tf.keras.Model, backbone_name: str) -> tf.keras.Model:
    """
    Configure the model for Phase 2 fine-tuning (Section 3.7.3):
      - Unfreeze the top FINETUNE_LAYERS (= 30) backbone layers.
      - Keep all earlier backbone layers frozen.
      - Recompile with the Phase 2 learning rate (LR = 1×10⁻⁵).

    IMPORTANT: model.compile() MUST be called after changing layer trainability
    for the new learning rate to take effect — this function handles that.

    Args:
        model:         The model returned by build_model() after Phase 1 training.
        backbone_name: "mobilenetv2" or "resnet50" (to locate the backbone sub-model)

    Returns:
        The same model object, reconfigured and recompiled for Phase 2.
    """
    backbone_name = backbone_name.lower()

    # Retrieve the backbone sub-model. tf.keras.applications constructors
    # assign their own model names (e.g. "mobilenetv2_1.00_224", "resnet50"),
    # which do not match `backbone_name`, so locate the backbone by type
    # instead: it is the only nested tf.keras.Model layer in this
    # architecture (the head consists of GAP/Dense/Dropout layers).
    base_model = next(
        layer for layer in model.layers if isinstance(layer, tf.keras.Model)
    )

    # Unfreeze only the top FINETUNE_LAYERS layers (Section 3.6.1 / 3.7.3)
    base_model.trainable = True
    for layer in base_model.layers[:-config.FINETUNE_LAYERS]:
        layer.trainable = False

    frozen_count    = sum(1 for l in base_model.layers if not l.trainable)
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"\n  Phase 2: {trainable_count} backbone layers unfrozen "
          f"(top {config.FINETUNE_LAYERS}), {frozen_count} remain frozen.")

    # Recompile with Phase 2 learning rate (Section 3.7.3)
    _compile_model(model, lr=config.PHASE2_LR)
    model._name = f"{backbone_name}_phase2"

    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Trainable (Phase 2) : {trainable_params:,}  (top backbone + head)")

    return model


# ---------------------------------------------------------------------------
# Internal compiler helper
# ---------------------------------------------------------------------------

def _compile_model(model: tf.keras.Model, lr: float) -> None:
    """
    Compile model with Adam optimiser and categorical cross-entropy loss.
    Sections 3.7.1 (loss), 3.7.2 (optimiser), 3.8.1 (accuracy metric).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=config.ADAM_BETA1,     # 0.9    — Section 3.7.2
            beta_2=config.ADAM_BETA2,     # 0.999  — Section 3.7.2
            epsilon=config.ADAM_EPSILON,  # 1e-7   — Section 3.7.2
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        # SparseCategoricalCrossentropy is used because labels are integers.
        # This is functionally identical to CategoricalCrossentropy on one-hot
        # labels, as specified in Section 3.7.1.
        metrics=["accuracy"],
    )
    print(f"  Compiled: Adam LR={lr}, SparseCategoricalCrossentropy")


# ---------------------------------------------------------------------------
# Training callbacks factory  (Section 3.7.4)
# ---------------------------------------------------------------------------

def get_callbacks(checkpoint_path: str) -> list:
    """
    Build the three training callbacks documented in Section 3.7.4.
    New callback instances must be created for each training phase to reset
    internal state (patience counters, best-loss trackers).

    (i)  ModelCheckpoint  — saves weights at lowest val_loss epoch
    (ii) EarlyStopping    — stops if val_loss does not improve for 10 epochs
    (iii) ReduceLROnPlateau — reduces LR by 0.2× if val_loss stagnates for 5 epochs
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.EARLY_STOP_PATIENCE,   # 10 — Section 3.7.4
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,        # 0.2  — Section 3.7.4
        patience=config.REDUCE_LR_PATIENCE,    # 5    — Section 3.7.4
        min_lr=config.REDUCE_LR_MIN_LR,        # 1e-7 — Section 3.7.4
        verbose=1,
    )

    return [checkpoint, early_stop, reduce_lr]
