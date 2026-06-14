# =============================================================================
# app.py
#
# Streamlit 1.32 web application for the Crop Disease Detection System.
#
# Usage:
#   streamlit run app.py
#
# The application allows a user to upload a tomato leaf image and receive:
#   - Predicted disease class
#   - Confidence score (probability of top prediction)
#   - Probability bar chart across all 6 classes
#   - Plain-language management recommendation
#
# Preprocessing in this file mirrors the training pipeline EXACTLY
# (load → resize 224×224 → [0,1] → ImageNet normalise) with NO augmentation,
# matching Section 3.5 and satisfying the parameter consistency requirement.
#
# Chapter 3 references:
#   Section 3.3    — system architecture and output specification
#   Section 3.5.1  — image resizing (224×224)
#   Section 3.5.2  — pixel normalisation (ImageNet mean/std)
#   Section 3.9.4  — Streamlit 1.32 web application framework
#   Section 3.9.3  — Pillow 9.4, Matplotlib 3.7
# =============================================================================

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf

import config

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Model loading  — cached so the model is loaded only once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model() -> tf.keras.Model:
    """
    Load the best MobileNetV2 checkpoint.
    Primary model per Section 3.6.1 — MobileNetV2 selected for deployment
    due to its 3.4M parameter footprint enabling CPU inference in 80–120 ms
    (Section 3.9.5).
    """
    # compile=False: the app only runs inference (model.predict), so the
    # optimiser/loss are unnecessary. It also avoids a cross-Keras-version
    # deserialisation error when loading a model trained on Colab
    # (Keras 2.15/2.17) under the local Keras 2.12 (see evaluate.py).
    model = tf.keras.models.load_model(config.MOBILENET_CKPT, compile=False)
    return model


# ---------------------------------------------------------------------------
# Preprocessing  (Section 3.5 — MUST match the training pipeline exactly)
# ---------------------------------------------------------------------------

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Preprocess a PIL image for model inference.
    This function implements exactly the same steps as preprocessing.py,
    minus data augmentation (Section 3.5.3: augmentation applied to
    training set only).

    Steps:
      1. Resize to 224×224  (Section 3.5.1)
      2. Convert to float32, divide by 255.0 → [0.0, 1.0]  (Section 3.5.2)
      3. Subtract ImageNet channel mean, divide by channel std  (Section 3.5.2)
      4. Add batch dimension → [1, 224, 224, 3]

    Args:
        pil_image: A PIL Image in RGB mode.

    Returns:
        NumPy float32 array of shape [1, 224, 224, 3], ready for model.predict().
    """
    # Step 1: resize to 224×224 (Section 3.5.1).
    # Use tf.image.resize (bilinear) — the SAME interpolation as the training
    # pipeline (preprocessing.py:load_and_resize) — so deployment preprocessing
    # matches training EXACTLY. (PIL's LANCZOS was used here previously, which
    # introduced a small train/serve skew: bilinear vs. LANCZOS differs enough
    # to flip the prediction on borderline, low-confidence images.)
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)                       # [H, W, 3], 0–255
    arr = tf.image.resize(
        arr, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH]
    ).numpy().astype(np.float32)                             # bilinear, still 0–255

    # Step 2: [0, 255] → float32 [0.0, 1.0] (Section 3.5.2)
    arr = arr / 255.0

    # Step 3: ImageNet channel normalisation (Section 3.5.2)
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)  # [0.485, 0.456, 0.406]
    std  = np.array(config.IMAGENET_STD,  dtype=np.float32)  # [0.229, 0.224, 0.225]
    arr  = (arr - mean) / std

    # Step 4: add batch dimension
    return np.expand_dims(arr, axis=0)  # [1, 224, 224, 3]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model: tf.keras.Model, image_array: np.ndarray) -> tuple:
    """
    Run inference and return the predicted class and full probability vector.

    Returns:
        predicted_class_idx:  int — index into config.CLASS_NAMES
        predicted_class_name: str — human-readable disease name
        confidence:           float — probability of predicted class (0.0–1.0)
        probabilities:        np.ndarray — full [6] probability distribution
    """
    probs          = model.predict(image_array, verbose=0)[0]  # [6] probabilities
    pred_idx       = int(np.argmax(probs))
    pred_name      = config.CLASS_NAMES[pred_idx]
    confidence     = float(probs[pred_idx])
    return pred_idx, pred_name, confidence, probs


# ---------------------------------------------------------------------------
# Results visualisation
# ---------------------------------------------------------------------------

def render_probability_chart(probabilities: np.ndarray) -> plt.Figure:
    """
    Horizontal bar chart of predicted probabilities for all 6 classes.
    Used in the Streamlit app alongside the top-1 prediction result.
    Matplotlib 3.7 (Section 3.9.3).
    """
    # Short labels for the chart
    short_names = [
        "Early Blight",
        "Late Blight",
        "Leaf Mold",
        "TYLCV",
        "Bacterial Spot",
        "Healthy",
    ]
    top_idx = int(np.argmax(probabilities))
    colours = ["#1f77b4"] * len(short_names)
    colours[top_idx] = "#2ca02c"   # Highlight the top prediction in green

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(short_names, probabilities * 100, color=colours, height=0.6)

    # Value labels on bars
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center", ha="left", fontsize=9
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Probabilities — All Classes", fontsize=10)
    ax.invert_yaxis()   # Highest probability class at the top
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    # Header
    st.title("🌿 Crop Disease Detection System")
    st.markdown(
        "Upload a photograph of a tomato leaf to receive an automated "
        "disease diagnosis and management recommendations.\n\n"
        "_Kwara State University — Computer Science Final Year Project, 2026_"
    )
    st.divider()

    # Load model (cached)
    try:
        model = load_model()
    except OSError:
        st.error(
            f"**Model checkpoint not found:** `{config.MOBILENET_CKPT}`\n\n"
            "Run `train.py --backbone mobilenetv2` to train the model first."
        )
        return

    # Upload widget
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        label="Select a tomato leaf image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="For best results, ensure the leaf is clearly visible and well-lit.",
    )

    if uploaded_file is None:
        st.info(
            "Awaiting image upload. The system can detect the following conditions:\n\n"
            + "\n".join(f"- {name}" for name in config.CLASS_NAMES)
        )
        return

    # ------------------------------------------------------------------
    # Display uploaded image
    # ------------------------------------------------------------------
    pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.markdown(f"**File:** {uploaded_file.name}")
        st.markdown(f"**Original size:** {pil_image.width} × {pil_image.height} px")
        st.markdown(
            f"**Preprocessed to:** {config.IMAGE_WIDTH} × {config.IMAGE_HEIGHT} px "
            f"(Section 3.5.1)"
        )

    st.divider()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    with st.spinner("Analysing image..."):
        image_array           = preprocess_image(pil_image)
        pred_idx, pred_name, confidence, probs = predict(model, image_array)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    st.subheader("Diagnosis Result")

    # Top-1 prediction and confidence
    is_healthy  = (pred_idx == config.CLASS_NAMES.index("Tomato Healthy"))
    alert_level = "success" if is_healthy else "error"

    if is_healthy:
        st.success(f"**{pred_name}**  —  Confidence: {confidence * 100:.1f}%")
    elif confidence >= 0.80:
        st.error(f"**{pred_name}**  —  Confidence: {confidence * 100:.1f}%")
    elif confidence >= 0.50:
        st.warning(f"**{pred_name}**  —  Confidence: {confidence * 100:.1f}%")
    else:
        st.warning(
            f"**{pred_name}**  —  Confidence: {confidence * 100:.1f}%\n\n"
            "_Low confidence — consider uploading a clearer image or consult "
            "an agricultural extension officer._"
        )

    # Confidence metric
    st.metric(
        label="Top Prediction",
        value=pred_name,
        delta=f"{confidence * 100:.1f}% confidence",
        delta_color="off",
    )

    # Probability chart (all 6 classes)
    st.subheader("Confidence Across All Classes")
    fig = render_probability_chart(probs)
    st.pyplot(fig)
    plt.close(fig)

    # Management recommendations (Section 3.3)
    st.subheader("Management Recommendation")
    recommendation = config.MANAGEMENT.get(pred_name, "")
    if is_healthy:
        st.success(recommendation)
    else:
        st.warning(recommendation)

    # Disclaimer
    st.divider()
    st.caption(
        "⚠️ This diagnosis is advisory only. For severe outbreaks or uncertain results, "
        "consult a certified agricultural extension officer or plant pathologist. "
        "Management recommendations are based on established agricultural literature "
        "and are not precision prescriptions for individual farm conditions."
    )


if __name__ == "__main__":
    main()
