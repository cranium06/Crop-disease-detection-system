# =============================================================================
# app.py  —  Streamlit 1.32 web application
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

SUPPORTED_CONDITIONS = [
    "Early Blight", "Late Blight", "Leaf Mold",
    "Yellow Leaf Curl Virus (TYLCV)", "Bacterial Spot", "Healthy",
]

# ---------------------------------------------------------------------------
# Page config + typography
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* System font stack — loads instantly, looks native on every OS */
    :root {
        --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
        --font-serif: Georgia, "Times New Roman", Times, serif;
    }

    /* Body text: 15px, relaxed line height */
    .stMarkdown p, .stMarkdown li {
        font-family: var(--font-sans) !important;
        font-size: 15px !important;
        line-height: 1.72 !important;
    }

    /* Page title */
    h1 {
        font-family: var(--font-serif) !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    /* Section headings */
    h2, h3, [data-testid="stSubheader"] {
        font-family: var(--font-sans) !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-sans) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    /* Metric */
    [data-testid="stMetricValue"] {
        font-family: var(--font-sans) !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: var(--font-sans) !important;
        font-size: 0.9rem !important;
    }

    /* Alert boxes */
    .stAlert p, .stAlert li {
        font-family: var(--font-sans) !important;
        font-size: 14.5px !important;
        line-height: 1.65 !important;
    }

    /* Caption */
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-family: var(--font-sans) !important;
        font-size: 12.5px !important;
    }

    /* Custom subtitle */
    .subtitle { font-size: 16px; opacity: 0.8; margin-bottom: 2px; }
    .byline   { font-size: 13px; opacity: 0.55; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model...")
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(config.MOBILENET_CKPT, compile=False)


# ---------------------------------------------------------------------------
# Preprocessing (matches training pipeline exactly)
# ---------------------------------------------------------------------------

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    rgb = pil_image.convert("RGB")
    arr = np.array(rgb, dtype=np.uint8)
    arr = tf.image.resize(
        arr, [config.IMAGE_HEIGHT, config.IMAGE_WIDTH]
    ).numpy().astype(np.float32)
    arr = arr / 255.0
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(config.IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, image_array):
    probs      = model.predict(image_array, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_name  = config.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_idx, pred_name, confidence, probs


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "Tomato Early Blight":                   "Early Blight",
    "Tomato Late Blight":                    "Late Blight",
    "Tomato Leaf Mold":                      "Leaf Mold",
    "Tomato Yellow Leaf Curl Virus (TYLCV)": "Yellow Leaf Curl Virus (TYLCV)",
    "Tomato Bacterial Spot":                 "Bacterial Spot",
    "Tomato Healthy":                        "Healthy",
}

SHORT_LABELS = [
    "Early Blight", "Late Blight", "Leaf Mold",
    "TYLCV", "Bacterial Spot", "Healthy",
]

def render_chart(probs):
    top = int(np.argmax(probs))
    colors = ["#78909C"] * 6
    colors[top] = "#2E7D32"

    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    bars = ax.barh(SHORT_LABELS, probs * 100, color=colors, height=0.55)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", ha="left", fontsize=8.5)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title("Prediction confidence across all classes",
                 fontsize=10, fontweight="600", pad=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    # ── Header ──────────────────────────────────────────────────
    st.title("Crop Disease Detection System")
    st.markdown('<p class="subtitle">Using Tomato as a Case Study</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="byline">'
        'Department of Computer Science, Kwara State University, Malete &mdash; 2026'
        '</p>', unsafe_allow_html=True)
    st.divider()

    # ── Load model ──────────────────────────────────────────────
    try:
        model = load_model()
    except OSError:
        st.error("Model checkpoint not found. Please ensure "
                 "**checkpoints/mobilenetv2_best.h5** exists.")
        return

    # ── Scope notice ────────────────────────────────────────────
    st.warning(
        "This system is designed **only for tomato leaves**. "
        "It can detect: " + ", ".join(SUPPORTED_CONDITIONS) + ". "
        "Uploading images of other plants or objects may produce "
        "inaccurate results."
    )

    # ── Upload / Camera ─────────────────────────────────────────
    st.subheader("Upload or capture a leaf image")
    st.markdown(
        "Provide a clear, well-lit photo of a single tomato leaf. "
        "The leaf should fill most of the frame for accurate results."
    )

    tab_upload, tab_camera = st.tabs(["Upload an image", "Take a photo"])
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Select a tomato leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="Use a clear photo where the leaf is the main subject.",
        )
    with tab_camera:
        camera_photo = st.camera_input(
            "Point your camera at a tomato leaf and capture",
            help="Fill the frame with one leaf under good lighting.",
        )

    image_source = camera_photo if camera_photo is not None else uploaded_file

    if image_source is None:
        st.info(
            "Upload a photo or use the camera to begin. "
            "The system will analyse the image and identify any of the "
            "following conditions:\n\n"
            + "\n".join(f"- {c}" for c in SUPPORTED_CONDITIONS)
        )
        return

    # ── Show the image ──────────────────────────────────────────
    pil_image = Image.open(io.BytesIO(image_source.getvalue())).convert("RGB")
    source_name = getattr(image_source, "name", None) or "Camera capture"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(pil_image, caption="Uploaded image", use_column_width=True)
    with col2:
        st.markdown(f"**File:** {source_name}")
        st.markdown(f"**Size:** {pil_image.width} x {pil_image.height} px")
        st.markdown(
            "The image will be resized to 224 x 224 pixels and normalised "
            "before being passed to the model."
        )

    st.divider()

    # ── Inference ───────────────────────────────────────────────
    with st.spinner("Analysing..."):
        image_array = preprocess_image(pil_image)
        pred_idx, pred_name, confidence, probs = predict(model, image_array)

    display_name = DISPLAY_NAMES.get(pred_name, pred_name)
    is_healthy = pred_idx == config.CLASS_NAMES.index("Tomato Healthy")

    # ── Diagnosis ───────────────────────────────────────────────
    st.subheader("Diagnosis")

    if confidence < 0.50:
        st.error(
            f"**Low confidence ({confidence*100:.1f}%) — unable to make a "
            f"reliable diagnosis.**\n\n"
            f"This may happen when the uploaded image is not a clear tomato "
            f"leaf, or when the photo is blurry, poorly lit, or shows "
            f"multiple overlapping leaves.\n\n"
            f"This system only recognises diseases in **tomato leaves**. "
            f"Please upload a clear image of a single tomato leaf and try again."
        )
        return

    if is_healthy:
        st.success(
            f"**Healthy** — no disease detected.\n\n"
            f"Confidence: **{confidence*100:.1f}%**. The leaf appears to be "
            f"in good condition. Continue with regular monitoring and "
            f"good cultural practices."
        )
    elif confidence >= 0.80:
        st.error(
            f"**{display_name}** detected.\n\n"
            f"Confidence: **{confidence*100:.1f}%**. "
            f"Review the management recommendation below."
        )
    else:
        st.warning(
            f"**{display_name}** — possible detection.\n\n"
            f"Confidence: **{confidence*100:.1f}%**. The model is moderately "
            f"confident. The leaf may be showing early symptoms. Monitor "
            f"the plant closely and review the recommendation below."
        )

    st.metric("Prediction", display_name,
              delta=f"{confidence*100:.1f}% confidence", delta_color="off")

    # ── Probability breakdown ───────────────────────────────────
    st.subheader("Confidence breakdown")
    fig = render_chart(probs)
    st.pyplot(fig)
    plt.close(fig)

    # ── Recommendation ──────────────────────────────────────────
    st.subheader("Management recommendation")
    recommendation = config.MANAGEMENT.get(pred_name, "")
    if is_healthy:
        st.success(recommendation)
    else:
        st.warning(recommendation)

    # ── Disclaimer ──────────────────────────────────────────────
    st.divider()
    st.caption(
        "This tool provides advisory diagnoses only and is not a substitute "
        "for professional assessment. For severe or uncertain cases, consult "
        "a certified agricultural extension officer or plant pathologist."
    )


if __name__ == "__main__":
    main()
