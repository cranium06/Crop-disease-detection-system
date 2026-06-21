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

SUPPORTED_DISEASES = [
    "Early Blight", "Late Blight", "Leaf Mold",
    "Yellow Leaf Curl Virus (TYLCV)", "Bacterial Spot",
]

SCOPE_NOTICE = (
    "This system is trained **exclusively on tomato leaf images** and can only "
    "identify the following conditions: "
    + ", ".join(SUPPORTED_DISEASES)
    + ", and Healthy. "
    "Uploading images of other crops, objects, or non-leaf content will produce "
    "**unreliable results**."
)

# ---------------------------------------------------------------------------
# Page config + custom styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:wght@400;700&display=swap');

    /* Main body text */
    .stMarkdown p, .stMarkdown li {
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        line-height: 1.7;
    }

    /* Headings */
    h1 {
        font-family: 'Merriweather', Georgia, serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        font-family: 'Merriweather', Georgia, serif !important;
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
        font-weight: 500;
    }

    /* Metric value */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    /* Info/success/warning/error boxes */
    .stAlert p {
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
    }

    /* Caption / disclaimer */
    .stCaption, [data-testid="stCaptionContainer"] p {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }

    /* Subtitle styling */
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        line-height: 1.7;
        margin-bottom: 0.3rem;
        opacity: 0.85;
    }
    .app-byline {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-style: italic;
        opacity: 0.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the detection model...")
def load_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(config.MOBILENET_CKPT, compile=False)
    return model


# ---------------------------------------------------------------------------
# Preprocessing  (must match training pipeline exactly)
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

def predict(model: tf.keras.Model, image_array: np.ndarray) -> tuple:
    probs      = model.predict(image_array, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_name  = config.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_idx, pred_name, confidence, probs


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def render_probability_chart(probabilities: np.ndarray) -> plt.Figure:
    short_names = [
        "Early Blight", "Late Blight", "Leaf Mold",
        "TYLCV", "Bacterial Spot", "Healthy",
    ]
    top_idx = int(np.argmax(probabilities))
    colours = ["#5a8a5a"] * len(short_names)
    colours[top_idx] = "#2e7d32"

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(short_names, probabilities * 100, color=colours, height=0.6)
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", ha="left",
            fontsize=9, fontfamily="sans-serif",
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=10, fontfamily="sans-serif")
    ax.set_title("How confident is the model for each condition?",
                 fontsize=11, fontfamily="sans-serif", fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Human-friendly disease names (drop the "Tomato" prefix for display)
# ---------------------------------------------------------------------------

DISPLAY_NAMES = {
    "Tomato Early Blight":                     "Early Blight",
    "Tomato Late Blight":                      "Late Blight",
    "Tomato Leaf Mold":                        "Leaf Mold",
    "Tomato Yellow Leaf Curl Virus (TYLCV)":   "Yellow Leaf Curl Virus (TYLCV)",
    "Tomato Bacterial Spot":                   "Bacterial Spot",
    "Tomato Healthy":                          "Healthy",
}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Header ──────────────────────────────────────────────────
    st.title("Crop Disease Detection System")
    st.markdown(
        '<p class="app-subtitle">'
        'Using Tomato as a Case Study'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="app-byline">'
        'A final-year research project &mdash; '
        'Department of Computer Science, Kwara State University, 2026'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load model ──────────────────────────────────────────────
    try:
        model = load_model()
    except OSError:
        st.error(
            "The detection model could not be found. "
            "Please make sure the model file is in the **checkpoints/** folder."
        )
        return

    # ── Scope notice ───────────────────────────────────────────
    st.warning(SCOPE_NOTICE)

    # ── Image input ─────────────────────────────────────────────
    st.subheader("Step 1 — Provide a Tomato Leaf Image")
    st.markdown(
        "Take a clear photo of a **single tomato leaf**, or upload one from "
        "your gallery. Make sure the leaf fills most of the frame and is "
        "well-lit for the best results."
    )

    tab_upload, tab_camera = st.tabs(["Upload from gallery", "Use your camera"])
    with tab_upload:
        uploaded_file = st.file_uploader(
            label="Choose a photo of a tomato leaf (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Pick a clear, well-lit image where the leaf is the main subject.",
        )
    with tab_camera:
        camera_photo = st.camera_input(
            "Point your camera at one tomato leaf and tap to capture",
            help="Hold steady, fill the frame with the leaf, and use natural light.",
        )

    image_source = camera_photo if camera_photo is not None else uploaded_file

    if image_source is None:
        st.info(
            "**Welcome!** This tool can help you identify the following "
            "tomato leaf conditions:\n\n"
            + "\n".join(f"- {d}" for d in SUPPORTED_DISEASES)
            + "\n- Healthy (no disease detected)"
            "\n\nUpload or capture a photo above to get started."
        )
        return

    # ── Display the uploaded image ──────────────────────────────
    pil_image = Image.open(io.BytesIO(image_source.getvalue())).convert("RGB")
    source_name = getattr(image_source, "name", None) or "Camera capture"

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(pil_image, caption="Your uploaded image", use_column_width=True)
    with col2:
        st.markdown(f"**File:** {source_name}")
        st.markdown(
            f"**Image size:** {pil_image.width} x {pil_image.height} pixels"
        )
        st.markdown(
            "The image will be resized and normalised automatically "
            "before analysis."
        )

    st.divider()

    # ── Run inference ───────────────────────────────────────────
    with st.spinner("Analysing your image — this takes just a moment..."):
        image_array = preprocess_image(pil_image)
        pred_idx, pred_name, confidence, probs = predict(model, image_array)

    # ── Results ─────────────────────────────────────────────────
    display_name = DISPLAY_NAMES.get(pred_name, pred_name)
    is_healthy = (pred_idx == config.CLASS_NAMES.index("Tomato Healthy"))

    st.subheader("Step 2 — Diagnosis Result")

    if is_healthy and confidence >= 0.50:
        st.success(
            f"**Your tomato leaf looks healthy!**\n\n"
            f"The model is **{confidence * 100:.1f}% confident** that this leaf "
            f"shows no signs of disease. Keep up the good work with your crop care!"
        )
    elif confidence >= 0.80:
        st.error(
            f"**Disease detected: {display_name}**\n\n"
            f"The model is **{confidence * 100:.1f}% confident** in this diagnosis. "
            f"Please review the management advice below and consider taking action soon."
        )
    elif confidence >= 0.50:
        st.warning(
            f"**Possible disease: {display_name}**\n\n"
            f"The model is **{confidence * 100:.1f}% confident** — this is a moderate "
            f"confidence level. The leaf may be showing early symptoms. Review the advice "
            f"below and monitor the plant closely."
        )
    else:
        st.error(
            f"**Uncertain result — please check your image.**\n\n"
            f"The model's confidence is only **{confidence * 100:.1f}%**, which is very "
            f"low. This usually means the image is **not a clear tomato leaf**, or the "
            f"photo is too blurry, dark, or cropped.\n\n"
            f"**Remember:** This system **only detects diseases in tomato leaves**. "
            f"It cannot identify diseases in other crops, and uploading non-tomato images "
            f"will produce unreliable results.\n\n"
            f"**Please try again with:**\n"
            f"- A clear, well-lit photo of a **single tomato leaf**\n"
            f"- The leaf filling most of the frame\n"
            f"- No heavy shadows or overlapping leaves"
        )
        return

    st.metric(
        label="Detected Condition",
        value=display_name,
        delta=f"{confidence * 100:.1f}% confidence",
        delta_color="off",
    )

    # ── Probability chart ───────────────────────────────────────
    st.subheader("Step 3 — Detailed Breakdown")
    st.markdown(
        "The chart below shows how confident the model is for **each possible "
        "condition**. A taller bar means the model thinks that condition is "
        "more likely."
    )
    fig = render_probability_chart(probs)
    st.pyplot(fig)
    plt.close(fig)

    # ── Management advice ───────────────────────────────────────
    st.subheader("Step 4 — What You Can Do")
    recommendation = config.MANAGEMENT.get(pred_name, "")
    if is_healthy:
        st.success(recommendation)
    else:
        st.warning(recommendation)

    # ── Disclaimer ──────────────────────────────────────────────
    st.divider()
    st.caption(
        "This diagnosis is for guidance only and should not replace professional "
        "advice. For severe or widespread symptoms, please consult a certified "
        "agricultural extension officer or plant pathologist. The recommendations "
        "provided are based on established agricultural literature and may need to "
        "be adapted to your specific farm conditions."
    )


if __name__ == "__main__":
    main()
