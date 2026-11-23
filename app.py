from io import BytesIO
from PIL import Image
import pandas as pd
import streamlit as st
from visionlab.core import CONDITIONS, degrade, load_detector, detect, summarize

st.set_page_config(page_title="Visual Model Robustness", layout="wide")
st.title("Visual Model Robustness")
st.caption("Test an object detector on identical images after controlled darkness, rain, fog, and blur perturbations")
st.warning("This is a computer vision research demo, not a system for navigation, driving, or safety-critical decisions.")

@st.cache_resource(show_spinner="Loading YOLO model (first use may download weights)...")
def get_model(weight):
    return load_detector(weight)

weights = st.sidebar.text_input("YOLO weights", "yolo11n.pt")
condition = st.sidebar.selectbox("Image condition", CONDITIONS)
severity = st.sidebar.slider("Perturbation strength", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
confidence = st.sidebar.slider("Detection confidence threshold", 0.05, 0.9, 0.25, 0.05)
upload = st.file_uploader("Upload a street image", type=["jpg", "jpeg", "png"])

if upload is not None:
    try:
        source = Image.open(upload).convert("RGB")
        modified = degrade(source, condition, severity)
        cols = st.columns(2)
        with cols[0]: st.image(source, caption="Original image", use_container_width=True)
        with cols[1]: st.image(modified, caption=f"{condition}, severity={severity}", use_container_width=True)
        if st.button("Run object detector", type="primary"):
            model = get_model(weights)
            detections, annotated = detect(model, modified, confidence)
            st.image(annotated, caption="Detection output", use_container_width=True)
            st.json(summarize(detections))
            result = pd.DataFrame([{"class": d.label, "confidence": d.confidence, "bbox": d.xyxy} for d in detections])
            st.dataframe(result, use_container_width=True, hide_index=True)
            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            st.download_button("Download annotated image", buffer.getvalue(), "detections.png", "image/png")
            st.info("Object counts and detection confidence are NOT precision/recall/mAP. Use labeled ground truth for valid accuracy metrics.")
    except Exception as ex:
        st.error(f"Image/model error: {ex}")
else:
    st.info("Upload a JPG or PNG to start. The app does not ship copyrighted street images.")

st.markdown("**For quantitative research:** see `benchmark.py` to evaluate a directory of images with manually verified YOLO bounding-box annotations.")
