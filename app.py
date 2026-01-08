import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("🧠 YOLO Object Detection App")
st.write("Upload an image and detect objects using YOLO")

# Load model
model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    st.subheader("Detection Result")
    results[0].show()
    st.image(results[0].plot(), use_container_width=True)
