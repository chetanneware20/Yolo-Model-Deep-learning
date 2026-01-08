import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="YOLO Detection", layout="centered")

st.title("🧠 YOLO Object Detection")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    st.subheader("Detection Result")
    result_img = results[0].plot()
    st.image(result_img, use_container_width=True)
