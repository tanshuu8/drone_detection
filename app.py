import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Title
st.title(" Drone Detection AI")
st.write("Upload an image or video and the AI will detect drones in real-time.")

# Load YOLOv8 pre-trained model (nano version = small + fast)
model = YOLO("yolov8n.pt")

# File uploader
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # If image file
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.read())
        results = model("temp.jpg")
        st.image(results[0].plot(), caption="Detection Result")

    # If video file
    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")
        cap.release()
