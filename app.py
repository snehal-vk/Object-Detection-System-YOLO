import streamlit as st

st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("Object Detection System using YOLO")
st.write("App loaded successfully ✅")

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import os
import tempfile


st.title("Object Detection System using CNN and YOLO")

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    file_suffix = uploaded_file.name.split(".")[-1]

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix="." + file_suffix)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    st.write("File uploaded successfully!")

    if st.button("Detect Objects"):
        results = model(temp_path, save=True)

        st.success("Detection completed!")

        detection_data = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            object_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection_data.append([
                object_name,
                round(confidence, 2),
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2)
            ])

        df = pd.DataFrame(
            detection_data,
            columns=["Object", "Confidence", "X1", "Y1", "X2", "Y2"]
        )

        st.subheader("Detection Results")
        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Detection CSV",
            csv,
            "detection_results.csv",
            "text/csv"
        )