import io

import requests
import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Object Detection with YOLOv8")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to make prediction
    if st.button("Detect Objects"):
        # Send the image to the FastAPI server for prediction
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict/", files=files)

        if response.status_code == 200:
            # Get the processed image from the response
            processed_image = Image.open(io.BytesIO(response.content))
            st.image(
                processed_image,
                caption="Processed Image with Detections",
                use_column_width=True,
            )
        else:
            st.error(f"Error: {response.json()['detail']}")
