import io

import requests
import streamlit as st
from PIL import Image

# Set the title of the app
st.title("Object Detection with YOLOv8")

# Provide a brief description of the app
st.markdown("""
    This application allows you to upload an image and run object detection using the YOLOv8 model.
    Simply upload an image, and click on the 'Detect Objects' button to see the results.
""")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to make prediction
    if st.button("Detect Objects"):
        with st.spinner("Processing..."):
            try:
                # Send the image to the FastAPI server for prediction
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("http://localhost:8080/predict/", files=files)

                # Check if the response is successful
                if response.status_code == 200:
                    # Get the processed image from the response
                    processed_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        processed_image,
                        caption="Processed Image with Detections",
                        use_column_width=True,
                    )
                else:
                    st.error(
                        f"Error: {response.json().get('detail', 'Unknown error occurred')}"
                    )

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to the prediction server: {e}")

# Display some example images and instructions
else:
    st.info("Please upload an image to get started.")
    st.markdown("""
        - Supported formats: JPG, JPEG, PNG.
        - Make sure the image is clear and well-lit for better detection results.
        - Click the 'Detect Objects' button after uploading your image.
    """)
