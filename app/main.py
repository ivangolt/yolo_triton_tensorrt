import tempfile

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from triton.client import (
    draw_bounding_box,
    get_triton_client,
    read_image,
    run_inference,
)

app = FastAPI()


@app.get("/")
async def root():
    """
    Default root endpoint.

    Returns:
        dict: A simple greeting message.
    """
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_location = temp_file.name
            with open(file_location, "wb") as f:
                f.write(await file.read())

        # Get Triton client
        triton_client = get_triton_client()
        expected_image_shape = (
            triton_client.get_model_metadata("yolov8_ensemble").inputs[0].shape[-2:]
        )

        # Read and preprocess the image
        original_image, input_image, scale = read_image(
            file_location, expected_image_shape
        )

        # Perform inference
        num_detections, detection_boxes, detection_scores, detection_classes = (
            run_inference("yolov8_ensemble", input_image, triton_client)
        )

        # Draw bounding boxes on the image
        for index in range(num_detections):
            box = detection_boxes[index]

            draw_bounding_box(
                original_image,
                detection_classes[index],
                detection_scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )

        # Save the processed and visualized image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
            temp_path = temp.name
            cv2.imwrite(temp_path, original_image)

        return FileResponse(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
