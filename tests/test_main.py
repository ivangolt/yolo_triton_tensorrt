"""Tests for FastAPI service"""

import cv2
import pytest
from fastapi.testclient import TestClient

from app import main  # noqa: F401
from app.main import app

client = TestClient(app)


@pytest.fixture
def bus_image():
    # Provide the path to your image file
    return "../data/bus.jpg"


def test_root():
    """
    Root endpoint to check the service status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_predict_success(monkeypatch):
    # Mock the Triton client and its methods
    class MockTritonClient:
        def get_model_metadata(self, model_name):
            return type(
                "Metadata",
                (object,),
                {"inputs": [type("Input", (object,), {"shape": [1, 3, 640, 640]})()]},
            )

    def mock_get_triton_client():
        return MockTritonClient()

    def mock_read_image(file_location, expected_image_shape):
        # Return a dummy image, processed image, and scale
        original_image = cv2.imread(file_location)
        input_image = cv2.resize(original_image, expected_image_shape)
        scale = 1.0
        return original_image, input_image, scale

    def mock_run_inference(model_name, input_image, triton_client):
        # Return dummy detection results
        num_detections = 1
        detection_boxes = [(0.1, 0.1, 0.4, 0.4)]
        detection_scores = [0.9]
        detection_classes = [1]
        return num_detections, detection_boxes, detection_scores, detection_classes

    def mock_draw_bounding_box(image, class_id, score, x_min, y_min, x_max, y_max):
        pass  # Do nothing for this mock

    monkeypatch.setattr("app.main.get_triton_client", mock_get_triton_client)
    monkeypatch.setattr("app.main.read_image", mock_read_image)
    monkeypatch.setattr("app.main.run_inference", mock_run_inference)
    monkeypatch.setattr("app.main.draw_bounding_box", mock_draw_bounding_box)


def test_predict(bus_image):
    # Open the image file in binary mode
    with open(bus_image, "rb") as image_file:
        # Make a POST request to the /predict/ endpoint
        response = client.post(
            "/predict/", files={"file": ("../data/bus.jpg", image_file, "image/jpeg")}
        )

        # Check that the request was successful
        assert response.status_code == 200

        # Optionally, save the returned image to verify its correctness
        with open("test_output.jpg", "wb") as output_file:
            output_file.write(response.content)

        # You can add more assertions here to validate the contents of the response,
        # such as verifying that the output image is not empty, etc.
        assert len(response.content) > 0  # Ensure some content is returned
