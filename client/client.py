import sys

import cv2
import numpy as np
import tritonclient.grpc as grpcclient


def get_triton_client(url: str = "localhost:8001"):
    """function implement triton inference client

    Args:
        url (_type_, optional): url to endpoint. Defaults to "localhost:8001".

    Returns:
        _type_: triton client
    """
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2,
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=False, keepalive_options=keepalive_options
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"({class_id}: {confidence:.2f})"
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def read_image(image_path: str, expected_image_shape) -> np.ndarray:
    expected_width = expected_image_shape[0]
    expected_height = expected_image_shape[1]
    expected_length = min((expected_height, expected_width))
    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / expected_length

    input_image = cv2.resize(image, (expected_width, expected_height))
    input_image = (input_image / 255.0).astype(np.float32)

    # Channel first
    input_image = input_image.transpose(2, 0, 1)

    # Expand dimensions
    input_image = np.expand_dims(input_image, axis=0)
    return original_image, input_image, scale


def run_inference(
    model_name: str,
    input_image: np.ndarray,
    triton_client: grpcclient.InferenceServerClient,
):
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("images", input_image.shape, "FP32"))
    # Initialize the data
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput("num_detections"))
    outputs.append(grpcclient.InferRequestedOutput("detection_boxes"))
    outputs.append(grpcclient.InferRequestedOutput("detection_scores"))
    outputs.append(grpcclient.InferRequestedOutput("detection_classes"))

    # Test with outputs
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    num_detections = results.as_numpy("num_detections")
    detection_boxes = results.as_numpy("detection_boxes")
    detection_scores = results.as_numpy("detection_scores")
    detection_classes = results.as_numpy("detection_classes")
    return num_detections, detection_boxes, detection_scores, detection_classes
