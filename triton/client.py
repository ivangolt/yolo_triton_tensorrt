import argparse
import sys

import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from utils.yolo_classes import YOLO_CLASSES


def get_triton_client(url: str = "localhost:8001"):
    """Function to implement triton grpc client

    Args:
        url (_type_, optional): end to end endpoint for inference server. Defaults to "localhost:8001".

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
    """
    Draws a bounding box on an image with a label indicating the class and confidence.

    Parameters:
    img (numpy.ndarray): The image on which to draw the bounding box.
    class_id (int): The ID of the detected object's class.
    confidence (float): The confidence score of the detection.
    x (int): The x-coordinate of the top left corner of the bounding box.
    y (int): The y-coordinate of the top left corner of the bounding box.
    x_plus_w (int): The x-coordinate of the bottom right corner of the bounding box.
    y_plus_h (int): The y-coordinate of the bottom right corner of the bounding box.

    Returns:
    None: The function modifies the input image in place.
    """
    class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else "Unknown"
    label = f"({class_name}: {confidence:.2f})"
    color = (
        255,
        0,
    )
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def read_image(image_path: str, expected_image_shape) -> np.ndarray:
    """
    Reads an image from the given path, resizes it to the expected shape while maintaining aspect ratio, and prepares it for model input.

    Args:
        image_path (str): The file path to the image that needs to be read and processed.
        expected_image_shape (Tuple[int, int]): A tuple representing the expected dimensions (width, height) of the image as required by the model.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: A tuple containing the following elements:
            - original_image (np.ndarray): The original image read from the file, in its original dimensions.
            - input_image (np.ndarray): The processed image, resized to the expected dimensions, normalized, and prepared for model input.
            - scale (float): The scaling factor applied to the original image to match the expected dimensions.

    Example:
        >>> original_image, input_image, scale = read_image("example.jpg", (640, 640))
        >>> print(f"Original Image Shape: {original_image.shape}")
        >>> print(f"Input Image Shape: {input_image.shape}")
        >>> print(f"Scale Factor: {scale}")
    """
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
    """
    Perform inference on an input image using a specified machine learning model deployed on a Triton Inference Server.

    Args:
        model_name (str): The name of the model deployed on the Triton Inference Server to be used for inference.
        input_image (np.ndarray): A NumPy array representing the input image to be processed by the model.
                                  The array should be pre-processed and formatted according to the model's requirements.
        triton_client (grpcclient.InferenceServerClient): An instance of the Triton Inference Server gRPC client used to
                                                          communicate with the server and perform the inference.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following elements:
            - num_detections (np.ndarray): A NumPy array indicating the number of detections made by the model.
            - detection_boxes (np.ndarray): A NumPy array containing the coordinates of the bounding boxes for each detection.
            - detection_scores (np.ndarray): A NumPy array containing the confidence scores for each detection.
            - detection_classes (np.ndarray): A NumPy array containing the class IDs for each detected object.

    Example:
        >>> input_image = preprocess_image("example.jpg")
        >>> triton_client = grpcclient.InferenceServerClient("localhost:8001")
        >>> num_detections, boxes, scores, classes = run_inference("yolov5", input_image, triton_client)
        >>> print(f"Detected {num_detections} objects.")
        >>> for i in range(num_detections):
        >>>     print(f"Object {i+1}: Class {classes[i]} with score {scores[i]} at {boxes[i]}")
    """
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


def main(image_path, model_name, url):
    triton_client = get_triton_client(url)
    expected_image_shape = (
        triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
    )
    original_image, input_image, scale = read_image(image_path, expected_image_shape)
    num_detections, detection_boxes, detection_scores, detection_classes = (
        run_inference(model_name, input_image, triton_client)
    )

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

    print(f"Detection boxes: {detection_classes}")
    cv2.imwrite("./results/output.jpg", original_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./data/dog.jpg")
    parser.add_argument("--model_name", type=str, default="yolov8_ensemble")
    parser.add_argument("--url", type=str, default="localhost:8001")
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url)
