import cv2
import numpy as np
import tritonclient.http as httpclient

# Load and preprocess the image
image_path = "path_to_image.jpg"
image = cv2.imread(image_path)
image_resized = cv2.resize(image, (416, 416))
input_data = np.transpose(image_resized, (2, 0, 1))  # HWC to CHW
input_data = np.expand_dims(input_data, axis=0).astype(np.float32) / 255.0  # Normalize

# Triton client setup
url = "localhost:8000"
model_name = "yolo_model"

client = httpclient.InferenceServerClient(url=url)
inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
inputs[0].set_data_from_numpy(input_data)

outputs = [httpclient.InferRequestedOutput("output")]

# Send the request to the Triton server
response = client.infer(model_name, inputs, outputs=outputs)

# Get the output data
output_data = response.as_numpy("output")

# Process the YOLO output (this will depend on your specific YOLO implementation)
# For instance, post-processing might include applying confidence thresholds, non-max suppression, etc.
print("YOLO Output:", output_data)
