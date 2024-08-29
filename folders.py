from pathlib import Path

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load an official model

# Export the model to ONNX format
onnx_file = model.export(format="onnx", dynamic=True)
# Define paths
model_name = "yolo"
triton_repo_path = Path("tmp") / "triton_repo"
triton_model_path = triton_repo_path / model_name

# Create directories
(triton_model_path / "1").mkdir(parents=True, exist_ok=True)
Path(onnx_file).rename(triton_model_path / "1" / "model.onnx")
(triton_model_path / "config.pbtxt").touch()
