import argparse

from ultralytics import YOLO


def load_yolo_model(model_name: str):
    """Load Yolo model and transform into onnx and tensorrt engine

    Args:
        model_name (str): yolo model name (e.g. "yolov8n.pt")
        n - nano model
        s - small model
        m - medium model
        l - large model
    """
    # Return ONNX model
    model = YOLO(model=model_name)
    model.export(format="onnx")
    onnx_model = YOLO(f"{model_name[:7]}.onnx")

    # Return TensorRT model
    model_trt = YOLO(model=model_name)
    model_trt.export(format="engine", device=0)
    tensorrt_model = YOLO(f"{model_name[:7]}.engine")

    # return onnx_model, tensorrt_model
    return onnx_model, tensorrt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and convert to onnx yolo model")
    parser.add_argument(
        "--model_name", type=str, help="Yolo model name (e.g ''yolov8.pt)"
    )

    args = parser.parse_args()

    # onnx_model, tensorrt_model = load_yolo_model(model_name=args.model_name)
    onnx_model, tensorrt_model = load_yolo_model(model_name=args.model_name)

    print(f"Model {args.model_name} has been loaded and converted to onnx and tensorrt")
