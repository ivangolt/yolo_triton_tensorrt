import argparse
import os
import shutil

from ultralytics import YOLO

# Define the files to be moved and the destination folder
files_to_move = ["config.bptxt", "yolov8s.onnx"]
destination_folder = "1"


def move_model_to_subfolder(files_to_move: list, file_destination: str):
    """Move model to subfolder for convenience

    Args:
        files_to_move (list): files to move
        file_destination (str): end destination for files
    """
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move each file to the destination folder
    for file in files_to_move:
        if os.path.exists(file):
            shutil.move(file, os.path.join(destination_folder, file))
            print(f"Moved {file} to {destination_folder}")
        else:
            print(f"{file} not found!")

    print("File move completed.")


def load_yolo_onnx_model(model_name: str):
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

    return onnx_model


def load_yolo_trt_model(model_name: str):
    """Load Yolo model and transform into TensorRT format

    Args:
        model_name (str): yolo model name (e.g. "yolov8n.pt")
        n - nano model
        s - small model
        m - medium model
        l - large model
    """

    # Return TensorRT model
    model_trt = YOLO(model=model_name)
    model_trt.export(format="engine", device=0)
    tensorrt_model = YOLO(f"{model_name[:7]}.engine")

    return tensorrt_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="load and convert to onnx yolo model")
    parser.add_argument(
        "--model_name", type=str, help="Yolo model name (e.g ''yolov8.pt)"
    )

    args = parser.parse_args()

    onnx_model = load_yolo_onnx_model(model_name=args.model_name)
    # tensorrt_model = load_yolo_trt_model(model_name=args.model_name)

    print(f"Model {args.model_name} has been loaded and converted to onnx and tensorrt")

    move_model_to_subfolder(
        files_to_move=files_to_move, file_destination=destination_folder
    )
