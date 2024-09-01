# Yolo triton tensorrt
# Overview
This repository  provides an ensemble model that combines a YOLOv8 model exported from the [Ultralytics](https://github.com/ultralytics/ultralytics) repository with NMS (Non-Maximum Suppression) post-processing for deployment on the Triton Inference Server using a TensorRT backend.


For more information about Triton's Ensemble Models, see their documentation on [Architecture.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md) and some of their [preprocessing examples](https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing).

# Directory Structure
```
├── Dockerfile                   # Docker file to build Triton image
├── LICENSE
├── notebooks                    # notebooks for example
├── utils
|   ├── load_model.py            # load model and convert to onnx and tensorrt format and move them to /models repository
|   ├── yolo_classes.py          # yolo classes names
├── data   
├── triton                   # triton model path
|   ├── client.py                # triton client in python
├── models                       
│   ├── postprocess
│   │   ├── 1
│   │   │   ├── model.py
│   │   └── config.pbtxt
│   ├── yolov8_ensemble
│   │   ├── 1
│   │   │   └── model.plan
│   │   └── config.pbtxt
│   └── yolov8_tensorrt
│       ├── 1
│       │   └── model.plan
│       └── config.pbtxt
└── README.md
```

# Quick Start
1. Install [Ultralytics](https://github.com/ultralytics/ultralytics) and TritonClient
```
pip install ultralytics tritonclient[all] 
```
2. (Optional): Update the Score and NMS threshold in [models/postprocess/1/model.py](models/postprocess/1/model.py#L59)
3. (Optional): Update the [models/yolov8_ensemble/config.pbtxt](models/yolov8_ensemble/config.pbtxt) file if your input resolution has changed.
4. Build the Docker Container for Triton Inference:
```
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
```
5. Load onnx Yolo Model 
```
python ./utils/load_model.py --model_name {model_name} (e.g. yolov8m.pt)
```
6. Inside the container of Triton Inference Server, use the `trtexec` tool to convert the YOLOv8 ONNX model to a TensorRT engine file. 
```
/usr/src/tensorrt/bin/trtexec --onnx=/path/to/your/folder/model.onnx --saveEngine=/models/yolov8.engine --fp16 --shapes=images:1x3x640x640
```
   Rename the yolov8.engine file to `model.plan` and place it under the `/models/yolov8_tensorrt/1` directory  and the `/models/yolov8_ensemble/1` directory (see directory structure above).

7. Run Triton Inference Server:
```
DOCKER_NAME="yolov8-triton"
docker run --gpus all \
    -it --rm \
    --name triton
    --net=host \
    -p 8000:8000                            # grpc
    -p 8001:8001                            # http  
    -p 8002:8002
    -v {abs_path_to_your_models}:/models \
    $DOCKER_NAME
```
8. Run the script with `python ./clients/client.py`. The inferred overlay image will be written to `./results/output.jpg`.