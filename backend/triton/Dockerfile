FROM nvcr.io/nvidia/tritonserver:23.02-py3

# Install dependencies
RUN pip install opencv-python pillow && \
    apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

VOLUME [ "/models" ]
CMD ["tritonserver", "--model-repository=/models" ]
