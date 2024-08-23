FROM nvcr.io/nvidia/tritonserver:23.08-py3
COPY /path/to/model/repository /models
COPY /path/to/config.pbtxt /models/simple_model/config.pbtxt

MAINTAINER ivangolt <milovidov.999@gmail.com>