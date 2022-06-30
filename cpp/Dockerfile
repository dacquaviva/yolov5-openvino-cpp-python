FROM openvino/ubuntu18_runtime:2022.1.0

ENV DEBIAN_FRONTEND=noninteractive
# Install build tools
USER root

RUN cd /opt/intel/openvino/extras/scripts/ && sh download_opencv.sh

WORKDIR /yolov5-openvino