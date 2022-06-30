## Yolov5-Openvino-Cpp

### Docker installation
This repository folder contains the Dockerfile to build a docker image with the Intel® Distribution of OpenVINO™ toolkit.

1) This command builds an image with OpenVINO™ 2022.1.0 release.
    ```
    docker build . -t openvino_container:2022.1.0
    ```
2) This command creates a docker container with OpenVINO™ 2022.1.0 release.
    ##### windows
    ```
    docker run -it --rm -v %cd%\..:/yolov5-openvino openvino_container:2022.1.0
    ```
    ##### linux
    ```
    docker run -it --rm -v $(pwd)/..:/yolov5-openvino openvino_container:2022.1.0
    ```
### Cmake build

From within the docker run:
```
mkdir build && cd build
```
Then create the make file using cmake:
```
cmake -S ../ -O ./
```
Then compile the program using:
```
make
```
Then run the executable:
```
./main
```
Result:

![IMAGE_DESCRIPTION](../imgs/result.png)