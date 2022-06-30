add_executable(main main.cpp)

target_link_libraries(main
/opt/intel/openvino/extras/opencv/lib/libopencv_dnn.so
/opt/intel/openvino/extras/opencv/lib/libopencv_core.so
/opt/intel/openvino/extras/opencv/lib/libopencv_imgcodecs.so
/opt/intel/openvino/extras/opencv/lib/libopencv_imgproc.so
/opt/intel/openvino/runtime/lib/intel64/libopenvino.so

)

target_include_directories(main PUBLIC 
/opt/intel/openvino/extras/opencv/include/
/opt/intel/openvino/runtime/include/
/opt/intel/openvino/runtime/include/ie/

)