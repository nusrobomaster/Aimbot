cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
add_subdirectory(planner/tinympc)
add_library(auto_aim OBJECT
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolo_0526_ort.cpp
    yolos/yolo_0526_trt.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)
target_include_directories(auto_aim PRIVATE /usr/local/include/onnxruntime)
target_link_libraries(auto_aim
    io
    openvino::runtime
    tinympcstatic
    onnxruntime
    onnxruntime_providers_shared
    nvinfer
    nvinfer_plugin
    nvonnxparser
)
if(CUDAToolkit_FOUND)
  target_link_libraries(auto_aim CUDA::cudart)
endif()