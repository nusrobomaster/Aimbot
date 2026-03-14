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
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    multithread/usb_communication.cpp
    planner/planner.cpp
)
target_include_directories(auto_aim PRIVATE /usr/local/include/onnxruntime)
target_compile_definitions(auto_aim PRIVATE
    ORT_TENSORRT_EXECUTION_PROVIDER
    ORT_CUDA_EXECUTION_PROVIDER
)
target_link_libraries(auto_aim io openvino::runtime tinympcstatic onnxruntime nvinfer nvinfer_plugin nvonnxparser)