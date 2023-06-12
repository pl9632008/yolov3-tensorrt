cmake_minimum_required(VERSION 3.0)
project(test)
aux_source_directory(. src)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
set(CMAKE_BUILD_TYPE Debug)

include_directories(/usr/local/cuda-11.3/include)
link_directories(/usr/local/cuda-11.3/lib64)

include_directories(/home/wjd1/TensorRT-8.0.0.3/include)
link_directories(/home/wjd1/TensorRT-8.0.0.3/lib)

add_executable(testrt ${src})

target_link_libraries(testrt cudart cudnn cublas nvinfer nvparsers nvinfer_plugin nvonnxparser nvrtc )
