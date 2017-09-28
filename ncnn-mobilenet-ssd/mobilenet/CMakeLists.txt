
find_package(OpenCV REQUIRED core highgui imgproc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

add_executable(mobilenet mobilenet.cpp)

target_link_libraries(mobilenet ncnn ${OpenCV_LIBS})
