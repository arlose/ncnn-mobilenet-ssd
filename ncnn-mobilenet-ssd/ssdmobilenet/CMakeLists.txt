
find_package(OpenCV REQUIRED core highgui imgproc imgcodecs)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

find_package(Protobuf REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(CAFFE_PROTO_SRCS CAFFE_PROTO_HDRS caffe.proto)

add_executable(ssdmobilenet ssdmobilenet.cpp ${CAFFE_PROTO_SRCS} ${CAFFE_PROTO_HDRS})

target_link_libraries(ssdmobilenet ncnn ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES})
