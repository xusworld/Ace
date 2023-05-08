message (STATUS "tools/converter/src/onnx/CMakeLists.txt")

protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS
    ${CMAKE_CURRENT_LIST_DIR}/proto/onnx.proto
    ${CMAKE_CURRENT_LIST_DIR}/proto/onnx-operators.proto
)

file(GLOB ONNX_SRC ${CMAKE_CURRENT_LIST_DIR}/*.cc)
file(GLOB ONNX_OPERATOR_SRC ${CMAKE_CURRENT_LIST_DIR}/operator/*.cc)
list(APPEND ONNX_SRC ${ONNX_OPERATOR_SRC})

add_library(onnx_converter OBJECT ${ONNX_SRC} ${ONNX_PROTO_SRCS})


set(CONVERTER_BACKENDS_OBJECTS "")
set(CONVERTER_BACKENDS_TARGETS "")

list(APPEND CONVERTER_BACKENDS_OBJECTS $<TARGET_OBJECTS:onnx_converter>)
list(APPEND CONVERTER_BACKENDS_TARGETS onnx_converter)
