SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

SET(Protobuf_USE_STATIC_LIBS ON)
find_package(Protobuf REQUIRED)
  
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
  set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
  set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/onnx)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/../..)


include(${CMAKE_CURRENT_LIST_DIR}/src/onnx/CMakeLists.txt)

add_executable(aceexec ${CMAKE_CURRENT_LIST_DIR}/src/main.cc)

add_library(converter_deps STATIC ${COMMON_SRC} ${CONVERTER_BACKENDS_OBJECTS} ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/flatbuffers/src/util.cpp)

target_link_libraries(converter_deps PUBLIC ${Protobuf_LIBRARIES})
target_link_libraries(aceexec -Wl,--whole-archive converter_deps ${MNN_DEPS} -lpthread -Wl,--no-whole-archive)
target_link_libraries(aceexec -lglog)
