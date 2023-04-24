SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../)

IF(MNN_PORTABLE_BUILD)
  SET(Protobuf_USE_STATIC_LIBS ON)
ENDIF()

find_package(Protobuf REQUIRED)
  
if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
  set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
  set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
endif()

add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# SET(MNN_CONVERTER_BACKENDS_TARGETS "")
# SET(MNN_CONVERTER_BACKENDS_OBJECTS "")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/src/onnx)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include(${CMAKE_CURRENT_LIST_DIR}/src/compression/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/src/onnx/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/src/optimizer/CMakeLists.txt)

file(GLOB COMMON_SRC ${CMAKE_CURRENT_LIST_DIR}/src/common/*.cpp ${CMAKE_CURRENT_LIST_DIR}/../../ace/core/utils.cc)

add_executable(dump2json ${CMAKE_CURRENT_LIST_DIR}/src/dump2json.cc)
add_executable(aceexec ${CMAKE_CURRENT_LIST_DIR}/src/converter.cc)

IF(MNN_BUILD_SHARED_LIBS)
  add_library(ConvertDeps SHARED ${COMMON_SRC} ${MNN_CONVERTER_BACKENDS_OBJECTS} ${CMAKE_CURRENT_LIST_DIR}/../../ace/core/utils.h ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/flatbuffers/src/util.cpp)
  add_dependencies(ConvertDeps MNN)
ELSE()
  add_library(ConvertDeps STATIC ${COMMON_SRC} ${MNN_CONVERTER_BACKENDS_OBJECTS} ${CMAKE_CURRENT_LIST_DIR}/../../3rd_party/flatbuffers/src/util.cpp)
ENDIF()
  
IF(NOT MNN_BUILD_SHARED_LIBS)
  target_link_libraries(ConvertDeps PUBLIC ${Protobuf_LIBRARIES})
  target_link_libraries(aceexec -Wl,--whole-archive MNNConvertDeps ${MNN_DEPS} -lpthread -Wl,--no-whole-archive)
  target_link_libraries(aceexec ConvertDeps -lglog)
ELSE()
  target_link_libraries(ConvertDeps PUBLIC ${MNN_DEPS} ${Protobuf_LIBRARIES})
  target_link_libraries(aceexec ConvertDeps -lglog)
ENDIF()
