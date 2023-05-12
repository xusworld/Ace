SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..)

# SET(Protobuf_USE_STATIC_LIBS ON)
# find_package(Protobuf REQUIRED)
  
# if (${CMAKE_VERSION} VERSION_LESS "3.6.0")
#   set(Protobuf_LIBRARIES ${PROTOBUF_LIBRARIES})
#   set(Protobuf_INCLUDE_DIRS ${PROTOBUF_INCLUDE_DIRS})
# endif()

# add_definitions(-DGOOGLE_PROTOBUF_NO_RTTI)

# include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB API_SRC ${CMAKE_CURRENT_LIST_DIR}/api/*.cc)
file(GLOB CORE_SRC ${CMAKE_CURRENT_LIST_DIR}/core/*.cc)
file(GLOB DEVICE_SRC ${CMAKE_CURRENT_LIST_DIR}/device/*.cc)
file(GLOB SHAPE_SRC ${CMAKE_CURRENT_LIST_DIR}/shape/*.cc)
file(GLOB UTILS_SRC ${CMAKE_CURRENT_LIST_DIR}/utils/*.cc)

list(APPEND ALL_SRC ${API_SRC} ${CORE_SRC} ${DEVICE_SRC} ${SHAPE_SRC} ${UTILS_SRC})

add_executable(acecore ${CMAKE_CURRENT_LIST_DIR}/main.cc ${ALL_SRC})

target_link_libraries(acecore -Wl,--whole-archive  -lpthread -Wl,--no-whole-archive)
target_link_libraries(acecore -lglog)
