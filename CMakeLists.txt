cmake_minimum_required(VERSION 3.0)


include(cmake/color.cmake)
include(cmake/utils.cmake)

# third party path
set(ACE_ROOT ${PROJECT_SOURCE_DIR})
set(ACE_THIRD_PARTY_PATH ${ACE_ROOT}/third-party)
set(ACE_BUILD_THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third-party)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/gflags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/glog.cmake)
# include(${CMAKE_CURRENT_LIST_DIR}/cmake/protobuf.cmake)
# include(${CMAKE_CURRENT_LIST_DIR}/cmake/flatbuffers.cmake)


set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

cmake_policy(SET CMP0048 NEW)
project(MNN VERSION ${VERSION} LANGUAGES C CXX ASM)
# complier options
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_LIST_DIR}/cmake"
)

include(FindPythonInterp REQUIRED)
# build options
option(MNN_USE_SYSTEM_LIB "For opencl and vulkan, use system lib or use dlopen" OFF)
option(MNN_BUILD_HARD "Build -mfloat-abi=hard or not" OFF)
option(MNN_BUILD_SHARED_LIBS "MNN build shared or static lib" ON)
option(MNN_WIN_RUNTIME_MT "MNN use /MT on Windows dll" OFF)
option(MNN_FORBID_MULTI_THREAD "Disable Multi Thread" OFF)
option(MNN_OPENMP "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" OFF)
option(MNN_USE_THREAD_POOL "Use MNN's own thread pool implementation" ON)
option(MNN_BUILD_DEMO "Build demo/exec or not" OFF)
option(MNN_BUILD_TOOLS "Build tools/cpp or not" ON)
option(MNN_BUILD_QUANTOOLS "Build Quantized Tools or not" OFF)
option(MNN_EVALUATION "Build Evaluation Tools or not" OFF)
option(MNN_BUILD_CONVERTER "Build Converter" ON)
option(MNN_DEBUG_MEMORY "MNN Debug Memory Access" OFF)
option(MNN_DEBUG_TENSOR_SIZE "Enable Tensor Size" OFF)
option(MNN_GPU_TRACE "Enable MNN Gpu Debug" OFF)
option(MNN_PORTABLE_BUILD "Link the static version of third party libraries where possible to improve the portability of built executables" OFF)
option(MNN_SEP_BUILD "Build MNN Backends and expression seperately. Only works with MNN_BUILD_SHARED_LIBS=ON" ON)
option(NATIVE_LIBRARY_OUTPUT "Native Library Path" OFF)
option(NATIVE_INCLUDE_OUTPUT "Native Include Path" OFF)
option(MNN_BUILD_MINI "Build MNN-MINI that just supports fixed shape models." OFF)
option(MNN_USE_SSE "Use SSE optimization for x86 if possiable" ON)
option(MNN_SUPPORT_TFLITE_QUAN "Enable MNN's tflite quantized op" ON)

IF (NOT DEFINED MNN_USE_SPARSE_COMPUTE)
  IF (TARGET_OS_IPHONE OR TARGET_OS_SIMULATOR)
    set(MNN_USE_SPARSE_COMPUTE OFF)
  ELSE()
    set(MNN_USE_SPARSE_COMPUTE ON)
  ENDIF()
ENDIF()

IF(NOT MNN_BUILD_SHARED_LIBS AND MNN_SEP_BUILD)
  message(WARNING "Close MNN_SEP_BUILD for static library")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
ENDIF()


include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros.cmake)

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT MNN_BUILD_SHARED_LIBS AND NOT (MSVC OR WIN32))
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  SET(MNN_SEP_BUILD OFF CACHE BOOL "<docstring>" FORCE)
  IF(MNN_BUILD_CONVERTER)
    SET(MNN_PORTABLE_BUILD ON CACHE BOOL "<docstring>" FORCE)
  ENDIF()
ENDIF()

if(MNN_FORBID_MULTI_THREAD)
    add_definitions(-DMNN_FORBIT_MULTI_THREADS)
endif()
if(MNN_SUPPORT_TFLITE_QUAN)
    add_definitions(-DMNN_SUPPORT_TFLITE_QUAN)
endif()

# debug options
if(MNN_DEBUG_MEMORY)
    add_definitions(-DMNN_DEBUG_MEMORY)
endif()
if(MNN_DEBUG_TENSOR_SIZE)
    add_definitions(-DMNN_DEBUG_TENSOR_SIZE)
endif()
if(MNN_GPU_TRACE)
    add_definitions(-DMNN_GPU_FORCE_FINISH)
endif()

# backend options
option(MNN_ONEDNN "Enable oneDNN" OFF)
option(MNN_AVX512 "Enable AVX512" OFF)
option(MNN_CUDA "Enable CUDA" OFF)
option(MNN_TENSORRT "Enable TensorRT" OFF)

if (MNN_USE_THREAD_POOL)
    message(STATUS "Use Threadpool, forbid openmp")
    set(MNN_OPENMP OFF)
    add_definitions(-DMNN_USE_THREAD_POOL)
endif()


# target options
option(MNN_BUILD_BENCHMARK "Build benchmark or not" ON)
option(MNN_BUILD_FOR_ANDROID_COMMAND "Build from command" OFF)
option(MNN_USE_LOGCAT "Use Logcat intead of print for info" ON)
set (MNN_HIDDEN FALSE)
IF(CMAKE_BUILD_TYPE MATCHES Debug)
ELSE()
    set(MNN_HIDDEN TRUE)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)


message(STATUS ">>>>>>>>>>>>>")
message(STATUS "MNN BUILD INFO:")
message(STATUS "\tSystem: ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\toneDNN: ${MNN_ONEDNN}")
message(STATUS "\tTensorRT: ${MNN_TENSORRT}")
message(STATUS "\tCUDA: ${MNN_CUDA}")
message(STATUS "\tOpenMP: ${MNN_OPENMP}")
message(STATUS "\tBF16: ${MNN_SUPPORT_BF16}")
message(STATUS "\tThreadPool: ${MNN_USE_THREAD_POOL}")
message(STATUS "\tHidden: ${MNN_HIDDEN}")
message(STATUS "\tBuild Path: ${CMAKE_CURRENT_BINARY_DIR}")

if(CMAKE_SYSTEM_NAME MATCHES "^Android" OR CMAKE_SYSTEM_NAME MATCHES "^Linux")
    add_definitions(-fPIC)
endif()

option(MNN_USE_CPP11 "Enable MNN use c++11" ON)
if (NOT MSVC)
    if(MNN_USE_CPP11)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STRICT_ANSI__")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      # This is to workaround libgcc.a
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^armv7")
        add_definitions(-mfpu=neon)    #please define in project/cross-compile/arm.toolchain.cmake
    endif()
    if(MNN_BUILD_HARD)
        add_definitions(-mfloat-abi=hard)  #better define in project/cross-compile/arm.toolchain.cmake
    endif()
endif()


IF(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DMNN_DEBUG -DDEBUG)
    if(MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DEBUG")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEBUG")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    endif()
else()
    if (MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    else()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
        if(CMAKE_SYSTEM_NAME MATCHES "^Android")
            if(MNN_BUILD_FOR_ANDROID_COMMAND)
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -s")
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie -fPIE -s")
                set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
            endif()
        endif()
    endif()
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)
if(CMAKE_SYSTEM_NAME MATCHES "^Android")
    IF(MNN_USE_LOGCAT)
        add_definitions(-DMNN_USE_LOGCAT)
    ENDIF()
    IF (NOT MNN_BUILD_FOR_ANDROID_COMMAND)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NATIVE_LIBRARY_OUTPUT}/${ANDROID_ABI})
    ENDIF()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "^Linux")
    if((CMAKE_SYSTEM_PROCESSOR MATCHES "^arm") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64"))
        set(aarch64_linux_include
            #/usr/include/c++/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9
            #/usr/lib/gcc/x86_64-linux-gnu/4.9/include
            #/usr/include/x86_64-linux-gnu/c++/4.9
        )
        include_directories(${aarch64_linux_include})
    endif()
endif()
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/
                    ${CMAKE_CURRENT_LIST_DIR}/ace/
                    ${CMAKE_CURRENT_LIST_DIR}/express/
                    ${CMAKE_CURRENT_LIST_DIR}/tools/
                    ${CMAKE_CURRENT_LIST_DIR}/codegen/
                    ${CMAKE_CURRENT_LIST_DIR}/schema/current/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/flatbuffers/include
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/half
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/imageHelper
                    ${CMAKE_CURRENT_LIST_DIR}/3rd_party/OpenCLHeaders/
                  )


set(MNN_OBJECTS_TO_LINK "")
set(MNN_TARGETS "")

# Core
FILE(GLOB MNN_Core_SRC ${CMAKE_CURRENT_LIST_DIR}/ace/core/*)
add_library(MNNCore OBJECT ${MNN_Core_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCore>)
list(APPEND MNN_TARGETS MNNCore)
if(MNN_BUILD_MINI)
    target_compile_options(MNNCore PRIVATE -DMNN_BUILD_MINI)
endif()

# CV
FILE(GLOB MNN_CV_SRC ${CMAKE_CURRENT_LIST_DIR}/ace/cv/*)
add_library(MNNCV OBJECT ${MNN_CV_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNCV>)
list(APPEND MNN_TARGETS MNNCV)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
    if (APPLE)
        add_definitions(-fno-stack-check) # Workaround a Xcode 11.X bug
    endif()
endif()
if (MNN_USE_SSE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)")
        target_compile_options(MNNCV PRIVATE -DMNN_USE_SSE)
        if (NOT MSVC)
            target_compile_options(MNNCV PRIVATE -msse4.1)
        endif()
    endif()
endif()

# Math
FILE(GLOB MNN_Math_SRC ${CMAKE_CURRENT_LIST_DIR}/ace/math/*)
add_library(MNNMath OBJECT ${MNN_Math_SRC})
list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNMath>)
list(APPEND MNN_TARGETS MNNMath)

# Transform
FILE(GLOB MNN_Transform_SRC ${CMAKE_CURRENT_LIST_DIR}/ace/shape/* ${CMAKE_CURRENT_LIST_DIR}/ace/geometry/*)
add_library(MNNTransform OBJECT ${MNN_Transform_SRC})
IF (NOT MNN_BUILD_MINI)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNNTransform>)
ENDIF()
list(APPEND MNN_TARGETS MNNTransform)

include(${CMAKE_CURRENT_LIST_DIR}/ace/backend/cpu/CMakeLists.txt)


SET(MNN_PUB_HDRS "")
SET(MNN_EXPR_PUB_HDRS "")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/MNNDefine.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/Interpreter.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/HalideRuntime.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/Tensor.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/ErrorCode.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/ImageProcess.hpp")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/Matrix.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/Rect.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/MNNForwardType.h")
list(APPEND MNN_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/AutoTime.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/Expr.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/ExprCreator.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/MathOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/NeuralNetWorkOp.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/Optimizer.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/Executor.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/Module.hpp")
list(APPEND MNN_EXPR_PUB_HDRS "${CMAKE_CURRENT_SOURCE_DIR}/include/ace/expr/NeuralNetWorkOp.hpp")

set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")

if (NOT APPLE)
  if(MNN_OPENMP)
      message(STATUS "[*] Checking OpenMP")
      find_package(OpenMP)
      # For CMake < 3.9, we need to make the target ourselves
      if(NOT TARGET OpenMP::OpenMP_CXX)
          find_package(Threads REQUIRED)
          add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
          set_property(TARGET OpenMP::OpenMP_CXX
              PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
          # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
          set_property(TARGET OpenMP::OpenMP_CXX
              PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
      endif()
      # TODO: Don't pollute global CFLAGS
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
      if (MSVC)
          set(OpenMP_C_FLAGS "/openmp ${OpenMP_C_FLAGS}")
          set(OpenMP_CXX_FLAGS "/openmp ${OpenMP_CXX_FLAGS}")
      endif()
      list(APPEND MNN_EXTRA_DEPENDS OpenMP::OpenMP_CXX)
    endif()
endif()

if ((NOT MSVC) AND MNN_HIDDEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
    # Omit frame pointer may cause difficult debug
    if ((NOT APPLE) AND (NOT WIN32))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
    endif()
endif()
if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math -fno-rtti -fno-exceptions ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstrict-aliasing -ffunction-sections -fdata-sections -ffast-math")
endif()

# Metal
set(MNN_DEPS "")
set(MNN_EXTRA_DEPENDS "")
list(APPEND MNN_DEPS MNN)

# # Plugin
# if(MNN_WITH_PLUGIN)
#     add_definitions(-DMNN_WITH_PLUGIN)
#     include(${CMAKE_CURRENT_LIST_DIR}/ace/plugin/CMakeLists.txt)
# endif()
# oneDNN
IF(MNN_ONEDNN)
    target_compile_definitions(MNNCPU PRIVATE "-DMNN_USE_ONEDNN")
    add_dependencies(MNNCPU oneDNN)
    include(cmake/oneDNN.cmake)
    set(ONEDNN_DIR ${CMAKE_CURRENT_LIST_DIR}/3rd_party/oneDNN)
    add_library(ONEDNN_COMMON OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/ace/common/CMakeFiles/dnnl_common.dir/*.o)
    set_property(TARGET ONEDNN_COMMON PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    add_library(ONEDNN_CPU OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/ace/cpu/CMakeFiles/dnnl_cpu.dir/*.o)
    set_property(TARGET ONEDNN_CPU PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    add_library(ONEDNN_CPU_X64 OBJECT IMPORTED)
    file(GLOB_RECURSE OBJECT_FILES ${ONEDNN_DIR}/ace/cpu/x64/CMakeFiles/dnnl_cpu_x64.dir/*.o)
    set_property(TARGET ONEDNN_CPU_X64 PROPERTY IMPORTED_OBJECTS ${OBJECT_FILES})
    include_directories(${ONEDNN_DIR}/include)
    list(APPEND MNN_TARGETS ${ONEDNN_COMMON})
    list(APPEND MNN_TARGETS ${ONEDNN_CPU})
    list(APPEND MNN_TARGETS ${ONEDNN_CPU_X64})
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_COMMON>)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_CPU>)
    list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:ONEDNN_CPU_X64>)
ENDIF()

# CUDA
IF(MNN_CUDA)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ace/backend/cuda/)
  list(APPEND MNN_TARGETS MNN_CUDA)
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_CUDA>)
  list(APPEND MNN_EXTRA_DEPENDS ${MNN_CUDA_LIBS})
ENDIF()

# Express
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/express/)
IF(MNN_SEP_BUILD)
  list(APPEND MNN_DEPS MNN_Express)
ELSE()
   list(APPEND MNN_TARGETS MNN_Express)
   list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_Express>)
ENDIF()

# TensorRT
IF(MNN_TENSORRT)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/ace/backend/tensorrt/)
  list(APPEND MNN_TARGETS MNN_TRT)
  list(APPEND MNN_OBJECTS_TO_LINK $<TARGET_OBJECTS:MNN_TRT>)
  list(APPEND MNN_EXTRA_DEPENDS ${MNN_TRT_LIBS})
ENDIF()

IF(MNN_SEP_BUILD)
  add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS} -lglog)
ELSE()
  IF(MNN_BUILD_SHARED_LIBS)
    add_library(MNN SHARED ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
    if (WIN32)
      foreach(TARGET ${MNN_TARGETS})
        target_compile_definitions(${TARGET} PRIVATE "-DBUILDING_MNN_DLL")
        target_compile_definitions(${TARGET} INTERFACE "-DUSING_MNN_DLL")
      endforeach()
      target_compile_definitions(MNN PRIVATE "-DBUILDING_MNN_DLL")
      target_compile_definitions(MNN INTERFACE "-DUSING_MNN_DLL")
    endif()
  ELSE()
    add_library(MNN STATIC ${CMAKE_CURRENT_LIST_DIR}/cmake/dummy.cpp ${MNN_OBJECTS_TO_LINK} ${MNN_PUB_HDRS} ${MNN_EXPR_PUB_HDRS})
  ENDIF()
  target_link_libraries(MNN PUBLIC ${MNN_EXTRA_DEPENDS} -lglog)
ENDIF()
if (MNN_ONEDNN)
    add_dependencies(MNN ONEDNN_COMMON ONEDNN_CPU ONEDNN_CPU_X64)
endif()


add_dependencies(MNN MNNCore MNNCV MNNTransform MNNMath MNNCPU)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools/converter)

if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
# Using -pthread, needed by thread-safe implemention of glibc, is better than only using -lpthread
# https://stackoverflow.com/questions/23250863/difference-between-pthread-and-lpthread-while-compiling
  target_link_libraries(MNN PUBLIC -pthread dl)
elseif(CMAKE_SYSTEM_NAME MATCHES "^Android")
  target_link_libraries(MNN PUBLIC log m)
else()
endif()
if (NOT MNN_BUILD_SHARED_LIBS)
    if(APPLE)
        set(MNN_DEPS -Wl,-all_load ${MNN_DEPS} -Wl,-noall_load)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # Static-link will not replace thread-related weak symbol in glibc with strong symbol
        # in pthread library, so we need use --whole-archive to pthread
        # https://stackoverflow.com/questions/35116327/when-g-static-link-pthread-cause-segmentation-fault-why
        if(CMAKE_SYSTEM_NAME MATCHES "^Linux")
            set(MNN_DEPS -Wl,--whole-archive ${MNN_DEPS} -lpthread -Wl,--no-whole-archive)
        else()
            set(MNN_DEPS -Wl,--whole-archive ${MNN_DEPS} -Wl,--no-whole-archive)
        endif()
    endif()
endif()
list(APPEND MNN_TARGETS MNN)
list(REMOVE_ITEM MNN_TARGETS MNN)
IF(MNN_BUILD_TOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/cpp/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_BENCHMARK)
include(${CMAKE_CURRENT_LIST_DIR}/benchmark/CMakeLists.txt)
ENDIF()
IF(MNN_BUILD_QUANTOOLS)
include(${CMAKE_CURRENT_LIST_DIR}/tools/quantization/CMakeLists.txt)
ENDIF()
IF(MNN_EVALUATION)
include(${CMAKE_CURRENT_LIST_DIR}/tools/evaluation/CMakeLists.txt)
ENDIF()

# Install headers
IF(CMAKE_SYSTEM_NAME MATCHES "^Android" AND NOT MNN_BUILD_FOR_ANDROID_COMMAND)
    IF(NOT NATIVE_INCLUDE_OUTPUT)
      set(NATIVE_INCLUDE_OUTPUT ".")
    ENDIF()
    set(MNN_INCLUDE_OUTPUT ${NATIVE_INCLUDE_OUTPUT}/MNN)
    add_custom_command(
      TARGET MNN
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -E make_directory "${MNN_INCLUDE_OUTPUT}/"
    )
    add_custom_command(
      TARGET MNN
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      -E make_directory "${MNN_INCLUDE_OUTPUT}/expr/"
    )
    FOREACH(header ${MNN_PUB_HDRS})
      add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${header} "${MNN_INCLUDE_OUTPUT}/"
      )
    ENDFOREACH()
    FOREACH(header ${MNN_EXPR_PUB_HDRS})
      add_custom_command(
        TARGET MNN
        POST_BUILD
        COMMAND ${CMAKE_COMMAND}
        ARGS -E copy ${header} "${MNN_INCLUDE_OUTPUT}/expr/"
      )
    ENDFOREACH()
ELSEIF(NOT APPLE)
  INSTALL(FILES ${MNN_PUB_HDRS} DESTINATION include/ace/)
  INSTALL(FILES ${MNN_EXPR_PUB_HDRS} DESTINATION include/ace/expr/)
  install(TARGETS MNN
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
  )
ELSE()
  install(TARGETS MNN
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      FRAMEWORK DESTINATION /Library/Frameworks/
  )
  FOREACH(HDR ${MNN_EXPR_PUB_HDRS})
    SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/expr/ )
  ENDFOREACH()
  FOREACH(HDR ${MNN_PUB_HDRS})
    SET_SOURCE_FILES_PROPERTIES(${HDR} PROPERTIES MACOSX_PACKAGE_LOCATION Headers/ )
  ENDFOREACH()
ENDIF()
