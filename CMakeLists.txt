# Minimum CMake required
cmake_minimum_required(VERSION 3.18)


# Project
project(ace C CXX)
include(cmake/color.cmake)
include(cmake/utils.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

set(VERSION_MAJOR "0")
set(VERSION_MINOR "1")
set(VERSION_PATCH "0")
set(VERSION "${Green}${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${ColorReset}")

# ace lib name and global directories
set(ace_lib_so "ace")
set(ace_lib_static "ace_static")

# set root dir of modules
set(ACE_ROOT ${PROJECT_SOURCE_DIR})
include_directories(${ACE_ROOT})


# Options
ace_option(USE_OPENMP "Use OpenMP's thread pool implementation." NO)
ace_option(USE_THREAD_POOL "Use custom thread pool implementation" NO)


#compile options for NVIDIA_GPU place
ace_option(USE_CUDA "Use Cuda libs." YES if NVIDIA_GPU)
ace_option(USE_CUBLAS "Use Cublas libs." YES if USE_CUDA)
ace_option(USE_CURAND "Use Curand libs." YES if USE_CUDA)
ace_option(USE_CUFFT "Use CuFFT libs." YES if USE_CUDA)
ace_option(USE_CUDNN "Use Cudnn libs." YES if USE_CUDA)

# build options for cuda.
ace_option(BUILD_CUBIN "BUILD with the -cubin option in Device mode" NO if USE_CUDA)
ace_option(COMPILE_PTX "Returns a list of PTX files generated from src." NO if USE_CUDA)

ace_option(ACE_USE_AVX "Use AVX instructions" NO)
ace_option(ACE_USE_AVX2 "Use AVX2 instructions" NO)
ace_option(ACE_USE_AVX512 "Use AVX512 instructions" NO)


# using 3rd party libs
ace_option(USE_LOGGER "Build native logger components." YES)
ace_option(USE_GLOG "Build Glog components." NO if NOT USE_LOGGER)
ace_option(USE_NANOPB "Use nanopb, a light-weight C implementation of protobuf" YES if USE_SGX)
ace_option(USE_PROTOBUF "Build Google protobuf components." YES if NOT USE_NANOPB)
ace_option(USE_OPENCV "Use static opencv libs." NO)
ace_option(USE_BOOST "Use static BOOST libs." NO)
ace_option(USE_OPENMP "Use Openmp when in android environment." YES if TARGET_ANDROID OR (USE_X86_PLACE AND NOT USE_SGX))
ace_option(USE_GTEST "Use googletest libs." NO if BUILD_WITH_UNIT_TEST)
ace_option(USE_PYTHON "Generate py wrappers." NO)
ace_option(USE_OPENCL "Use OpenCL ." YES if AMD_GPU)
ace_option(USE_GFLAGS "Build Google gflags components." NO)
ace_option(USE_MKL "Use mkl libs." YES if USE_SGX)
ace_option(USE_MKLML "Use MKLML libs." YES if USE_X86_PLACE AND NOT USE_SGX)
ace_option(USE_XBYAK "Use XBYAK libs." YES if USE_X86_PLACE)


set(ACE_CONVERTER_DEPENDENCIES) 

# Third party path
set(ACE_ROOT ${PROJECT_SOURCE_DIR})
set(ACE_THIRD_PARTY_PATH ${ACE_ROOT}/third-party)
set(ACE_BUILD_THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/third-party)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/gflags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/glog.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/protobuf.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/flatbuffers.cmake)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    if (ACE_USE_AVX) 
      message(STATUS "AVX ON")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=c++17 -m64 -mavx ")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17  -m64 -mavx ")
    elseif(ACE_USE_AVX2)
        message(STATUS "AVX2 ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=c++17 -m64 -mavx -msse -msse2 -mavx2 -mfma ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -m64 -mavx -msse -msse2 -mavx2 -mfma ")
    elseif(ACE_USE_AVX512)
        message(STATUS "AVX512 ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -std=c++17 -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq ")
    else()
        message(STATUS "All AVX OFF")
    endif()
else()
    if (ACE_USE_AVX) 
        message(STATUS "AVX ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++17 -m64 -mavx ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17  -m64 -mavx ")
    elseif(ACE_USE_AVX2)
        message(STATUS "AVX2 ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++17 -m64 -mavx -msse -msse2 -mavx2 -mfma ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -m64 -mavx -msse -msse2 -mavx2 -mfma ")
    elseif(ACE_USE_AVX512)
        message(STATUS "AVX512 ON")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -std=c++17 -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -m64 -mavx512f -mfma -mavx512vl -mavx512bw -mavx512dq ")
    else()
        message(STATUS "All AVX OFF")
    endif()
endif(CMAKE_BUILD_TYPE MATCHES Debug)


message(STATUS "CMAKE_C_FLAGS: " ${CMAKE_C_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})

#add_subdirectory(ace)
add_subdirectory(tools)
