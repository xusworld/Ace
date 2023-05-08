set(CUDA_MIN_VERSION "9.0")
find_package(CUDA ${CUDA_MIN_VERSION})

set (EXTRA_LIBS "")
if(CUDA_PROFILE)
    set(EXTRA_LIBS  -lnvToolsExt)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()

    if (WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /FS")
    endif ()

    include(${CMAKE_CURRENT_SOURCE_DIR}/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})


    IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "10.0") OR (CUDA_VERSION VERSION_EQUAL "10.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    ENDIF()

    IF ((CUDA_VERSION VERSION_GREATER "11.0") OR (CUDA_VERSION VERSION_EQUAL "11.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
    ENDIF()

    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
                    " archs: ${CUDA_ARCH_FLAGS_readable})")

    else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()


file(GLOB_RECURSE CUDA_SRC ${CMAKE_CURRENT_LIST_DIR}/* ${CMAKE_CURRENT_SOURCE_DIR}/*)
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!! ${CUDA_INCLUDE_DIRS}")

if(WIN32)
else()

    cuda_add_library(Cuda_Main SHARED ${CUDA_SRC})
    if(CUDA_PROFILE)
      target_compile_options(Cuda_Main PRIVATE -DCUDA_PROFILE)
      target_link_libraries(Cuda_Main  ${CUDA_INCLUDE_DIRS}/../lib/libnvToolsExt.so)
    endif()

    set(CUDA_LIBS Cuda_Main PARENT_SCOPE)
    add_library(CUDA OBJECT Register.cpp)
endif()