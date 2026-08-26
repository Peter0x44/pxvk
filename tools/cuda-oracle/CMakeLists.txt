cmake_minimum_required(VERSION 3.20)

project(pxvk_cuda_oracle LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "cuda-oracle is a Windows-only NVIDIA reference tool")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "cuda-oracle must be built with a 32-bit compiler")
endif()

find_path(
    CUDA_DRIVER_INCLUDE_DIR
    NAMES cuda.h
    HINTS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
    REQUIRED
)

add_executable(cuda-oracle main.c)
target_include_directories(cuda-oracle PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-oracle PRIVATE c_std_11)
target_compile_definitions(cuda-oracle PRIVATE _WIN32_WINNT=0x0601)
target_compile_options(cuda-oracle PRIVATE -Wall -Wextra)
target_link_options(cuda-oracle PRIVATE -static-libgcc)

configure_file(probe.ptx probe.ptx COPYONLY)
