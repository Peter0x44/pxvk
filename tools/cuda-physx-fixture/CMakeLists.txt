cmake_minimum_required(VERSION 3.20)

project(pxvk_cuda_physx_fixture LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "cuda-physx-fixture is a Windows-only NVIDIA reference tool")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "cuda-physx-fixture must be built with a 32-bit compiler")
endif()

find_path(
    CUDA_DRIVER_INCLUDE_DIR
    NAMES cuda.h
    HINTS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
    REQUIRED
)

add_executable(cuda-physx-fixture main.c)
target_include_directories(cuda-physx-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-physx-fixture PRIVATE c_std_11)
target_compile_definitions(cuda-physx-fixture PRIVATE _WIN32_WINNT=0x0601)
if(MSVC)
    target_compile_definitions(cuda-physx-fixture PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(cuda-physx-fixture PRIVATE /W4)
else()
    target_compile_options(cuda-physx-fixture PRIVATE -Wall -Wextra)
    if(MINGW)
        target_link_options(cuda-physx-fixture PRIVATE -static-libgcc)
    endif()
endif()
