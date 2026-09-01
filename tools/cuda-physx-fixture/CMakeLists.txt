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

add_executable(cuda-deformable-fixture deformable.c)
target_include_directories(cuda-deformable-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-deformable-fixture PRIVATE c_std_11)
target_compile_definitions(cuda-deformable-fixture PRIVATE _WIN32_WINNT=0x0601)

add_executable(cuda-deformable-solver-fixture solver.c)
target_include_directories(
    cuda-deformable-solver-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-deformable-solver-fixture PRIVATE c_std_11)
target_compile_definitions(
    cuda-deformable-solver-fixture PRIVATE _WIN32_WINNT=0x0601)

add_executable(cuda-deformable-collision-fixture collision.c)
target_include_directories(
    cuda-deformable-collision-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-deformable-collision-fixture PRIVATE c_std_11)
target_compile_definitions(
    cuda-deformable-collision-fixture PRIVATE _WIN32_WINNT=0x0601)

add_executable(cuda-fluid-constraints-fixture constraints.c)
target_include_directories(
    cuda-fluid-constraints-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-fluid-constraints-fixture PRIVATE c_std_11)
target_compile_definitions(
    cuda-fluid-constraints-fixture PRIVATE _WIN32_WINNT=0x0601)

add_executable(cuda-fluid-dynamic-response-fixture dynamic_response.c)
target_include_directories(
    cuda-fluid-dynamic-response-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-fluid-dynamic-response-fixture PRIVATE c_std_11)
target_compile_definitions(
    cuda-fluid-dynamic-response-fixture PRIVATE _WIN32_WINNT=0x0601)

add_executable(cuda-deformable-sequence-fixture sequence.c)
target_include_directories(
    cuda-deformable-sequence-fixture PRIVATE "${CUDA_DRIVER_INCLUDE_DIR}")
target_compile_features(cuda-deformable-sequence-fixture PRIVATE c_std_11)
target_compile_definitions(
    cuda-deformable-sequence-fixture PRIVATE _WIN32_WINNT=0x0601)
target_compile_definitions(cuda-physx-fixture PRIVATE _WIN32_WINNT=0x0601)
if(MSVC)
    foreach(target
            cuda-physx-fixture
            cuda-deformable-fixture
            cuda-deformable-solver-fixture
            cuda-deformable-collision-fixture
            cuda-fluid-constraints-fixture
            cuda-fluid-dynamic-response-fixture
            cuda-deformable-sequence-fixture)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4)
    endforeach()
else()
    foreach(target
            cuda-physx-fixture
            cuda-deformable-fixture
            cuda-deformable-solver-fixture
            cuda-deformable-collision-fixture
            cuda-fluid-constraints-fixture
            cuda-fluid-dynamic-response-fixture
            cuda-deformable-sequence-fixture)
        target_compile_options(${target} PRIVATE -Wall -Wextra)
        if(MINGW)
            target_link_options(${target} PRIVATE -static-libgcc)
        endif()
    endforeach()
endif()
