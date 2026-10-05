cmake_minimum_required(VERSION 3.20)

project(pxvk_nvcuda_trace LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "pxvk-nvcuda-trace is a Windows-only x86 DLL")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "pxvk-nvcuda-trace must be built with a 32-bit compiler")
endif()

find_path(
    CUDA_DRIVER_INCLUDE_DIR
    NAMES cuda.h
    HINTS
        "$ENV{CUDA_PATH}/include"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include"
    REQUIRED
)

add_library(
    nvcuda
    SHARED
    src/launch_params.c
    src/nvcuda_trace.c
    src/capture.c
)
target_include_directories(
    nvcuda
    PRIVATE
        "${CUDA_DRIVER_INCLUDE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_compile_features(nvcuda PRIVATE c_std_11)
target_compile_definitions(nvcuda PRIVATE _WIN32_WINNT=0x0601)
target_compile_options(nvcuda PRIVATE -Wall -Wextra)
target_link_options(nvcuda PRIVATE -static-libgcc -Wl,--kill-at)
set_target_properties(nvcuda PROPERTIES PREFIX "" OUTPUT_NAME "nvcuda")

add_executable(pxvk-trace src/launcher.c)
target_compile_features(pxvk-trace PRIVATE c_std_11)
target_compile_definitions(pxvk-trace PRIVATE _WIN32_WINNT=0x0601)
target_compile_options(pxvk-trace PRIVATE -Wall -Wextra -Wpedantic)
target_link_options(pxvk-trace PRIVATE -static-libgcc -municode)

include(CTest)
if(BUILD_TESTING)
    add_executable(
        launch_params_test
        src/launch_params.c
        tests/launch_params_test.c
    )
    target_include_directories(launch_params_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(launch_params_test PRIVATE c_std_11)
    target_compile_options(launch_params_test PRIVATE -Wall -Wextra -Wpedantic)
    target_link_options(launch_params_test PRIVATE -static-libgcc)
    add_test(NAME launch_params COMMAND launch_params_test)
    add_executable(proxy_smoke tests/proxy_smoke.c)
    target_compile_features(proxy_smoke PRIVATE c_std_11)
    target_compile_definitions(proxy_smoke PRIVATE _WIN32_WINNT=0x0601)
    target_compile_options(proxy_smoke PRIVATE -Wall -Wextra)
    target_link_options(proxy_smoke PRIVATE -static-libgcc)

    add_executable(injected_smoke tests/injected_smoke.c)
    target_compile_features(injected_smoke PRIVATE c_std_11)
    target_compile_definitions(injected_smoke PRIVATE _WIN32_WINNT=0x0601)
    target_compile_options(injected_smoke PRIVATE -Wall -Wextra -Wpedantic)
    target_link_options(injected_smoke PRIVATE -static-libgcc)
    add_test(
        NAME launcher_smoke
        COMMAND
            pxvk-trace
            --output "${CMAKE_CURRENT_BINARY_DIR}/launcher-smoke-output"
            --
            "$<TARGET_FILE:injected_smoke>"
    )
endif()
