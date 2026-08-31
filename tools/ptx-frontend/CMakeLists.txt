cmake_minimum_required(VERSION 3.20)

project(pxvk_ptx_frontend LANGUAGES CXX)

include(FetchContent)
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03
)
FetchContent_MakeAvailable(nlohmann_json)

set(SPIRV_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    SPIRVHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG 29981f65241605e08b0ede4cfeb999fe3b723c6a
)
FetchContent_MakeAvailable(SPIRVHeaders)

add_executable(ptx-frontend main.cpp)
target_compile_features(ptx-frontend PRIVATE cxx_std_17)
target_link_libraries(ptx-frontend PRIVATE nlohmann_json::nlohmann_json)

add_executable(ptx-spirv-emitter spirv_emitter.cpp)
target_compile_features(ptx-spirv-emitter PRIVATE cxx_std_17)
target_link_libraries(
    ptx-spirv-emitter
    PRIVATE nlohmann_json::nlohmann_json SPIRV-Headers::SPIRV-Headers
)

enable_testing()
add_test(NAME ptx-frontend-self-test COMMAND ptx-frontend --self-test)
add_test(
    NAME ptx-spirv-emitter-oracle
    COMMAND ptx-spirv-emitter
        "${CMAKE_CURRENT_SOURCE_DIR}/expected/oracle-manifest.json"
        "${CMAKE_CURRENT_BINARY_DIR}/oracle-explicit-rte.spv"
)
add_test(
    NAME ptx-constant-kernel-emitter
    COMMAND ptx-spirv-emitter
        --entry constant_probe 1 0
        "${CMAKE_CURRENT_SOURCE_DIR}/expected/oracle-manifest.json"
        "${CMAKE_CURRENT_BINARY_DIR}/constant-kernel.spv"
)
add_test(
    NAME ptx-cfg-frontend
    COMMAND ptx-frontend
        "${CMAKE_CURRENT_SOURCE_DIR}/cfg_probe.ptx"
        "${CMAKE_CURRENT_BINARY_DIR}/cfg-probe-manifest.json"
)
add_test(
    NAME ptx-cfg-emitter
    COMMAND ptx-spirv-emitter
        --entry cfg_probe 8 0
        "${CMAKE_CURRENT_BINARY_DIR}/cfg-probe-manifest.json"
        "${CMAKE_CURRENT_BINARY_DIR}/cfg-probe.spv"
)
set_tests_properties(ptx-cfg-emitter PROPERTIES DEPENDS ptx-cfg-frontend)
add_test(
    NAME ptx-barrier-cfg-frontend
    COMMAND ptx-frontend
        "${CMAKE_CURRENT_SOURCE_DIR}/barrier_cfg_probe.ptx"
        "${CMAKE_CURRENT_BINARY_DIR}/barrier-cfg-probe-manifest.json"
)
add_test(
    NAME ptx-barrier-cfg-emitter
    COMMAND ptx-spirv-emitter
        --entry barrier_cfg_probe 8 8
        "${CMAKE_CURRENT_BINARY_DIR}/barrier-cfg-probe-manifest.json"
        "${CMAKE_CURRENT_BINARY_DIR}/barrier-cfg-probe.spv"
)
set_tests_properties(
    ptx-barrier-cfg-emitter PROPERTIES DEPENDS ptx-barrier-cfg-frontend)
if(MSVC)
    foreach(target ptx-frontend ptx-spirv-emitter)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    endforeach()
else()
    foreach(target ptx-frontend ptx-spirv-emitter)
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(MINGW)
            target_link_options(${target} PRIVATE -static-libgcc -static-libstdc++)
        endif()
    endforeach()
endif()
