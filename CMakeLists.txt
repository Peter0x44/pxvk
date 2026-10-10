cmake_minimum_required(VERSION 3.20)
project(pxvk LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    SPIRVHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG 29981f65241605e08b0ede4cfeb999fe3b723c6a)
FetchContent_GetProperties(SPIRVHeaders)
if(NOT spirvheaders_POPULATED)
    FetchContent_Populate(SPIRVHeaders)
endif()

set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG e3b1eec08173d6b825cd3ac88c885a63b621504a)
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 776893306c5d3b22b6185b5d4a258b81d94572bf)
FetchContent_MakeAvailable(VulkanHeaders volk)

if(MSVC)
    add_compile_options(/W4 /EHsc)
else()
    add_compile_options(-Wall -Wextra -Wno-missing-field-initializers -Wno-cast-function-type)
endif()

# PTX -> SPIR-V compiler library (no Windows, CUDA or Vulkan dependencies).
add_library(pxvk-ptx STATIC
    src/ptx/parser.cpp
    src/ptx/cfg.cpp
    src/ptx/compiler.cpp
    src/ptx/fatbin.cpp)
target_include_directories(pxvk-ptx PUBLIC src/ptx ${spirvheaders_SOURCE_DIR}/include)

add_executable(pxvk-ptxc tools/ptxc/main.cpp)
target_link_libraries(pxvk-ptxc PRIVATE pxvk-ptx)
if(NOT MSVC)
    target_link_options(pxvk-ptxc PRIVATE -static)
endif()

# Vulkan compute backend.
add_library(pxvk-vk STATIC src/vk/device.cpp ${volk_SOURCE_DIR}/volk.c)
target_include_directories(pxvk-vk PUBLIC src/vk)
target_compile_definitions(pxvk-vk PUBLIC VK_NO_PROTOTYPES)
target_link_libraries(pxvk-vk PUBLIC volk::volk_headers Vulkan::Headers)

add_executable(pxvk-replay tools/replay/main.cpp)
target_link_libraries(pxvk-replay PRIVATE pxvk-ptx pxvk-vk)
if(NOT MSVC)
    target_link_options(pxvk-replay PRIVATE -static)
endif()

add_executable(pxvk-opcheck tools/opcheck/main.cpp)
target_link_libraries(pxvk-opcheck PRIVATE pxvk-ptx pxvk-vk)
if(NOT MSVC)
    target_link_options(pxvk-opcheck PRIVATE -static)
endif()

add_executable(pxvk-nativeinfo tools/opcheck/nativeinfo.cpp)
target_link_libraries(pxvk-nativeinfo PRIVATE pxvk-ptx)
if(NOT MSVC)
    target_link_options(pxvk-nativeinfo PRIVATE -static)
endif()

# The nvcuda.dll replacement (32-bit only).
if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND WIN32)
    add_library(nvcuda SHARED src/runtime/runtime.cpp src/runtime/loader.cpp src/runtime/stubs.cpp src/runtime/exports.def)
    set_target_properties(nvcuda PROPERTIES PREFIX "" OUTPUT_NAME "nvcuda")
    target_link_libraries(nvcuda PRIVATE pxvk-ptx pxvk-vk)
    target_compile_definitions(nvcuda PRIVATE NOMINMAX _WIN32_WINNT=0x0601)
    if(NOT MSVC)
        target_link_options(nvcuda PRIVATE -static -Wl,--kill-at)
    endif()

    # NVAPI facade so PhysX selects hardware PhysX without an NVIDIA driver.
    add_library(pxvk-nvapi SHARED src/runtime/nvapi.cpp src/runtime/nvapi.def)
    set_target_properties(pxvk-nvapi PROPERTIES PREFIX "")
    if(NOT MSVC)
        target_link_options(pxvk-nvapi PRIVATE -static -Wl,--kill-at)
    endif()

    add_executable(pxvk-launch WIN32 src/runtime/launcher.cpp)
    target_link_libraries(pxvk-launch PRIVATE shell32)
    if(NOT MSVC)
        target_link_options(pxvk-launch PRIVATE -municode -static)
    endif()
endif()

# Benchmark/capture harness for GLUT samples (32-bit).
if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND WIN32)
    add_library(pxvk-harness SHARED tools/harness/harness.cpp)
    set_target_properties(pxvk-harness PROPERTIES PREFIX "")
    target_link_libraries(pxvk-harness PRIVATE ws2_32)
    add_executable(pxvk-run tools/harness/launcher.cpp)
    add_executable(pxvk-stackpeek tools/harness/stackpeek.cpp)
    target_link_libraries(pxvk-stackpeek PRIVATE psapi)
    if(NOT MSVC)
        target_link_options(pxvk-harness PRIVATE -static -Wl,--kill-at)
        target_link_options(pxvk-run PRIVATE -municode -static)
        target_link_options(pxvk-stackpeek PRIVATE -static)
    endif()
endif()
