cmake_minimum_required(VERSION 3.20)

project(pxvk_runtime LANGUAGES C CXX)

if(NOT WIN32)
    message(FATAL_ERROR "pxvk-runtime is the Windows PE32 nvcuda.dll facade")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "pxvk-runtime must be configured with a 32-bit compiler")
endif()

include(FetchContent)
include(CTest)

set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG e3b1eec08173d6b825cd3ac88c885a63b621504a
    GIT_SHALLOW FALSE)
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 776893306c5d3b22b6185b5d4a258b81d94572bf
    GIT_SHALLOW FALSE)
FetchContent_MakeAvailable(VulkanHeaders volk)

add_library(nvcuda SHARED
    runtime.cpp
    vulkan_backend.cpp
    exports.def
    ../nvcuda-trace/src/launch_params.c
    ${volk_SOURCE_DIR}/volk.c)
set_target_properties(nvcuda PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    PREFIX ""
    OUTPUT_NAME "nvcuda")
target_compile_definitions(nvcuda PRIVATE VK_NO_PROTOTYPES NOMINMAX _WIN32_WINNT=0x0601)
target_include_directories(nvcuda PRIVATE ../nvcuda-trace/src)
target_link_libraries(nvcuda PRIVATE volk::volk_headers Vulkan::Headers)

if(MSVC)
    target_compile_options(nvcuda PRIVATE /W4 /EHsc)
else()
    target_compile_options(nvcuda PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-cast-function-type -Wno-missing-field-initializers)
    target_link_options(nvcuda PRIVATE -static-libgcc -static-libstdc++ -Wl,--kill-at)
endif()

add_executable(pxvk-launch WIN32 runtime_launcher.cpp)
set_target_properties(pxvk-launch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
target_link_libraries(pxvk-launch PRIVATE shell32)
if(MSVC)
    target_compile_options(pxvk-launch PRIVATE /W4 /EHsc)
else()
    target_compile_options(pxvk-launch PRIVATE
        -Wall -Wextra -Wpedantic -Wno-cast-function-type)
    target_link_options(pxvk-launch PRIVATE
        -municode -static-libgcc -static-libstdc++)
endif()

if(BUILD_TESTING)
    add_executable(pxvk-runtime-smoke runtime_smoke.cpp)
    set_target_properties(pxvk-runtime-smoke PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)
    target_include_directories(pxvk-runtime-smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    if(MSVC)
        target_compile_options(pxvk-runtime-smoke PRIVATE /W4 /EHsc)
    else()
        target_compile_options(pxvk-runtime-smoke PRIVATE
            -Wall -Wextra -Wpedantic -Wno-cast-function-type)
        target_link_options(pxvk-runtime-smoke PRIVATE -static-libgcc -static-libstdc++)
    endif()
    add_test(NAME pxvk-runtime-smoke
             COMMAND pxvk-runtime-smoke $<TARGET_FILE:nvcuda>)
    set_tests_properties(pxvk-runtime-smoke PROPERTIES
        WORKING_DIRECTORY $<TARGET_FILE_DIR:nvcuda>)
endif()
