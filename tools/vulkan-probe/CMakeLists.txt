cmake_minimum_required(VERSION 3.20)

project(pxvk_vulkan_probe LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "vulkan-probe currently targets the Windows x86 loader")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "vulkan-probe must be built with a 32-bit compiler")
endif()

set(VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK root")
if(NOT VULKAN_SDK_ROOT)
    message(FATAL_ERROR "Set VULKAN_SDK or pass -DVULKAN_SDK_ROOT=<SDK root>")
endif()

set(VOLK_SOURCE "${VULKAN_SDK_ROOT}/Include/Volk/volk.c")
if(NOT EXISTS "${VOLK_SOURCE}")
    message(FATAL_ERROR "Volk source not found at ${VOLK_SOURCE}")
endif()

add_executable(
    vulkan-probe
    main.c
    "${VOLK_SOURCE}"
)
target_include_directories(
    vulkan-probe
    PRIVATE
        "${VULKAN_SDK_ROOT}/Include"
        "${VULKAN_SDK_ROOT}/Include/Volk"
)
target_compile_features(vulkan-probe PRIVATE c_std_11)
target_compile_definitions(
    vulkan-probe
    PRIVATE
        _WIN32_WINNT=0x0601
)
if(MSVC)
    target_compile_options(vulkan-probe PRIVATE /W4)
else()
    target_compile_options(vulkan-probe PRIVATE -Wall -Wextra)
    if(MINGW)
        target_link_options(vulkan-probe PRIVATE -static-libgcc)
    endif()
endif()
