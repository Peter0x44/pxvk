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
    set(VULKAN_SDK_ROOT "C:/VulkanSDK/1.4.357.0")
endif()

add_executable(
    vulkan-probe
    main.c
    "${VULKAN_SDK_ROOT}/Include/Volk/volk.c"
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
target_compile_options(vulkan-probe PRIVATE -Wall -Wextra)
target_link_options(vulkan-probe PRIVATE -static-libgcc)
