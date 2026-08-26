cmake_minimum_required(VERSION 3.22.1)

project(pxvk_vulkan_probe LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "vulkan-probe currently targets the Windows x86 loader")
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "vulkan-probe must be built with a 32-bit compiler")
endif()

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG e3b1eec08173d6b825cd3ac88c885a63b621504a
)
FetchContent_MakeAvailable(VulkanHeaders)
set(VULKAN_HEADERS_INSTALL_DIR "${vulkanheaders_SOURCE_DIR}" CACHE PATH "" FORCE)

set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 776893306c5d3b22b6185b5d4a258b81d94572bf
)
FetchContent_MakeAvailable(volk)

add_executable(vulkan-probe main.c "${volk_SOURCE_DIR}/volk.c")
target_link_libraries(vulkan-probe PRIVATE volk::volk_headers Vulkan::Headers)
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
