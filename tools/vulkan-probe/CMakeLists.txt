cmake_minimum_required(VERSION 3.22.1)

project(pxvk_vulkan_probe LANGUAGES C CXX)

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

set(SPIRV_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(SPIRV_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    SPIRVHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG 29981f65241605e08b0ede4cfeb999fe3b723c6a
)
FetchContent_MakeAvailable(SPIRVHeaders)

set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 776893306c5d3b22b6185b5d4a258b81d94572bf
)
FetchContent_MakeAvailable(volk)

function(pxvk_configure_vulkan_target target)
    target_link_libraries(${target} PRIVATE volk::volk_headers Vulkan::Headers)
    target_compile_definitions(${target} PRIVATE _WIN32_WINNT=0x0601)
    if(MSVC)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra)
        if(MINGW)
            target_link_options(${target} PRIVATE -static-libgcc)
        endif()
    endif()
endfunction()

add_executable(vulkan-probe main.c "${volk_SOURCE_DIR}/volk.c")
target_compile_features(vulkan-probe PRIVATE c_std_11)
pxvk_configure_vulkan_target(vulkan-probe)

add_executable(vulkan-memory-smoke memory_smoke.cpp "${volk_SOURCE_DIR}/volk.c")
target_compile_features(vulkan-memory-smoke PRIVATE cxx_std_17)
target_link_libraries(vulkan-memory-smoke PRIVATE SPIRV-Headers::SPIRV-Headers)
pxvk_configure_vulkan_target(vulkan-memory-smoke)
if(MINGW)
    target_link_options(vulkan-memory-smoke PRIVATE -static-libstdc++)
endif()
set_property(
    TARGET vulkan-memory-smoke
    APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/oracle.spv")
add_custom_command(
    TARGET vulkan-memory-smoke POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/oracle.spv"
        "$<TARGET_FILE_DIR:vulkan-memory-smoke>/oracle.spv"
)
