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

add_executable(ptx-frontend main.cpp)
target_compile_features(ptx-frontend PRIVATE cxx_std_17)
target_link_libraries(ptx-frontend PRIVATE nlohmann_json::nlohmann_json)

enable_testing()
add_test(NAME ptx-frontend-self-test COMMAND ptx-frontend --self-test)
if(MSVC)
    target_compile_definitions(ptx-frontend PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(ptx-frontend PRIVATE /W4 /permissive-)
else()
    target_compile_options(ptx-frontend PRIVATE -Wall -Wextra -Wpedantic)
    if(MINGW)
        target_link_options(ptx-frontend PRIVATE -static-libgcc -static-libstdc++)
    endif()
endif()
