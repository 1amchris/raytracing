cmake_minimum_required(VERSION 3.10)
project(RayTracingApp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(RayTracingApp ${SOURCES})

target_compile_definitions(RayTracingApp PRIVATE
    $<$<CONFIG:Debug>:WL_PLATFORM_WINDOWS;WL_DEBUG>
    $<$<CONFIG:Release>:WL_PLATFORM_WINDOWS;WL_RELEASE>
    $<$<CONFIG:Dist>:WL_PLATFORM_WINDOWS;WL_DIST>
)
target_compile_options(RayTracingApp PRIVATE
    $<$<CONFIG:Debug>:/std:c++17 /W3 /Od /Zi>
    $<$<CONFIG:Release>:/std:c++17 /W3 /O2>
    $<$<CONFIG:Dist>:/std:c++17 /W3 /O2>
)
target_link_options(RayTracingApp PRIVATE
    $<$<CONFIG:Debug>:/SUBSYSTEM:CONSOLE /DEBUG>
    $<$<CONFIG:Release>:/SUBSYSTEM:CONSOLE /DEBUG>
    $<$<CONFIG:Dist>:/SUBSYSTEM:WINDOWS>
)
