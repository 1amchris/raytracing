cmake_minimum_required(VERSION 3.21)
project(RayTracingApp LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_VERBOSE_MAKEFILE ON)  # For debugging 

set(ENTRYPOINT src/main.cu)
file(GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp" "src/*.h" "src/*.hpp")
list(FILTER SOURCES EXCLUDE REGEX ${ENTRYPOINT})

add_executable(${PROJECT_NAME} ${ENTRYPOINT} ${SOURCES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 75)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:WL_PLATFORM_WINDOWS;WL_DEBUG>
    $<$<CONFIG:Release>:WL_PLATFORM_WINDOWS;WL_RELEASE>
    $<$<CONFIG:Dist>:WL_PLATFORM_WINDOWS;WL_DIST>
)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>: /W3 /O2>
    $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe="--diag_suppress=20012">
)
# target_compile_options(${PROJECT_NAME} PRIVATE
#     # $<$<CONFIG:Debug>:/std:c++17 /W3 /Od /Zi>
#     # $<$<CONFIG:Release>:/std:c++17 /W3 /O2>
#     # $<$<CONFIG:Dist>:/std:c++17 /W3 /O2>
# )
# # target_link_options(RayTracingApp PRIVATE
# #     $<$<CONFIG:Debug>:/SUBSYSTEM:CONSOLE /DEBUG>
# #     $<$<CONFIG:Release>:/SUBSYSTEM:CONSOLE /DEBUG>
# #     $<$<CONFIG:Dist>:/SUBSYSTEM:WINDOWS>
# # )
