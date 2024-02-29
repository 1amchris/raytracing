cmake_minimum_required(VERSION 3.20)
project(Walnut)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

# Add source files
set(SOURCES
    src/Walnut/Application.cpp
    src/Walnut/ImGui/ImGuiBuild.cpp
    src/Walnut/Image.cpp
    src/Walnut/Input/Input.cpp
    src/Walnut/Random.cpp
)

# Add executable target
add_library(Walnut STATIC ${SOURCES})

# Link libraries
target_link_libraries(Walnut PUBLIC
    Vulkan::Vulkan # also includes glm
    glfw
    dwmapi
    imgui
)

# Include directories
target_include_directories(Walnut PUBLIC
    ../vendor/glfw/include
    ../vendor/imgui
    ../vendor/stb_image
    src
)

# Compiler options
target_compile_definitions(Walnut PRIVATE
    $<$<CONFIG:Debug>:WL_PLATFORM_WINDOWS;WL_DEBUG>
    $<$<CONFIG:Release>:WL_PLATFORM_WINDOWS;WL_RELEASE>
    $<$<CONFIG:Dist>:WL_PLATFORM_WINDOWS;WL_DIST>
)
target_compile_options(Walnut PRIVATE
    $<$<CONFIG:Debug>:/std:c++17 /W3 /Od /Zi>
    $<$<CONFIG:Release>:/std:c++17 /W3 /O2>
    $<$<CONFIG:Dist>:/std:c++17 /W3 /O2>
)

# Linker options
target_link_options(Walnut PRIVATE
    $<$<CONFIG:Debug>:/SUBSYSTEM:WINDOWS>
    $<$<CONFIG:Release>:/SUBSYSTEM:WINDOWS>
    $<$<CONFIG:Dist>:/SUBSYSTEM:WINDOWS>
)
