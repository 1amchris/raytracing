cmake_minimum_required(VERSION 3.21)
project(RayTracing)

set(APP_NAME RayMarchingApp)

find_package(Vulkan REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_INSTALL OFF CACHE INTERNAL "")

add_subdirectory(vendor/glfw)
add_subdirectory(vendor/imgui)
add_subdirectory(Walnut)
add_subdirectory(${APP_NAME})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${APP_NAME})

target_include_directories(${APP_NAME} PRIVATE
    vendor/glfw/include
    vendor/imgui
    Walnut/src
)

target_link_directories(${APP_NAME} PRIVATE
    vendor/glfw/${CMAKE_BUILD_TYPE}
    vendor/imgui/${CMAKE_BUILD_TYPE}
    Walnut/${CMAKE_BUILD_TYPE}/src
)

target_link_libraries(${APP_NAME} PRIVATE
    Vulkan::Vulkan # also includes glm
    Walnut
    glfw
    dwmapi
    imgui
)

set_target_properties(glfw update_mappings PROPERTIES FOLDER Vendor/GLFW)
set_target_properties(imgui PROPERTIES FOLDER Vendor/ImGui)
set_target_properties(Walnut PROPERTIES FOLDER Core)
