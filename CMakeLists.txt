cmake_minimum_required(VERSION 3.20)

project(NovusEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_executable(NovusEngine
    src/main.cpp
 )

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)

target_link_libraries(NovusEngine PRIVATE
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    Vulkan::Vulkan
)

target_compile_definitions(NovusEngine PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
)