cmake_minimum_required(VERSION 3.20)

project(NovusEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_executable(NovusEngine
    src/main.cpp "src/ECS/entity.h" "src/ECS/component.h" "src/ECS/components/transform_component.h" "src/ECS/components/mesh_component.h" "src/ECS/components/camera_component.h" "src/ECS/event/event.h" "src/ECS/components/physics_component.h" "src/ECS/components/camera_component.cpp" "src/renderer/renderer.h")

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(ktx CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)

find_program(SLANGC_EXECUTABLE slangc REQUIRED)

set(SHADER_SLANG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang
)

function (add_slang_shader_target TARGET)
  cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o pbr.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_slang_shader_target( shaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(NovusEngine shaders)

target_link_libraries(NovusEngine PRIVATE
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    KTX::ktx        
    Vulkan::Vulkan
)

target_compile_definitions(NovusEngine PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS=1
)