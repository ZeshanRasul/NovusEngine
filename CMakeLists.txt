cmake_minimum_required(VERSION 3.20)

project(NovusEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# ImGui files
set(IMGUI_SOURCES
    include/imgui.cpp
    include/imgui_draw.cpp
    include/imgui_widgets.cpp
    include/imgui_tables.cpp
    include/imgui_demo.cpp
    include/imgui.h
    include/imconfig.h
    include/imgui_internal.h
)

# Our custom ImGui Vulkan integration
set(IMGUI_VULKAN_SOURCES
    src/imgui_vulkan_util.cpp
)

add_executable(NovusEngine
    src/main.cpp
    src/renderer/renderer.cpp
    "src/ECS/entity.h" "src/ECS/component.h" "src/ECS/components/transform_component.h" "src/ECS/components/mesh_component.h" "src/ECS/components/camera_component.h" "src/ECS/event/event.h" "src/ECS/components/physics_component.h" "src/ECS/components/camera_component.cpp" "src/renderer/renderer.h"
    ${IMGUI_SOURCES}
    ${IMGUI_VULKAN_SOURCES} "src/imgui_vulkan_util.h" "src/input/input_system.h" "src/input/input_system.cpp" "src/vulkan/command_pool.h" "src/vulkan/command_pool.cpp" "src/vulkan/command_buffer.h" "src/vulkan/command_buffer.cpp" "src/vulkan/texture.h" "src/vulkan/texture.cpp" "src/vulkan/buffer.h" "src/vulkan/buffer.cpp" "src/vulkan/utils.h" "src/vulkan/utils.cpp" "src/vulkan/image.h" "src/vulkan/image.cpp" "src/vulkan/image_view.h" "src/vulkan/image_view.cpp" "src/vulkan/pipeline.h" "src/vulkan/pipeline.cpp" "src/model/model.h" "src/model/model.cpp" "src/vulkan/uniform_buffer.h" "src/vulkan/uniform_buffer.cpp" "src/vulkan/descriptors.h" "src/vulkan/descriptors.cpp" "src/vulkan/swap_chain.h" "src/vulkan/swap_chain.cpp" "src/vulkan/depth_target.h" "src/vulkan/depth_target.cpp" "src/vulkan/sync.h" "src/vulkan/sync.cpp" "src/vulkan/material.h" "src/vulkan/material.cpp")

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(ktx CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)

find_program(SLANGC_EXECUTABLE slangc REQUIRED)

set(SLANG_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
set(PBR_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pbr.slang)
set(IMGUI_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/imgui.slang)

function (add_slang_shader_target TARGET)
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv ${SHADERS_DIR}/pbr.spv ${SHADERS_DIR}/imgui.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SLANG_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${PBR_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o pbr.spv
          COMMAND ${SLANGC_EXECUTABLE} ${IMGUI_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o imgui.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SLANG_SHADER_SOURCE} ${PBR_SHADER_SOURCE} ${IMGUI_SHADER_SOURCE}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv ${SHADERS_DIR}/pbr.spv ${SHADERS_DIR}/imgui.spv)
endfunction()

add_slang_shader_target(shaders)
add_dependencies(NovusEngine shaders)

target_include_directories(NovusEngine PRIVATE src)



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

