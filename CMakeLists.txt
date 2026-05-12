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
set( _assimp_HEADER_SEARCH_DIRS
"/usr/include"
"/usr/local/include"
"${CMAKE_SOURCE_DIR}/includes"
"C:/Program Files/assimp/include" )

set( _assimp_LIB_SEARCH_DIRS
"/usr/lib"
"/usr/local/lib"
"${CMAKE_SOURCE_DIR}/lib"
"C:/Program Files/assimp/lib" )

set(assimp_dir  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindAssimp.cmake)

add_executable(NovusEngine
    src/main.cpp
    src/vma_impl.cpp
    src/renderer/renderer.cpp
    "src/ECS/entity.h" "src/ECS/component.h" "src/ECS/components/transform_component.h" "src/ECS/components/mesh_component.h" "src/ECS/components/camera_component.h" "src/ECS/event/event.h" "src/ECS/components/physics_component.h" "src/ECS/components/camera_component.cpp" "src/renderer/renderer.h" "src/renderer/shadow_pass.h" "src/renderer/shadow_pass.cpp"
    ${IMGUI_SOURCES}
    ${IMGUI_VULKAN_SOURCES} "src/imgui_vulkan_util.h" "src/input/input_system.h" "src/input/input_system.cpp" "src/vulkan/command_pool.h" "src/vulkan/command_pool.cpp" "src/vulkan/command_buffer.h" "src/vulkan/command_buffer.cpp" "src/vulkan/texture.h" "src/vulkan/texture.cpp" "src/vulkan/buffer.h" "src/vulkan/buffer.cpp" "src/vulkan/utils.h" "src/vulkan/utils.cpp" "src/vulkan/image.h" "src/vulkan/image.cpp" "src/vulkan/image_view.h" "src/vulkan/image_view.cpp" "src/vulkan/pipeline.h" "src/vulkan/pipeline.cpp" "src/model/model.h" "src/model/model.cpp" "src/vulkan/uniform_buffer.h" "src/vulkan/uniform_buffer.cpp" "src/vulkan/descriptors.h" "src/vulkan/descriptors.cpp" "src/vulkan/swap_chain.h" "src/vulkan/swap_chain.cpp" "src/vulkan/depth_target.h" "src/vulkan/depth_target.cpp" "src/vulkan/sync.h" "src/vulkan/sync.cpp" "src/vulkan/material.h" "src/vulkan/material.cpp" "src/model/AssimpAnimChannel.cpp" "src/model/AssimpAnimChannel.h" "src/model/AssimpAnimClip.cpp" "src/model/AssimpAnimClip.h" "src/model/AssimpBone.cpp" "src/model/AssimpBone.h" "src/model/AssimpInstance.cpp" "src/model/AssimpInstance.h" "src/model/AssimpMesh.cpp" "src/model/AssimpMesh.h" "src/model/AssimpModel.cpp" "src/model/AssimpModel.h" "src/model/AssimpNode.cpp" "src/model/AssimpNode.h" "src/model/InstanceSettings.h" "src/model/AssimpLoadedModel.cpp" "src/model/AssimpLoadedModel.h" "src/model/ModelAndInstanceData.h" "src/renderer/VkRenderData.h" "src/Logger.cpp" "src/Logger.h" "src/vulkan/IndexBuffer.cpp" "src/vulkan/IndexBuffer.h" "src/vulkan/Texture (2).cpp" "src/vulkan/Texture (2).h" "src/vulkan/VertexBuffer.cpp" "src/vulkan/VertexBuffer.h" "src/vulkan/CommandBuffer.cpp" "src/vulkan/CommandBuffer.h")

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(ktx CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)
find_package(assimp REQUIRED)

find_program(SLANGC_EXECUTABLE slangc REQUIRED)

set(SLANG_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
set(PBR_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pbr.slang)
set(IMGUI_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/imgui.slang)
set(SKINNING_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/skinning.slang)

function (add_slang_shader_target TARGET)
  set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
  set (ENTRY_POINTS -entry vertMain -entry fragMain)
  add_custom_command (
          OUTPUT ${SHADERS_DIR}
          COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
  )
  add_custom_command (
          OUTPUT  ${SHADERS_DIR}/slang.spv ${SHADERS_DIR}/pbr.spv ${SHADERS_DIR}/imgui.spv ${SHADERS_DIR}/shadow.spv ${SHADERS_DIR}/skinning.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SLANG_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
          COMMAND ${SLANGC_EXECUTABLE} ${PBR_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o pbr.spv
          COMMAND ${SLANGC_EXECUTABLE} ${IMGUI_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o imgui.spv
          COMMAND ${SLANGC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadow.slang -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -o shadow.spv
          COMMAND ${SLANGC_EXECUTABLE} ${SKINNING_SHADER_SOURCE} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o skinning.spv
          WORKING_DIRECTORY ${SHADERS_DIR}
          DEPENDS ${SHADERS_DIR} ${SLANG_SHADER_SOURCE} ${PBR_SHADER_SOURCE} ${IMGUI_SHADER_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadow.slang ${SKINNING_SHADER_SOURCE}
          COMMENT "Compiling Slang Shaders"
          VERBATIM
  )
  add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv ${SHADERS_DIR}/pbr.spv ${SHADERS_DIR}/imgui.spv ${SHADERS_DIR}/shadow.spv ${SHADERS_DIR}/skinning.spv)
endfunction()

add_slang_shader_target(shaders)
add_dependencies(NovusEngine shaders)

target_include_directories(NovusEngine PRIVATE src include build/vcpkg_installed/x64-windows/include build/vcpkg_installed/x64-windows/include/nlohmann)
include_directories(${GLFW3_INCLUDE_DIR} ${GLM_INCLUDE_DIRS} ${ASSIMP_INCLUDE_DIR} ${VMA_SOURCE_DIR}/vma)



target_link_libraries(NovusEngine PRIVATE
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    KTX::ktx        
    Vulkan::Vulkan
    assimp::assimp
)

target_compile_definitions(NovusEngine PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS=1
    GLM_ENABLE_EXPERIMENTAL
)

