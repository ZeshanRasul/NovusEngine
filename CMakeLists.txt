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
    include/ImGuizmo.h
    include/ImGuizmo.cpp
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
    "src/ECS/entity.h" "src/ECS/component.h" "src/ECS/components/transform_component.h"  "src/ECS/components/camera_component.h" "src/ECS/components/animation_component.h" "src/ECS/event/event.h" "src/ECS/components/physics_component.h" "src/ECS/components/player_controller_component.h" "src/ECS/components/camera_component.cpp" "src/renderer/renderer.h" "src/renderer/shadow_pass.h" "src/renderer/shadow_pass.cpp"
    ${IMGUI_SOURCES}
    ${IMGUI_VULKAN_SOURCES} "src/imgui_vulkan_util.h" "src/input/input_system.h" "src/input/input_system.cpp" "src/vulkan/command_pool.h" 
    "src/vulkan/command_pool.cpp" "src/vulkan/command_buffer.h" "src/vulkan/command_buffer.cpp" "src/vulkan/texture.h" "src/vulkan/texture.cpp" 
    "src/vulkan/buffer.h" "src/vulkan/buffer.cpp" "src/vulkan/utils.h" "src/vulkan/utils.cpp" "src/vulkan/image.h" "src/vulkan/image.cpp" 
    "src/vulkan/image_view.h" "src/vulkan/image_view.cpp" "src/vulkan/pipeline.h" "src/vulkan/pipeline.cpp" "src/model/model.h" "src/model/model.cpp" 
    "src/vulkan/uniform_buffer.h" "src/vulkan/uniform_buffer.cpp" "src/vulkan/shader_storage_buffer.h" "src/vulkan/shader_storage_buffer.cpp" "src/vulkan/descriptors.h" "src/vulkan/descriptors.cpp" "src/vulkan/swap_chain.h" "src/vulkan/swap_chain.cpp" 
    "src/vulkan/depth_target.h" "src/vulkan/depth_target.cpp" "src/vulkan/sync.h" "src/vulkan/sync.cpp" "src/vulkan/material.h" "src/vulkan/material.cpp" "src/model/AssimpAnimChannel.cpp"
    "src/model/AssimpAnimChannel.h" "src/model/AssimpAnimClip.cpp" "src/model/AssimpAnimClip.h" "src/model/AssimpBone.cpp" "src/model/AssimpBone.h" "src/model/AssimpInstance.cpp" 
    "src/model/AssimpInstance.h" "src/model/AssimpMesh.cpp" "src/model/AssimpMesh.h" "src/model/AssimpModel.cpp" "src/model/AssimpModel.h" "src/model/AssimpNode.cpp" "src/model/AssimpNode.h" 
    "src/model/InstanceSettings.h" "src/model/AssimpLoadedModel.cpp" "src/model/AssimpLoadedModel.h" "src/model/ModelAndInstanceData.h" "src/renderer/VkRenderData.h" "src/Logger.cpp" "src/Logger.h" 
    "src/vulkan/IndexBuffer.cpp" "src/vulkan/IndexBuffer.h" "src/vulkan/Texture (2).cpp" "src/vulkan/Texture (2).h" "src/vulkan/VertexBuffer.cpp" "src/vulkan/VertexBuffer.h" "src/vulkan/CommandBuffer.cpp" 
    "src/vulkan/CommandBuffer.h" "lib/ImGuiFileDialog.cpp"
    "src/ECS/entt/scene.h"
    "src/ECS/entt/assimp_instance_component.h"
    "src/ECS/entt/assimp_systems.h"
    "src/gameplay/gameplay_layer.h"
    "src/physics/physics_system.h" "src/physics/physics_system.cpp" "src/ECS/jolt_physics/physics_world.h" "src/ECS/jolt_physics/physics_world.cpp" "include/IconsFontAwesome7.h")

find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(tinyobjloader CONFIG REQUIRED)
find_package(ktx CONFIG REQUIRED)
find_package(Vulkan 1.4 REQUIRED)
find_package(assimp REQUIRED)
find_package(imguizmo CONFIG REQUIRED)
find_package(jolt CONFIG REQUIRED)

find_program(SLANGC_EXECUTABLE slangc REQUIRED)

set(SLANG_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.slang)
set(PBR_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/pbr.slang)
set(IMGUI_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/imgui.slang)
set(SKINNING_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/skinning.slang)
set(SHADOW_SKINNING_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shadow_skinning.slang)
set(TRS_COMPUTE_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/trs_matrix.comp)
set(BONE_COMPUTE_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/bone_matrix.comp)
set(FXAA_SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fxaa.slang)

function(add_slang_shader_target TARGET)
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(SPIRV_OUTPUT_FILES "")

    # Each entry: "source|output|entry1|entry2|..."
    set(SLANG_SHADERS
        "shader.slang|slang.spv|vertMain|fragMain"
        "pbr.slang|pbr.spv|vertMain|fragMain"
        "imgui.slang|imgui.spv|vertMain|fragMain"
        "shadow.slang|shadow.spv|vertMain"
        "skinning.slang|skinning.spv|vertMain|fragMain"
        "shadow_skinning.slang|shadow_skinning.spv|vertMain"
        "fxaa.slang|fxaa.spv|vertMain|fragMain"
        "bloom_extract.slang|bloom_extract.spv|vertMain|fragMain"
        "bloom_blur.slang|bloom_blur.spv|vertMain|fragMain"
        "debug_lines.slang|debug_lines.spv|vertMain|fragMain"
    )

    foreach(SHADER_ENTRY IN LISTS SLANG_SHADERS)
        # Split on | into a proper CMake list
        string(REPLACE "|" ";" SHADER_PARTS "${SHADER_ENTRY}")

        list(GET SHADER_PARTS 0 SHADER_FILE)
        list(GET SHADER_PARTS 1 OUTPUT_FILE)

        # Everything from index 2 onward is an entry point
        list(LENGTH SHADER_PARTS PARTS_LEN)
        set(ENTRY_ARGS "")
        math(EXPR EP_START "2")
        math(EXPR EP_END "${PARTS_LEN} - 1")
        foreach(IDX RANGE ${EP_START} ${EP_END})
            list(GET SHADER_PARTS ${IDX} EP)
            list(APPEND ENTRY_ARGS -entry ${EP})
        endforeach()

        set(SOURCE_PATH ${SHADERS_DIR}/${SHADER_FILE})
        set(OUTPUT_PATH ${SHADERS_DIR}/${OUTPUT_FILE})

        add_custom_command(
            OUTPUT  ${OUTPUT_PATH}
            COMMAND ${SLANGC_EXECUTABLE}
                    ${SOURCE_PATH}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -fvk-use-entrypoint-name
                    ${ENTRY_ARGS}
                    -o ${OUTPUT_PATH}
            DEPENDS ${SOURCE_PATH}
            WORKING_DIRECTORY ${SHADERS_DIR}
            COMMENT "Compiling ${SHADER_FILE} -> ${OUTPUT_FILE}"
            VERBATIM
        )

        list(APPEND SPIRV_OUTPUT_FILES ${OUTPUT_PATH})
    endforeach()

    add_custom_target(${TARGET} DEPENDS ${SPIRV_OUTPUT_FILES})
endfunction()

add_slang_shader_target(shaders)
add_dependencies(NovusEngine shaders)
set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)

file(GLOB GLSL_SOURCE_FILES
   ${SHADERS_DIR}/*.comp
)

if(Vulkan_GLSLC_EXECUTABLE)
  message("Using glslc to compile shaders")
  foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${SHADERS_DIR}/${FILE_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SPIRV} ${GLSL}
      DEPENDS ${GLSL}
      WORKING_DIRECTORY ${SHADERS_DIR}
      VERBATIM
    )
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
  endforeach(GLSL)
elseif (Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
  message("Using glslangValidator to compile shaders")
  foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    set(SPIRV "${SHADERS_DIR}/${FILE_NAME}.spv")
    add_custom_command(
      OUTPUT ${SPIRV}
      COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V -o ${SPIRV} ${GLSL}
      DEPENDS ${GLSL}
      WORKING_DIRECTORY ${SHADERS_DIR}
      VERBATIM
    )

    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
  endforeach(GLSL)
endif()

add_custom_target(compute_shaders DEPENDS ${SPIRV_BINARY_FILES})
add_dependencies(NovusEngine compute_shaders)

target_include_directories(NovusEngine PRIVATE src include build/vcpkg_installed/x64-windows/include build/vcpkg_installed/x64-windows/include/nlohmann build/vcpkg-release/vcpkg_installed/x64-windows/include)
include_directories(${GLFW3_INCLUDE_DIR} ${GLM_INCLUDE_DIRS} ${ASSIMP_INCLUDE_DIR} ${VMA_SOURCE_DIR}/vma)



target_link_libraries(NovusEngine PRIVATE
    glfw
    glm::glm
    tinyobjloader::tinyobjloader
    KTX::ktx        
    Vulkan::Vulkan
    assimp::assimp
    Jolt::Jolt
)

target_compile_definitions(NovusEngine PRIVATE
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS=1
    GLM_ENABLE_EXPERIMENTAL
)


add_subdirectory("NovusEngineProject")
