#pragma once
#include "../component.h"
#include "../../renderer/renderer_types.h"
#include <vector>
#include <vulkan/vulkan_raii.hpp>

// Holds all GPU resources and render data for a renderable entity.
// This component replaces the old monolithic GameObject structure and is
// consumed directly by the Vulkan renderer.
class RenderableComponent : public Component {
public:
    // Vertex / index data (CPU side kept for reference; GPU side is below)
    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;

    // GPU buffers
    vk::raii::Buffer       vertexBuffer       = { {} };
    vk::raii::DeviceMemory vertexBufferMemory  = nullptr;

    vk::raii::Buffer       indexBuffer        = { {} };
    vk::raii::DeviceMemory indexBufferMemory   = nullptr;

    // Per-frame uniform buffers (one per frame-in-flight)
    std::vector<vk::raii::Buffer>       uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*>                  uniformBuffersMapped;

    // PBR texture slots for a single material slot
    struct PBRTextures {
        vk::raii::Image        baseColorImage            = nullptr;
        vk::raii::DeviceMemory baseColorMemory           = nullptr;
        vk::raii::ImageView    baseColorView             = nullptr;

        vk::raii::Image        metallicRoughnessImage    = nullptr;
        vk::raii::DeviceMemory metallicRoughnessMemory   = nullptr;
        vk::raii::ImageView    metallicRoughnessView     = nullptr;

        vk::raii::Image        normalImage               = nullptr;
        vk::raii::DeviceMemory normalMemory              = nullptr;
        vk::raii::ImageView    normalView                = nullptr;

        vk::raii::Image        occlusionImage            = nullptr;
        vk::raii::DeviceMemory occlusionMemory           = nullptr;
        vk::raii::ImageView    occlusionView             = nullptr;

        vk::raii::Image        emissiveImage             = nullptr;
        vk::raii::DeviceMemory emissiveMemory            = nullptr;
        vk::raii::ImageView    emissiveView              = nullptr;
    };

    // Per-material arrays
    std::vector<Material>                              materials;
    std::vector<PBRTextures>                           materialTextures;
    std::vector<std::vector<vk::raii::DescriptorSet>> materialDescriptorSets;

    // Submesh ranges (one entry per glTF primitive)
    std::vector<Mesh> meshes;

	std::string sourceModelFile;
};
