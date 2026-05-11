#pragma once

#include "../renderer/renderer_types.h"

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

struct PushConstantBlock
{
    glm::vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    int baseColorTextureSet;
    int physicalDescriptorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    float alphaMask;
    float alphaMaskCutoff;
};

class MaterialPushConstants
{
public:
    static void push(vk::CommandBuffer commandBuffer,
        vk::PipelineLayout pipelineLayout,
        const Material& material);
};