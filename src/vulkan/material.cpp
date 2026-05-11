#include "material.h"

void MaterialPushConstants::push(vk::CommandBuffer commandBuffer,
    vk::PipelineLayout pipelineLayout,
    const Material& material)
{
    PushConstantBlock pushConstants{};
    pushConstants.baseColorFactor = material.baseColorFactor;
    pushConstants.metallicFactor = material.metallicFactor;
    pushConstants.roughnessFactor = material.roughnessFactor;
    pushConstants.baseColorTextureSet = static_cast<int>(material.baseColorTextureIndex);
    pushConstants.physicalDescriptorTextureSet = static_cast<int>(material.metallicRoughnessTextureIndex);
    pushConstants.normalTextureSet = static_cast<int>(material.normalTextureIndex);
    pushConstants.occlusionTextureSet = static_cast<int>(material.occlusionTextureIndex);
    pushConstants.emissiveTextureSet = static_cast<int>(material.emissiveTextureIndex);
    pushConstants.alphaMask = 0.0f;
    pushConstants.alphaMaskCutoff = material.alphaCutoff;

    commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstantBlock), &pushConstants);
}