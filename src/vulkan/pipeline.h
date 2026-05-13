#pragma once

#include <string>
#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

namespace Pipeline
{
    struct ShaderStageDesc
    {
        std::string shaderPath;
        vk::ShaderStageFlagBits stage = vk::ShaderStageFlagBits::eVertex;
        const char* entryPoint = "main";
    };

    struct PipelineConfig
    {
        std::vector<ShaderStageDesc> shaderStages;

        std::vector<vk::VertexInputBindingDescription> vertexBindings;
        std::vector<vk::VertexInputAttributeDescription> vertexAttributes;

        std::vector<vk::DescriptorSetLayout> descriptorSetLayouts;
        std::vector<vk::PushConstantRange> pushConstantRanges;

        std::vector<vk::Format> colorAttachmentFormats;
        vk::Format depthAttachmentFormat = vk::Format::eUndefined;
        vk::Format stencilAttachmentFormat = vk::Format::eUndefined;

        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList;
        vk::PolygonMode polygonMode = vk::PolygonMode::eFill;
        vk::CullModeFlags cullMode = vk::CullModeFlagBits::eBack;
        vk::FrontFace frontFace = vk::FrontFace::eCounterClockwise;
        vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1;
        bool depthBiasEnable = false;
        float depthBiasConstantFactor = 0.0f;
        float depthBiasClamp = 0.0f;
        float depthBiasSlopeFactor = 0.0f;

        bool depthTestEnable = true;
        bool depthWriteEnable = true;
        vk::CompareOp depthCompareOp = vk::CompareOp::eLess;

        bool blendEnable = true;

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
    };

    struct PipelineBundle
    {
        vk::raii::PipelineLayout layout = nullptr;
        vk::raii::Pipeline pipeline = nullptr;
    };

    [[nodiscard]] PipelineBundle createPipeline(
        vk::raii::Device const& device,
        PipelineConfig const& config);

    [[nodiscard]] PipelineBundle createComputePipeline(
        vk::raii::Device const& device,
        std::string const& computeShaderPath,
        std::vector<vk::DescriptorSetLayout> const& descriptorSetLayouts = {},
        std::vector<vk::PushConstantRange> const& pushConstantRanges = {},
        const char* entryPoint = "main");

    [[nodiscard]] PipelineBundle createFxaaPipeline(
        vk::raii::Device const& device,
        vk::Format colorFormat,
        vk::raii::DescriptorSetLayout const& descriptorSetLayout);
}