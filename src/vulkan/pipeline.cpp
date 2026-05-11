#include "pipeline.h"

#include <array>
#include <fstream>
#include <stdexcept>

namespace
{
    std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Failed to open shader file: " + filename);

        std::vector<char> buffer(static_cast<size_t>(file.tellg()));
        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        return buffer;
    }

    vk::raii::ShaderModule createShaderModule(
        vk::raii::Device const& device,
        std::vector<char> const& code)
    {
        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size(),
            .pCode = reinterpret_cast<uint32_t const*>(code.data())
        };

        return vk::raii::ShaderModule(device, createInfo);
    }
}

namespace Pipeline
{
    PipelineBundle createPipeline(vk::raii::Device const& device, PipelineConfig const& config)
    {
        if (config.shaderStages.empty())
            throw std::runtime_error("PipelineConfig requires at least one shader stage.");

        if (config.colorAttachmentFormats.empty() &&
            config.depthAttachmentFormat == vk::Format::eUndefined &&
            config.stencilAttachmentFormat == vk::Format::eUndefined)
        {
            throw std::runtime_error("PipelineConfig requires at least one attachment format.");
        }

        std::vector<vk::raii::ShaderModule> shaderModules;
        shaderModules.reserve(config.shaderStages.size());

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStageInfos;
        shaderStageInfos.reserve(config.shaderStages.size());

        for (auto const& shaderStage : config.shaderStages)
        {
            auto shaderCode = readFile(shaderStage.shaderPath);
            shaderModules.emplace_back(createShaderModule(device, shaderCode));

            shaderStageInfos.emplace_back(vk::PipelineShaderStageCreateInfo{
                .stage = shaderStage.stage,
                .module = *shaderModules.back(),
                .pName = shaderStage.entryPoint
            });
        }

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = static_cast<uint32_t>(config.vertexBindings.size()),
            .pVertexBindingDescriptions = config.vertexBindings.data(),
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(config.vertexAttributes.size()),
            .pVertexAttributeDescriptions = config.vertexAttributes.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = config.topology,
            .primitiveRestartEnable = vk::False
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode = config.polygonMode,
            .cullMode = config.cullMode,
            .frontFace = config.frontFace,
            .depthBiasEnable = config.depthBiasEnable,
            .depthBiasConstantFactor = config.depthBiasConstantFactor,
            .depthBiasClamp = config.depthBiasClamp,
            .depthBiasSlopeFactor = config.depthBiasSlopeFactor,
            .lineWidth = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = config.samples,
            .sampleShadingEnable = vk::False
        };

        vk::PipelineDepthStencilStateCreateInfo depthStencil{
            .depthTestEnable = config.depthTestEnable,
            .depthWriteEnable = config.depthWriteEnable,
            .depthCompareOp = config.depthCompareOp,
            .depthBoundsTestEnable = vk::False,
            .stencilTestEnable = vk::False
        };

        std::vector<vk::PipelineColorBlendAttachmentState> colorBlendAttachments;
        colorBlendAttachments.reserve(config.colorAttachmentFormats.size());

        for (size_t i = 0; i < config.colorAttachmentFormats.size(); ++i)
        {
            colorBlendAttachments.emplace_back(vk::PipelineColorBlendAttachmentState{
                .blendEnable = config.blendEnable,
                .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                .colorBlendOp = vk::BlendOp::eAdd,
                .srcAlphaBlendFactor = vk::BlendFactor::eOne,
                .dstAlphaBlendFactor = vk::BlendFactor::eZero,
                .alphaBlendOp = vk::BlendOp::eAdd,
                .colorWriteMask =
                    vk::ColorComponentFlagBits::eR |
                    vk::ColorComponentFlagBits::eG |
                    vk::ColorComponentFlagBits::eB |
                    vk::ColorComponentFlagBits::eA
            });
        }

        vk::PipelineColorBlendStateCreateInfo colorBlending{
            .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = static_cast<uint32_t>(colorBlendAttachments.size()),
            .pAttachments = colorBlendAttachments.empty() ? nullptr : colorBlendAttachments.data()
        };

        vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(config.dynamicStates.size()),
            .pDynamicStates = config.dynamicStates.data()
        };

        vk::PipelineLayoutCreateInfo layoutInfo{
            .setLayoutCount = static_cast<uint32_t>(config.descriptorSetLayouts.size()),
            .pSetLayouts = config.descriptorSetLayouts.data(),
            .pushConstantRangeCount = static_cast<uint32_t>(config.pushConstantRanges.size()),
            .pPushConstantRanges = config.pushConstantRanges.data()
        };

        PipelineBundle result;
        result.layout = vk::raii::PipelineLayout(device, layoutInfo);

        vk::PipelineRenderingCreateInfo renderingInfo{
            .colorAttachmentCount = static_cast<uint32_t>(config.colorAttachmentFormats.size()),
            .pColorAttachmentFormats = config.colorAttachmentFormats.empty() ? nullptr : config.colorAttachmentFormats.data(),
            .depthAttachmentFormat = config.depthAttachmentFormat,
            .stencilAttachmentFormat = config.stencilAttachmentFormat
        };

        vk::GraphicsPipelineCreateInfo pipelineInfo{
            .pNext = &renderingInfo,
            .stageCount = static_cast<uint32_t>(shaderStageInfos.size()),
            .pStages = shaderStageInfos.data(),
            .pVertexInputState = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = *result.layout,
            .renderPass = nullptr,
            .subpass = 0
        };

        result.pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
        return result;
    }
}