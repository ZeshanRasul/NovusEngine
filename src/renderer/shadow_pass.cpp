#include "shadow_pass.h"

#include "../vulkan/depth_target.h"
#include "../vulkan/image.h"
#include "../vulkan/image_view.h"
#include "../vulkan/material.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/descriptors.h"
#include "renderer_types.h"

void ShadowPass::createResources(vk::raii::Device& device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::Image& shadowImage,
    vk::raii::DeviceMemory& shadowImageMemory,
    vk::raii::ImageView& shadowImageView,
    vk::raii::Sampler& shadowSampler)
{
    const auto depthFormat = DepthTarget::findDepthFormat(physicalDevice);

    Image::createImage(device, physicalDevice, ShadowMapWidth, ShadowMapHeight, depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        shadowImage,
        shadowImageMemory);

    shadowImageView = ImageView::createImageView(device, shadowImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

  vk::SamplerCreateInfo samplerInfo{};
  samplerInfo.magFilter = vk::Filter::eNearest;
  samplerInfo.minFilter = vk::Filter::eNearest;
  samplerInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;
  samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
  samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
  samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
  samplerInfo.mipLodBias = 0.0f;
  samplerInfo.anisotropyEnable = vk::False;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.compareEnable = vk::False;
  samplerInfo.compareOp = vk::CompareOp::eNever;
  samplerInfo.minLod = 0.0f;
  samplerInfo.maxLod = 1.0f;
  samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
  samplerInfo.unnormalizedCoordinates = vk::False;
  shadowSampler = vk::raii::Sampler(device, samplerInfo);
}

void ShadowPass::createPipeline(vk::raii::Device& device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::PipelineLayout& shadowPipelineLayout,
    vk::raii::Pipeline& shadowPipeline,
    vk::raii::DescriptorSetLayout& shadowDescriptorLayout)
{
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

    Pipeline::PipelineConfig config{};
    config.shaderStages = {
        { "D:\\dev\\Graphics\\NovusEngine\\shaders\\shadow.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" }
    };
    config.vertexBindings = { Vertex::getBindingDescription() };
    config.vertexAttributes = {
     attributeDescriptions.begin(),
        attributeDescriptions.end()
    };
	config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);
	config.cullMode = vk::CullModeFlagBits::eNone;
	config.depthBiasEnable = true;
	config.depthBiasConstantFactor = 1.25f;
	config.depthBiasSlopeFactor = 1.75f;
	config.blendEnable = false;
	config.descriptorSetLayouts = { {shadowDescriptorLayout} };
	config.pushConstantRanges = { vk::PushConstantRange{
		.stageFlags = vk::ShaderStageFlagBits::eVertex,
		.offset = 0,
		.size = sizeof(int)
	}};

	auto pipelineBundle = Pipeline::createPipeline(device, config);
	shadowPipelineLayout = std::move(pipelineBundle.layout);
	shadowPipeline = std::move(pipelineBundle.pipeline);
}