#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class ShadowPass
{
public:
    static constexpr uint32_t ShadowMapWidth = 2048;
    static constexpr uint32_t ShadowMapHeight = 2048;

    static void createResources(vk::raii::Device& device,
        vk::raii::PhysicalDevice& physicalDevice,
        vk::raii::Image& shadowImage,
        vk::raii::DeviceMemory& shadowImageMemory,
        vk::raii::ImageView& shadowImageView,
        vk::raii::Sampler& shadowSampler);

    static void createPipeline(vk::raii::Device& device,
        vk::raii::PhysicalDevice& physicalDevice,
        vk::raii::PipelineLayout& shadowPipelineLayout,
        vk::raii::Pipeline& shadowPipeline,
        vk::raii::DescriptorSetLayout& shadowDescriptorLayout);
};
