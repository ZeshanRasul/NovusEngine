#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class DepthTarget
{
public:
    static vk::Format findSupportedFormat(vk::raii::PhysicalDevice& physicalDevice,
        const std::vector<vk::Format>& candidates,
        vk::ImageTiling tiling,
        vk::FormatFeatureFlags features);

    static vk::Format findDepthFormat(vk::raii::PhysicalDevice& physicalDevice);
    static bool hasStencilComponent(vk::Format format);

    static void createDepthResources(vk::raii::Device& device,
        vk::raii::PhysicalDevice& physicalDevice,
        vk::Extent2D swapChainExtent,
        vk::raii::Image& depthImage,
        vk::raii::DeviceMemory& depthImageMemory,
        vk::raii::ImageView& depthImageView);
};