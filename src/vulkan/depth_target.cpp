#include "depth_target.h"

#include <stdexcept>

#include "image.h"
#include "image_view.h"

vk::Format DepthTarget::findSupportedFormat(vk::raii::PhysicalDevice& physicalDevice,
    const std::vector<vk::Format>& candidates,
    vk::ImageTiling tiling,
    vk::FormatFeatureFlags features)
{
    for (const auto format : candidates)
    {
        const auto properties = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features)
            return format;

        if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features)
            return format;
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::Format DepthTarget::findDepthFormat(vk::raii::PhysicalDevice& physicalDevice)
{
    return findSupportedFormat(
        physicalDevice,
        { vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool DepthTarget::hasStencilComponent(vk::Format format)
{
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void DepthTarget::createDepthResources(vk::raii::Device& device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::Extent2D swapChainExtent,
    vk::raii::Image& depthImage,
    vk::raii::DeviceMemory& depthImageMemory,
    vk::raii::ImageView& depthImageView)
{
    const auto depthFormat = findDepthFormat(physicalDevice);
    Image::createImage(device, physicalDevice, swapChainExtent.width, swapChainExtent.height, depthFormat,
        vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
    depthImageView = ImageView::createImageView(device, depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}