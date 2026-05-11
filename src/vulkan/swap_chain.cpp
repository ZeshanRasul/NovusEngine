#include "swap_chain.h"

#include <algorithm>
#include <cassert>
#include <limits>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "image_view.h"

vk::SurfaceFormatKHR SwapChain::chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    assert(!formats.empty());
    const auto it = std::ranges::find_if(formats, [](const auto& format) {
        return format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
    });

    return it != formats.end() ? *it : formats[0];
}

vk::PresentModeKHR SwapChain::choosePresentMode(const std::vector<vk::PresentModeKHR>& modes)
{
    assert(std::ranges::any_of(modes, [](auto mode) { return mode == vk::PresentModeKHR::eFifo; }));
    return std::ranges::any_of(modes, [](auto mode) { return mode == vk::PresentModeKHR::eMailbox; })
        ? vk::PresentModeKHR::eMailbox
        : vk::PresentModeKHR::eFifo;
}

vk::Extent2D SwapChain::chooseExtent(vk::SurfaceCapabilitiesKHR const& capabilities, GLFWwindow* window)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        return capabilities.currentExtent;

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);

    return {
        std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

uint32_t SwapChain::chooseMinImageCount(vk::SurfaceCapabilitiesKHR const& capabilities)
{
    auto minImageCount = std::max(3u, capabilities.minImageCount);
    if ((0 < capabilities.maxImageCount) && (capabilities.maxImageCount < minImageCount))
        minImageCount = capabilities.maxImageCount;

    return minImageCount;
}

void SwapChain::createSwapChain(vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::Device& device,
    vk::raii::SurfaceKHR& surface,
    GLFWwindow* window,
    vk::raii::SwapchainKHR& swapChain,
    std::vector<vk::Image>& swapChainImages,
    vk::Extent2D& swapChainExtent,
    vk::SurfaceFormatKHR& swapChainSurfaceFormat)
{
    auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseExtent(surfaceCapabilities, window);
    const uint32_t minImageCount = chooseMinImageCount(surfaceCapabilities);

    const auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    const auto availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    swapChainSurfaceFormat = chooseSurfaceFormat(availableFormats);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface = *surface,
        .minImageCount = minImageCount,
        .imageFormat = swapChainSurfaceFormat.format,
        .imageColorSpace = swapChainSurfaceFormat.colorSpace,
        .imageExtent = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform = surfaceCapabilities.currentTransform,
        .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode = choosePresentMode(availablePresentModes),
        .clipped = true,
        .oldSwapchain = nullptr
    };

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
}

void SwapChain::createImageViews(vk::raii::Device& device,
    const std::vector<vk::Image>& swapChainImages,
    vk::Format swapChainFormat,
    std::vector<vk::raii::ImageView>& swapChainImageViews)
{
    assert(swapChainImageViews.empty());
    swapChainImageViews.reserve(swapChainImages.size());
    for (auto& image : swapChainImages)
        swapChainImageViews.emplace_back(ImageView::createImageView(device, image, swapChainFormat, vk::ImageAspectFlagBits::eColor));
}

void SwapChain::cleanupSwapChain(std::vector<vk::raii::ImageView>& swapChainImageViews,
    vk::raii::SwapchainKHR& swapChain)
{
    swapChainImageViews.clear();
    swapChain = nullptr;
}