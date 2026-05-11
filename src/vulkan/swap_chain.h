#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

struct GLFWwindow;

class SwapChain
{
public:
    static vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats);
    static vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& modes);
    static vk::Extent2D chooseExtent(vk::SurfaceCapabilitiesKHR const& capabilities, GLFWwindow* window);
    static uint32_t chooseMinImageCount(vk::SurfaceCapabilitiesKHR const& capabilities);

    static void createSwapChain(vk::raii::PhysicalDevice& physicalDevice,
        vk::raii::Device& device,
        vk::raii::SurfaceKHR& surface,
        GLFWwindow* window,
        vk::raii::SwapchainKHR& swapChain,
        std::vector<vk::Image>& swapChainImages,
        vk::Extent2D& swapChainExtent,
        vk::SurfaceFormatKHR& swapChainSurfaceFormat);

    static void createImageViews(vk::raii::Device& device,
        const std::vector<vk::Image>& swapChainImages,
        vk::Format swapChainFormat,
        std::vector<vk::raii::ImageView>& swapChainImageViews);

    static void cleanupSwapChain(std::vector<vk::raii::ImageView>& swapChainImageViews,
        vk::raii::SwapchainKHR& swapChain);
};