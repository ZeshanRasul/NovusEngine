#pragma once

#include <ktx.h>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class Texture
{
public:
	static void loadTextureFromFile(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const std::string& filepath, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory, vk::raii::ImageView& imageView, bool isSRGB);
};