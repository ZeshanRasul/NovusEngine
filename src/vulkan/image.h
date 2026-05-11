#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class Image
{
public:
	static void createImage(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t width, uint32_t height, vk::Format format,
		vk::ImageTiling tiling, vk::ImageUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory);

	static void transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image,
		vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
};