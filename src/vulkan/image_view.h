#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class ImageView
{
public:
	static vk::raii::ImageView createImageView(vk::raii::Device& device, vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags);
};
