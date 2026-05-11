#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class CommandPool
{
public:
	static bool init(vk::raii::Device& device, uint32_t queueIndex, vk::raii::CommandPool& cmdPool);
};
