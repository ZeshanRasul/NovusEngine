#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class CommandBuffer
{
public:
	static bool init(vk::raii::Device& device, uint32_t queueIndex, vk::raii::CommandPool& cmdPool, std::vector<vk::raii::CommandBuffer>& cmdBuffers, uint32_t framesInFlight);

	static vk::raii::CommandBuffer beginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& cmdPool);

	static void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer, vk::raii::Queue& queue);
};