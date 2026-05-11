#include "command_buffer.h"

bool CommandBuffer::init(vk::raii::Device& device, uint32_t queueIndex, vk::raii::CommandPool& cmdPool, std::vector<vk::raii::CommandBuffer>& cmdBuffers, uint32_t framesInFlight)
{
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = cmdPool,
										 .level = vk::CommandBufferLevel::ePrimary,
										 .commandBufferCount = framesInFlight };
	cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);

	return false;
}
