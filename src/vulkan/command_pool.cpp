#include "command_pool.h"

bool CommandPool::init(vk::raii::Device& device, uint32_t queueIndex, vk::raii::CommandPool& cmdPool)
{
	vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
										.queueFamilyIndex = queueIndex };
	cmdPool = vk::raii::CommandPool(device, poolInfo);
	return true;
}
