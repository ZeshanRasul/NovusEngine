#include "command_buffer.h"

bool CommandBuffer::init(vk::raii::Device& device, uint32_t queueIndex, vk::raii::CommandPool& cmdPool, std::vector<vk::raii::CommandBuffer>& cmdBuffers, uint32_t framesInFlight)
{
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = cmdPool,
										 .level = vk::CommandBufferLevel::ePrimary,
										 .commandBufferCount = framesInFlight };
	cmdBuffers = vk::raii::CommandBuffers(device, allocInfo);

	return false;
}

vk::raii::CommandBuffer CommandBuffer::beginSingleTimeCommands(vk::raii::Device& device, vk::raii::CommandPool& cmdPool)
{
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = cmdPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
	vk::raii::CommandBuffer       commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

	vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	commandBuffer.begin(beginInfo);

	return commandBuffer;
}

void CommandBuffer::endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer, vk::raii::Queue& queue)
{
	commandBuffer.end();

	vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
	queue.submit(submitInfo, nullptr);
	queue.waitIdle();
}
