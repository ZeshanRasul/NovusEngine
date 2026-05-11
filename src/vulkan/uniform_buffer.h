#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "../ECS/entity.h"
#include "../ECS/components/renderable_component.h"
#include "buffer.h"

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;

	glm::vec4 lightPositions[4];
	glm::vec4 lightColors[4];
	glm::vec4 camPos;
	float     exposure;
	float     gamma;
	float     prefilteredCubeMipLevels;
	float     scaleIBLAmbient;
};

class UniformBuffer
{
public:
	static void createUniformBuffers(std::vector<std::unique_ptr<Entity>>& entities, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t framesInFlight)
	{
		for (auto& entityPtr : entities)
		{
			auto& gameObject = *entityPtr->GetComponent<RenderableComponent>();
			gameObject.uniformBuffers.clear();
			gameObject.uniformBuffersMemory.clear();
			gameObject.uniformBuffersMapped.clear();

			for (size_t i = 0; i < framesInFlight; i++)
			{
				vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
				vk::raii::Buffer       buffer({});
				vk::raii::DeviceMemory bufferMem({});
				Buffer::createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
					vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					buffer, bufferMem);
				gameObject.uniformBuffers.emplace_back(std::move(buffer));
				gameObject.uniformBuffersMemory.emplace_back(std::move(bufferMem));
				gameObject.uniformBuffersMapped.emplace_back(gameObject.uniformBuffersMemory[i].mapMemory(0, bufferSize));
			}
		}
	}
};