#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "../ECS/entity.h"
#include "../ECS/components/renderable_component.h"
#include "../ECS/components/transform_component.h"
#include "../ECS/components/camera_component.h"
#include "buffer.h"

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	alignas(16) glm::mat4 lightSpaceMatrix;

	glm::vec4 directionalLightDirection;
	glm::vec4 directionalLightColor;
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
	static void createUniformBuffers(std::vector<std::unique_ptr<Entity>>& entities, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t framesInFlight);

	static void updateUniformBuffer(uint32_t currentFrame, RenderableComponent* renderable,
		TransformComponent* transform, Camera* cam, VkExtent2D swapChainExtent);
};