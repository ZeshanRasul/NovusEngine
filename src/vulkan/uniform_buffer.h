#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <entt/entt.hpp>
#include "../ECS/components/renderable_component.h"
#include "../ECS/components/transform_component.h"
#include "../ECS/components/camera_component.h"
#include "buffer.h"

static constexpr uint32_t SHADOW_CASCADE_COUNT = 5;

struct ShadowSettings
{
	float shadowMaxDistance = 3000.0f;
	float lambda = 0.3f;
	float biasScale = 0.0015f;
	float biasMin = 0.0002f;
	float enabled = 1.0f;
	glm::vec3 lightDirection = glm::normalize(glm::vec3(23.0f, 90.0f, -8.0f));
	float cascadeBlendFactor = 0.15f;
	float cascadeDebugView = 0.0f;
	float shadowPadding = 15.0f;
	float coveragePaddingFactor = 0.08f;
	float depthPaddingFactor = 0.2f;
	float casterPadding = 60.0f;
	float farCascadeExpansion = 2.0f;
};

struct UniformBufferObject

{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
	alignas(16) glm::mat4 lightSpaceMatrices[SHADOW_CASCADE_COUNT];

	glm::vec4 directionalLightDirection;
	glm::vec4 directionalLightColor;
    glm::vec4 lightPositions[MAX_POINT_LIGHTS];
	glm::vec4 lightColors[MAX_POINT_LIGHTS];
	glm::vec4 camPos;
	glm::vec4 cascadeSplits; // xyz = split distances, w unused
	float     exposure;
	float     gamma;
	float     prefilteredCubeMipLevels;
	float     scaleIBLAmbient;
	glm::vec4 shadowTuning;
	glm::vec4 shadowDebug;
};

class UniformBuffer
{
public:
	static void createUniformBuffers(entt::registry& registry, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t framesInFlight);

	static void updateUniformBuffer(uint32_t currentFrame, RenderableComponent* renderable,
		TransformComponent* transform, Camera* cam, VkExtent2D swapChainExtent, const ShadowSettings& shadowSettings,
        const std::array<glm::vec4, MAX_POINT_LIGHTS>& pointLightPositions,
		const std::array<glm::vec4, MAX_POINT_LIGHTS>& pointLightColors);
};