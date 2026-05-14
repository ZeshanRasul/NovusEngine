#include <cmath>
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include "../renderer/shadow_pass.h"
#include "uniform_buffer.h"

void UniformBuffer::createUniformBuffers(entt::registry& registry, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t framesInFlight)
{
    for (auto [entity, gameObject] : registry.view<RenderableComponent>().each())
	{
      (void)entity;

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

void UniformBuffer::updateUniformBuffer(uint32_t currentFrame, RenderableComponent* renderable, TransformComponent* transform, Camera* cam, VkExtent2D swapChainExtent, const ShadowSettings& shadowSettings,
	const std::array<glm::vec4, 4>& pointLightPositions,
	const std::array<glm::vec4, 4>& pointLightColors)
{
	UniformBufferObject ubo{};

	ubo.model = transform ? transform->GetTransformMatrix() : glm::mat4(1.0f);

	if (cam)
	{
		ubo.view = cam->getViewMatrix();
		ubo.proj = cam->getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.005f, 3000.0f);
	}
	else
	{
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(55.0f), swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.005f, 3000.0f);
		ubo.proj[1][1] *= -1;
	}

	// ---------------------------------------------------------------------------
	// Cascaded Shadow Maps
	// ---------------------------------------------------------------------------
	// Camera frustum parameters (must match the projection built above)
	const float camNear = 0.005f;
	const float camFar = 3000.0f;
	const float shadowMaxDistance = glm::clamp(shadowSettings.shadowMaxDistance, camNear, camFar);
	const float cascadeFar = glm::min(camFar, shadowMaxDistance);
	const float lambda = glm::clamp(shadowSettings.lambda, 0.0f, 1.0f);

	// Compute cascade split distances in view space
	float splits[SHADOW_CASCADE_COUNT];
	for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
	{
		float p = static_cast<float>(i + 1) / static_cast<float>(SHADOW_CASCADE_COUNT);
		float logSplit = camNear * std::pow(cascadeFar / camNear, p);
		float uniSplit = camNear + (cascadeFar - camNear) * p;
		splits[i] = lambda * logSplit + (1.0f - lambda) * uniSplit;
	}
	ubo.cascadeSplits = glm::vec4(splits[0], splits[1], splits[2], splits[3]);

	// Shared light direction
	glm::vec3 lightDir = shadowSettings.lightDirection;
	if (glm::dot(lightDir, lightDir) < 1e-6f)
	{
		lightDir = glm::vec3(23.0f, 90.0f, -8.0f);
	}
	lightDir = glm::normalize(lightDir);
	ubo.directionalLightDirection = glm::vec4(lightDir, 0.0f);
	ubo.directionalLightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);

	const float aspectRatio = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
	const float tanHalfFov = std::tan(glm::radians(cam ? cam->getZoom() : 45.0f) * 0.5f);
	const glm::mat4 invView = glm::inverse(ubo.view);
	const glm::vec3 camPosition = glm::vec3(invView[3]);
	const glm::vec3 camRight = glm::normalize(glm::vec3(invView[0]));
	const glm::vec3 camUp = glm::normalize(glm::vec3(invView[1]));
	const glm::vec3 camForward = glm::normalize(-glm::vec3(invView[2]));

	// Build a light-space matrix per cascade
	float prevSplit = camNear;
	for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade)
	{
		float nearDist = prevSplit;
		float farDist = splits[cascade];
		prevSplit = farDist;
		const float cascadeT = SHADOW_CASCADE_COUNT > 1 ? static_cast<float>(cascade) / static_cast<float>(SHADOW_CASCADE_COUNT - 1) : 0.0f;
		const float cascadeExpansion = glm::mix(1.0f, glm::max(1.0f, shadowSettings.farCascadeExpansion), cascadeT * cascadeT);

		const float nearHalfHeight = tanHalfFov * nearDist;
		const float nearHalfWidth = nearHalfHeight * aspectRatio;
		const float farHalfHeight = tanHalfFov * farDist;
		const float farHalfWidth = farHalfHeight * aspectRatio;

		const glm::vec3 nearCenter = camPosition + camForward * nearDist;
		const glm::vec3 farCenter = camPosition + camForward * farDist;

		glm::vec3 frustumCorners[8] = {
			nearCenter + camUp * nearHalfHeight - camRight * nearHalfWidth,
			nearCenter + camUp * nearHalfHeight + camRight * nearHalfWidth,
			nearCenter - camUp * nearHalfHeight + camRight * nearHalfWidth,
			nearCenter - camUp * nearHalfHeight - camRight * nearHalfWidth,
			farCenter + camUp * farHalfHeight - camRight * farHalfWidth,
			farCenter + camUp * farHalfHeight + camRight * farHalfWidth,
			farCenter - camUp * farHalfHeight + camRight * farHalfWidth,
			farCenter - camUp * farHalfHeight - camRight * farHalfWidth,
		};

		// Frustum centre for the light-view lookat
		glm::vec3 center(0.0f);
		for (const auto& c : frustumCorners) center += c;
		center /= 8.0f;

		float radius = 0.0f;
		for (const auto& c : frustumCorners)
		{
			radius = glm::max(radius, glm::length(c - center));
		}

		radius = std::ceil(radius * 32.0f) / 32.0f;

		glm::mat4 lightView = glm::lookAtRH(center - lightDir * (radius * 2.0f), center, glm::vec3(0.0f, 0.0f, 1.0f));

		// Compute AABB in light space around the frustum corners
		glm::vec3 mn(1e9f), mx(-1e9f);
		const glm::vec3 casterOffset = -lightDir * (shadowSettings.casterPadding * cascadeExpansion);
		for (const auto& c : frustumCorners)
		{
			glm::vec3 lc = glm::vec3(lightView * glm::vec4(c, 1.0f));
			mn = glm::min(mn, lc);
			mx = glm::max(mx, lc);

			glm::vec3 casterLc = glm::vec3(lightView * glm::vec4(c + casterOffset, 1.0f));
			mn = glm::min(mn, casterLc);
			mx = glm::max(mx, casterLc);
		}

		glm::vec3 centerLS = glm::vec3(lightView * glm::vec4(center, 1.0f));
		const float texelSize = (radius * 2.0f) / static_cast<float>(ShadowPass::ShadowMapWidth);
		centerLS.x = std::floor(centerLS.x / texelSize) * texelSize;
		centerLS.y = std::floor(centerLS.y / texelSize) * texelSize;

		// Small padding to avoid shadow edge clipping
		const float padding = shadowSettings.shadowPadding * cascadeExpansion;
		const float coveragePadding = glm::max(0.0f, radius * shadowSettings.coveragePaddingFactor * cascadeExpansion);
		const float depthPadding = glm::max(0.0f, radius * shadowSettings.depthPaddingFactor * cascadeExpansion);
		glm::mat4 lightProj = glm::orthoRH_ZO(centerLS.x - radius - coveragePadding, centerLS.x + radius + coveragePadding,
			centerLS.y - radius - coveragePadding, centerLS.y + radius + coveragePadding,
			-mx.z - padding - depthPadding, -mn.z + padding + depthPadding);

		ubo.lightSpaceMatrices[cascade] = lightProj * lightView;
	}

    for (size_t i = 0; i < 4; ++i)
	{
		ubo.lightPositions[i] = pointLightPositions[i];
		ubo.lightColors[i] = pointLightColors[i];
	}

	ubo.camPos = glm::vec4(cam ? cam->getPosition() : glm::vec3(2.0f, 2.0f, 2.0f), 1.0f);
	ubo.exposure = 2.0f;
	ubo.gamma = 2.2f;
	ubo.prefilteredCubeMipLevels = 1.0f;
	ubo.scaleIBLAmbient = 0.002f;
	ubo.shadowTuning = glm::vec4(shadowSettings.biasScale, shadowSettings.biasMin, shadowSettings.cascadeBlendFactor, 0.0f);
	ubo.shadowDebug = glm::vec4(shadowSettings.cascadeDebugView, 0.0f, 0.0f, 0.0f);

	memcpy(renderable->uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}
