#include <glm/glm.hpp>
#include "uniform_buffer.h"

void UniformBuffer::createUniformBuffers(std::vector<std::unique_ptr<Entity>>& entities, vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, uint32_t framesInFlight)
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

void UniformBuffer::updateUniformBuffer(uint32_t currentFrame, RenderableComponent* renderable, TransformComponent* transform, Camera* cam, VkExtent2D swapChainExtent)
{
	UniformBufferObject ubo{};

	ubo.model = transform ? transform->GetTransformMatrix() : glm::mat4(1.0f);

	if (cam)
	{
		ubo.view = cam->getViewMatrix();
		ubo.proj = cam->getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10000.0f);
	}
	else
	{
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 10000.0f);
		ubo.proj[1][1] *= -1;
	}

	ubo.lightPositions[0] = glm::vec4(0.0f, 15.0f, 0.0f, 1.0f);
	ubo.lightPositions[1] = glm::vec4(-10.0f, 10.0f, 5.0f, 1.0f);
	ubo.lightPositions[2] = glm::vec4(10.0f, 10.0f, -5.0f, 1.0f);
	ubo.lightPositions[3] = glm::vec4(0.0f, 10.0f, -10.0f, 1.0f);

	ubo.lightColors[0] = glm::vec4(1000.0f, 1000.0f, 1000.0f, 1.0f);
	ubo.lightColors[1] = glm::vec4(800.0f, 200.0f, 200.0f, 1.0f);
	ubo.lightColors[2] = glm::vec4(200.0f, 200.0f, 800.0f, 1.0f);
	ubo.lightColors[3] = glm::vec4(200.0f, 800.0f, 200.0f, 1.0f);

	ubo.camPos = glm::vec4(cam ? cam->getPosition() : glm::vec3(2.0f, 2.0f, 2.0f), 1.0f);
	ubo.exposure = 1.0f;
	ubo.gamma = 2.2f;
	ubo.prefilteredCubeMipLevels = 1.0f;
	ubo.scaleIBLAmbient = 0.2f;

	memcpy(renderable->uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}
