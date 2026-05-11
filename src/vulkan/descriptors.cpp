#include "descriptors.h"
#include "../ECS/components/renderable_component.h"
#include "uniform_buffer.h""

void DescriptorPool::createDescriptorPool(vk::raii::Device& device, std::vector<std::unique_ptr<Entity>>& entities, vk::raii::DescriptorPool& descriptorPool, uint32_t framesInFlight)
{
	uint32_t materialCount = 0;
	for (auto& entityPtr : entities)
	{
		auto* renderable = entityPtr->GetComponent<RenderableComponent>();
		if (renderable)
			materialCount += static_cast<uint32_t>(std::max<size_t>(1, renderable->materials.size()));
	}

	std::array<vk::DescriptorPoolSize, 2> poolSize{ {
		{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = materialCount * framesInFlight },
		{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 5 * materialCount * framesInFlight }
	} };

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = materialCount * framesInFlight,
		.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		.pPoolSizes = poolSize.data()
	};

	descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

}

void DescriptorSet::createDescriptorSets(vk::raii::Device& device, std::vector<std::unique_ptr<Entity>>& entities, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout,
	vk::raii::ImageView& defaultTextureView, vk::raii::ImageView& defaultNormalView, vk::raii::Sampler& textureSampler, uint32_t framesInFlight)
{
	for (auto& entityPtr : entities)
	{
		auto& gameObject = *entityPtr->GetComponent<RenderableComponent>();
		gameObject.materialDescriptorSets.clear();
		gameObject.materialDescriptorSets.resize(gameObject.materials.size());

		for (size_t materialIndex = 0; materialIndex < gameObject.materials.size(); ++materialIndex)
		{
			auto& materialTextures = gameObject.materialTextures[materialIndex];
			auto& descriptorSetsForMaterial = gameObject.materialDescriptorSets[materialIndex];

			std::vector<vk::DescriptorSetLayout> layouts(framesInFlight, *descriptorSetLayout);
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptorPool,
				.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
				.pSetLayouts = layouts.data()
			};

			descriptorSetsForMaterial = device.allocateDescriptorSets(allocInfo);

			for (size_t i = 0; i < framesInFlight; i++)
			{
				vk::DescriptorBufferInfo bufferInfo{
					.buffer = *gameObject.uniformBuffers[i],
					.offset = 0,
					.range = sizeof(UniformBufferObject)
				};

				vk::ImageView baseColorView = (*materialTextures.baseColorView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.baseColorView) : vk::ImageView(*defaultTextureView);
				vk::ImageView metallicRoughnessView = (*materialTextures.metallicRoughnessView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.metallicRoughnessView) : vk::ImageView(*defaultTextureView);
				vk::ImageView normalView = (*materialTextures.normalView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.normalView) : vk::ImageView(*defaultNormalView);
				vk::ImageView occlusionView = (*materialTextures.occlusionView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.occlusionView) : vk::ImageView(*defaultTextureView);
				vk::ImageView emissiveView = (*materialTextures.emissiveView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.emissiveView) : vk::ImageView(*defaultTextureView);

				vk::DescriptorImageInfo baseColorInfo{ .sampler = *textureSampler, .imageView = baseColorView,        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo metallicRoughnessInfo{ .sampler = *textureSampler, .imageView = metallicRoughnessView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo normalInfo{ .sampler = *textureSampler, .imageView = normalView,            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo occlusionInfo{ .sampler = *textureSampler, .imageView = occlusionView,         .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo emissiveInfo{ .sampler = *textureSampler, .imageView = emissiveView,          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

				std::array descriptorWrites{
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,        .pBufferInfo = &bufferInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &baseColorInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &metallicRoughnessInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &occlusionInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &emissiveInfo }
				};

				device.updateDescriptorSets(descriptorWrites, {});
			}
		}
	}

}

void DescriptorSetLayout::createEntityDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout, size_t numBindings)
{
	const int expectedBindings = 6;
	std::array<vk::DescriptorSetLayoutBinding, expectedBindings> bindings{ {
	{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
	{.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	} };

	vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()),
												  .pBindings = bindings.data() };
	descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}
