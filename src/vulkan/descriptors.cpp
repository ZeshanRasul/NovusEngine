#include "descriptors.h"
#include <array>
#include <algorithm>
#include "../ECS/components/renderable_component.h"
#include "uniform_buffer.h"

void DescriptorPool::createDescriptorPool(vk::raii::Device& device, entt::registry& registry, vk::raii::DescriptorPool& descriptorPool, uint32_t framesInFlight)
{
	uint32_t materialCount = 0;
    for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
      (void)entity;
		materialCount += static_cast<uint32_t>(std::max<size_t>(1, renderable.materials.size()));
	}

	std::array<vk::DescriptorPoolSize, 2> poolSize{ {
		 {.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = materialCount * framesInFlight },
		 {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 13 * materialCount * framesInFlight }
	 } };

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = materialCount * framesInFlight,
		.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		.pPoolSizes = poolSize.data()
	};

	descriptorPool = vk::raii::DescriptorPool(device, poolInfo);

}

void DescriptorPool::createFxaaDescriptorPool(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool, uint32_t framesInFlight)
{
	std::array<vk::DescriptorPoolSize, 1> poolSize{ {
          {.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 3 * framesInFlight}
	} };

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = framesInFlight,
		.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		.pPoolSizes = poolSize.data()
	};

	descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void DescriptorSet::createDescriptorSets(vk::raii::Device& device, entt::registry& registry, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout,
	vk::raii::ImageView& defaultTextureView, vk::raii::ImageView& defaultNormalView, vk::raii::Sampler& textureSampler,
	std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowImageViews, vk::raii::Sampler& shadowSampler,
	vk::raii::ImageView& iblIrradianceView, vk::raii::ImageView& iblPrefilteredView,
	vk::raii::ImageView& iblBrdfLutView, vk::raii::Sampler& iblSampler,
	uint32_t framesInFlight)
{
    for (auto [entity, gameObject] : registry.view<RenderableComponent>().each())
	{
      (void)entity;
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
				vk::DescriptorImageInfo shadowInfo0{ .sampler = *shadowSampler, .imageView = shadowImageViews[0], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo shadowInfo1{ .sampler = *shadowSampler, .imageView = shadowImageViews[1], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo shadowInfo2{ .sampler = *shadowSampler, .imageView = shadowImageViews[2], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo shadowInfo3{ .sampler = *shadowSampler, .imageView = shadowImageViews[3], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo shadowInfo4{ .sampler = *shadowSampler, .imageView = shadowImageViews[4], .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo iblIrrInfo{ .sampler = *iblSampler, .imageView = *iblIrradianceView,  .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo iblPreInfo{ .sampler = *iblSampler, .imageView = *iblPrefilteredView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
				vk::DescriptorImageInfo iblBrdfInfo{ .sampler = *iblSampler, .imageView = *iblBrdfLutView,    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

				std::array descriptorWrites{
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 0,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,        .pBufferInfo = &bufferInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 1,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &baseColorInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 2,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &metallicRoughnessInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 3,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 4,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &occlusionInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 5,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &emissiveInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 6,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfo0 },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 7,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfo1 },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 8,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfo2 },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 9,  .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfo3 },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 10, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfo4 },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 11, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblIrrInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 12, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblPreInfo },
					vk::WriteDescriptorSet{.dstSet = *descriptorSetsForMaterial[i], .dstBinding = 13, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblBrdfInfo }
				};

				device.updateDescriptorSets(descriptorWrites, {});
			}
		}
	}

}

void DescriptorSet::createFxaaDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout, vk::raii::ImageView& inputImageView,
  vk::raii::ImageView& bloomImageView, vk::raii::ImageView& depthImageView, vk::raii::Sampler& textureSampler, uint32_t framesInFlight, std::vector<vk::raii::DescriptorSet>& fxaaDescriptorSets)
{
 fxaaDescriptorSets.clear();

	std::vector<vk::DescriptorSetLayout> layouts(framesInFlight, *descriptorSetLayout);
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *descriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
		.pSetLayouts = layouts.data()
	};
	fxaaDescriptorSets = device.allocateDescriptorSets(allocInfo);
	for (size_t i = 0; i < framesInFlight; i++)
	{
      vk::DescriptorImageInfo sceneInfo{ .sampler = *textureSampler, .imageView = *inputImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::DescriptorImageInfo bloomInfo{ .sampler = *textureSampler, .imageView = *bloomImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
       vk::DescriptorImageInfo depthInfo{ .sampler = *textureSampler, .imageView = *depthImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		std::array<vk::WriteDescriptorSet, 3> descriptorWrites{ {
			{ .dstSet = *fxaaDescriptorSets[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &sceneInfo },
          { .dstSet = *fxaaDescriptorSets[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &bloomInfo },
			{ .dstSet = *fxaaDescriptorSets[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depthInfo }
		} };
		device.updateDescriptorSets(descriptorWrites, {});
	};
}

void DescriptorSetLayout::createEntityDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout, size_t numBindings)
{
	const int expectedBindings = 14; // 1 UBO + 5 material textures + 5 cascade shadow maps + 3 IBL
	(void)numBindings;
	std::array<vk::DescriptorSetLayoutBinding, expectedBindings> bindings{ {
	{.binding = 0,  .descriptorType = vk::DescriptorType::eUniformBuffer,        .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
	{.binding = 1,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 2,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 3,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 4,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 5,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 6,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 7,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 8,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 9,  .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 10, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 11, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 12, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	{.binding = 13, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
	} };

	vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()),
												  .pBindings = bindings.data() };
	descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void DescriptorSetLayout::createFxaaDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout)
{
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{ {
		{.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
		{.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment }
		} };

	vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()),
												  .pBindings = bindings.data() };
	descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}
