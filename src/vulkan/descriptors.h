#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include "../ECS/entity.h"

class DescriptorPool
{
public:
	static void createDescriptorPool(vk::raii::Device& device, std::vector<std::unique_ptr<Entity>>& entities, vk::raii::DescriptorPool& descriptorPool, uint32_t framesInFlight);
};

class DescriptorSet
{
public:
	static void createDescriptorSets(vk::raii::Device& device, std::vector<std::unique_ptr<Entity>>& entities, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout,
		vk::raii::ImageView& defaultTextureView, vk::raii::ImageView& defaultNormalView, vk::raii::Sampler& textureSampler, uint32_t framesInFlight);
};

class DescriptorSetLayout
{
public:
	static void createEntityDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout, size_t numBindings);
};