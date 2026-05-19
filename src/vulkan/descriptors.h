#pragma once

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <array>
#include <entt/entt.hpp>
#include "uniform_buffer.h"

class DescriptorPool
{
public:
  static void createDescriptorPool(vk::raii::Device& device, entt::registry& registry, vk::raii::DescriptorPool& descriptorPool, uint32_t framesInFlight);

	static void createFxaaDescriptorPool(vk::raii::Device& device, vk::raii::DescriptorPool & descriptorPool, uint32_t framesInFlight);
};

class DescriptorSet
{
public:
    static void createDescriptorSets(vk::raii::Device& device, entt::registry& registry, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout,
		vk::raii::ImageView& defaultTextureView, vk::raii::ImageView& defaultNormalView, vk::raii::Sampler& textureSampler,
		std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowImageViews, vk::raii::Sampler& shadowSampler,
		vk::raii::ImageView& iblIrradianceView, vk::raii::ImageView& iblPrefilteredView,
		vk::raii::ImageView& iblBrdfLutView, vk::raii::Sampler& iblSampler,
		uint32_t framesInFlight);

	static void createFxaaDescriptorSets(vk::raii::Device& device, vk::raii::DescriptorPool& descriptorPool, vk::raii::DescriptorSetLayout& descriptorSetLayout,
    vk::raii::ImageView& inputImageView, vk::raii::ImageView& bloomImageView, vk::raii::ImageView& depthImageView, vk::raii::Sampler& textureSampler, uint32_t framesInFlight, std::vector<vk::raii::DescriptorSet>& fxaaDescriptorSets);
};

class DescriptorSetLayout
{
public:
	static void createEntityDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout, size_t numBindings);
	static void createFxaaDescriptorSetLayout(vk::raii::Device& device, vk::raii::DescriptorSetLayout& descriptorSetLayout);
};