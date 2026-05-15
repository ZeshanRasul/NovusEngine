// ImGuiVulkanUtil.h
#pragma once

#include <vulkan/vulkan_raii.hpp>
#include "../include/imgui.h"
#include "../include/imconfig.h"   
#include <glm/glm.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <ImGuizmo.h>
#include <IconsFontAwesome7.h>

enum OPERATION
{
	TRANSLATE,
	ROTATE,
	SCALE
};

enum MODE
{
	LOCAL,
	WORLD
};

class ImGuiVulkanUtil {
private:
	// Core GPU rendering resources for UI display
	// These objects form the foundation of our ImGui-to-Vulkan rendering pipeline
	vk::raii::Sampler sampler{ nullptr };                    // Texture sampling configuration for font rendering
	vk::raii::Buffer vertexBuffer = nullptr;                                    // Dynamic vertex buffer for UI geometry
	vk::raii::Buffer indexBuffer = nullptr;                                     // Dynamic index buffer for UI triangle connectivity
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;                    // GPU memory allocation for vertex buffer
	vk::raii::DeviceMemory indexBufferMemory = nullptr;                     // GPU memory allocation for index buffer
	uint32_t vertexCount = 0;                              // Current vertex count for draw commands
	uint32_t indexCount = 0;                               // Current index count for draw commands
	vk::raii::Image fontImage = nullptr;                                        // GPU texture containing ImGui font atlas
	vk::raii::DeviceMemory fontImageMemory = nullptr;                    // GPU memory allocation for font texture
	vk::raii::ImageView fontImageView = nullptr;                                // Shader-accessible view of font texture
	// Vulkan pipeline infrastructure for UI rendering
// These objects define the complete GPU processing pipeline for ImGui elements
	vk::raii::PipelineCache pipelineCache{ nullptr };        // Pipeline compilation cache for faster startup
	vk::raii::PipelineLayout pipelineLayout{ nullptr };      // Resource binding layout (textures, uniforms)
	vk::raii::Pipeline pipeline{ nullptr };                  // Complete graphics pipeline for UI rendering
	vk::raii::DescriptorPool descriptorPool{ nullptr };      // Pool for allocating descriptor sets
	vk::raii::DescriptorSetLayout descriptorSetLayout{ nullptr }; // Layout defining shader resource bindings
	vk::raii::DescriptorSet descriptorSet{ nullptr };        // Actual resource bindings for font texture

	// Vulkan device context and system integration
	// These references connect our UI system to the broader Vulkan application context
	vk::raii::Device* device = nullptr;                    // Primary Vulkan device for resource creation
	vk::raii::PhysicalDevice* physicalDevice = nullptr;    // GPU hardware info for capability queries
	vk::raii::Queue* graphicsQueue = nullptr;              // Command submission queue for UI rendering
	uint32_t graphicsQueueFamily = 0;                      // Queue family index for validation

	// UI state management and rendering configuration
	// These members control the visual appearance and dynamic behavior of the UI system
	ImGuiStyle vulkanStyle;                                 // Custom visual styling for Vulkan applications

	// Push constants for efficient per-frame parameter updates
	// This structure enables fast updates of transformation and styling data
	struct PushConstBlock {
		glm::vec2 scale;                                    // UI scaling factors for different screen sizes
		glm::vec2 translate;                                // Translation offset for UI positioning
	} pushConstBlock;

	// Dynamic state tracking for performance optimization
	bool needsUpdateBuffers = false;                        // Flag indicating buffer resize requirements

	// Modern Vulkan rendering configuration
	vk::PipelineRenderingCreateInfo renderingInfo{};        // Dynamic rendering setup parameters
	vk::Format colorFormat = vk::Format::eB8G8R8A8Unorm;   // Target framebuffer format

public:
	// Lifecycle management for proper resource initialization and cleanup
	ImGuiVulkanUtil(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice,
		vk::raii::Queue& graphicsQueue, uint32_t graphicsQueueFamily);
	~ImGuiVulkanUtil();

	// Core functionality methods for ImGui integration
	void init(float width, float height);                   // Initialize ImGui context and configure display
	void initResources();                                    // Create all Vulkan resources for rendering
	void setStyle(uint32_t index);                          // Apply visual styling themes

	void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory);

	vk::raii::ImageView createImageView(vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags);

	// Frame-by-frame rendering operations
	bool newFrame();                                         // Begin new ImGui frame and generate geometry
	void updateBuffers();                                    // Upload updated geometry to GPU buffers
	void drawFrame(vk::raii::CommandBuffer& commandBuffer, vk::ImageView swapchainImageView); // Record rendering commands to command buffer
	ImTextureID registerTexture(vk::ImageView imageView, vk::Sampler textureSampler, vk::ImageLayout imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal);
	void clearRegisteredTextures();

	// Input event handling for interactive UI elements
	void handleKey(int key, int scancode, int action, int mods); // Process keyboard input events
	bool getWantKeyCapture();                               // Query if ImGui wants keyboard focus
	void charPressed(uint32_t key);                         // Handle character input for text widgets

	void transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
	{
		vk::ImageMemoryBarrier barrier{ .oldLayout = oldLayout,
									   .newLayout = newLayout,
									   .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
									   .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
									   .image = image,
									   .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1} };

		vk::PipelineStageFlags sourceStage;
		vk::PipelineStageFlags destinationStage;

		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
		{
			barrier.srcAccessMask = {};
			barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eTransfer;
		}
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else
		{
			throw std::invalid_argument("unsupported layout transition!");
		}
		commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
	}

	void Manipulate(const float* view, const float* projection, OPERATION operation, MODE mode,
		float* matrix, float* deltaMatrix = 0, float* snap = 0);


	//void EditTransform(float* cameraView, float* cameraProjection, float* matrix)
	//{
	//	static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::ROTATE);
	//	static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
	//	if (ImGui::IsKeyPressed(ImGuiKey_T))
	//		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	//	if (ImGui::IsKeyPressed(ImGuiKey_E))
	//		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	//	if (ImGui::IsKeyPressed(ImGuiKey_R))
	//		mCurrentGizmoOperation = ImGuizmo::SCALE;
	//	if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
	//		mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
	//	ImGui::SameLine();
	//	if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
	//		mCurrentGizmoOperation = ImGuizmo::ROTATE;
	//	ImGui::SameLine();
	//	if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
	//		mCurrentGizmoOperation = ImGuizmo::SCALE;
	//	float matrixTranslation[3], matrixRotation[3], matrixScale[3];
	//	ImGuizmo::DecomposeMatrixToComponents(matrix, matrixTranslation, matrixRotation, matrixScale);
	//	ImGui::InputFloat3("Tr", matrixTranslation);
	//	ImGui::InputFloat3("Rt", matrixRotation);
	//	ImGui::InputFloat3("Sc", matrixScale);
	//	ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, matrix);

	//	if (mCurrentGizmoOperation != ImGuizmo::SCALE)
	//	{
	//		if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
	//			mCurrentGizmoMode = ImGuizmo::LOCAL;
	//		ImGui::SameLine();
	//		if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
	//			mCurrentGizmoMode = ImGuizmo::WORLD;
	//	}
	//	static bool useSnap(false);
	//	if (ImGui::IsKeyPressed(ImGuiKey_S))
	//		useSnap = !useSnap;
	//	ImGui::Checkbox("##useSnap", &useSnap);
	//	ImGui::SameLine();
	//	glm::vec4 snap;
	//	switch (mCurrentGizmoOperation)
	//	{
	//	case ImGuizmo::TRANSLATE:
	//		snap = config.mSnapTranslation;
	//		ImGui::InputFloat3("Snap", &snap.x);
	//		break;
	//	case ImGuizmo::ROTATE:
	//		snap = config.mSnapRotation;
	//		ImGui::InputFloat("Angle Snap", &snap.x);
	//		break;
	//	case ImGuizmo::SCALE:
	//		snap = config.mSnapScale;
	//		ImGui::InputFloat("Scale Snap", &snap.x);
	//		break;
	//	default:
	//		break;
	//	}
	//	ImGuiIO& io = ImGui::GetIO();
	//	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
	//	ImGuizmo::Manipulate(cameraView, cameraProjection, mCurrentGizmoOperation,
	//		mCurrentGizmoMode, matrix, NULL, useSnap ? &snap.x : NULL);
	//}


private:
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
		vk::MemoryPropertyFlags properties,
		vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
	void copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size);
	void copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer,
		vk::raii::Image& image, uint32_t width, uint32_t height);
	vk::raii::CommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer);
	static std::vector<char> readFile(const std::string& filename);
	vk::raii::ShaderModule createShaderModule(const std::vector<char>& code);

	vk::raii::CommandPool commandPool{ nullptr };
	std::vector<vk::raii::DescriptorSet> userTextureDescriptorSets;
};
