#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "ECS/components/camera_component.h"
#include "ECS/entity.h"
#include "ECS/components/transform_component.h"
#include "renderer/renderer_types.h"
#include "ECS/components/renderable_component.h"
#include "../imgui_vulkan_util.h"
#include "../input/input_system.h"

constexpr uint32_t WIDTH               = 1920;
constexpr uint32_t HEIGHT              = 1080;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class Renderer
{
public:
	Renderer()  = default;
	~Renderer() = default;

	void run();

private:
	// -------------------------------------------------------------------------
	// Nested types
	// -------------------------------------------------------------------------
	struct PushConstantBlock
	{
		glm::vec4 baseColorFactor;
		float     metallicFactor;
		float     roughnessFactor;
		int       baseColorTextureSet;
		int       physicalDescriptorTextureSet;
		int       normalTextureSet;
		int       occlusionTextureSet;
		int       emissiveTextureSet;
		float     alphaMask;
		float     alphaMaskCutoff;
	};

	// -------------------------------------------------------------------------
	// Initialisation / shutdown
	// -------------------------------------------------------------------------
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanup();

	// -------------------------------------------------------------------------
	// Vulkan setup
	// -------------------------------------------------------------------------
	void                     createInstance();
	void                     setupDebugMessenger();
	void                     createSurface();
	bool                     isDeviceSuitable(vk::raii::PhysicalDevice const& pd);
	void                     pickPhysicalDevice();
	void                     createLogicalDevice();
	std::vector<const char*> getRequiredInstanceExtensions();

	bool deviceInit();

	// Swap chain
	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& formats);
	vk::PresentModeKHR   chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& modes);
	vk::Extent2D         chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities);
	uint32_t             chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& caps);
	void                 createSwapChain();
	void                 recreateSwapChain();
	void                 cleanupSwapChain();

	// Image views
	void                createSwapChainImageViews();

	// Shaders / pipelines
	static std::vector<char> readFile(const std::string& filename);
	vk::raii::ShaderModule   createShaderModule(const std::vector<char>& code) const;
	void                     createDescriptorSetLayout();
	void                     createGraphicsPipeline();
	bool                     createPBRPipeline();

	// Commands
	void                    createCommandPool();
	void                    createCommandBuffers();
	vk::raii::CommandBuffer beginSingleTimeCommands();
	void                    endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer);
	void                    recordCommandBuffer(uint32_t imageIndex);

	// Buffers
	void     createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
						  vk::MemoryPropertyFlags properties,
						  vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory);
	void     copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size);
	void     createVertexBuffer(RenderableComponent& gameObj);
	void     createIndexBuffer(RenderableComponent& gameObj);

	// Images
	void transition_image_layout(vk::Image image,
								 vk::ImageLayout old_layout, vk::ImageLayout new_layout,
								 vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
								 vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask,
								 vk::ImageAspectFlags image_aspect_flags);

	// Depth
	vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates,
								   vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	vk::Format findDepthFormat();
	bool       hasStencilComponent(vk::Format format);
	void       createDepthResources();

	// Textures / samplers
	void loadPBRTextures(const Material& material, RenderableComponent::PBRTextures& textures);
	void createDefaultTextures();
	void createTextureSampler();

	// Descriptors
	void createDescriptorPool();
	void createDescriptorSets();

	// Sync
	void createSyncObjects();

	// Frame
	void drawFrame();
	void pushMaterialProperties(vk::CommandBuffer commandBuffer,
								const RenderableComponent* model, uint32_t materialIndex);

	// Scene
	void setupGameObjects();

	// Callbacks
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
	static void mousePosCallback(GLFWwindow* window, double xpos, double ypos);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL
		debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
					  vk::DebugUtilsMessageTypeFlagsEXT type,
					  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);

	// ImGUI
	void renderImgui();

	// -------------------------------------------------------------------------
	// Members
	// -------------------------------------------------------------------------
	GLFWwindow* window = nullptr;

	vk::raii::Context                context;
	vk::raii::Instance               instance       = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::raii::Device                 device         = nullptr;
	vk::raii::Queue                  queue          = nullptr;
	uint32_t                         queueIndex     = ~0u;

	vk::raii::SurfaceKHR             surface               = nullptr;
	vk::raii::SwapchainKHR           swapChain             = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::Extent2D                     swapChainExtent;
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	bool                             framebufferResized = false;

	vk::raii::DescriptorPool             descriptorPool      = nullptr;
	vk::raii::DescriptorSetLayout        descriptorSetLayout = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	vk::raii::PipelineLayout pipelineLayout   = nullptr;
	vk::raii::Pipeline       graphicsPipeline = nullptr;

	vk::raii::PipelineLayout pbrPipelineLayout = nullptr;
	vk::raii::Pipeline       pbrPipeline       = nullptr;

	vk::raii::CommandPool                commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t                         frameIndex = 0;

	vk::raii::Buffer       vertexBuffer       = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer       indexBuffer        = nullptr;
	vk::raii::DeviceMemory indexBufferMemory  = nullptr;

	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*>                  uniformBuffersMapped;

	vk::raii::Image        textureImage       = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView   = nullptr;
	vk::raii::Sampler      textureSampler     = nullptr;

	vk::raii::Image        depthImage       = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView    depthImageView   = nullptr;

	vk::raii::Image        defaultTextureImage  = nullptr;
	vk::raii::DeviceMemory defaultTextureMemory = nullptr;
	vk::raii::ImageView    defaultTextureView   = nullptr;

	vk::raii::Image        defaultNormalImage  = nullptr;
	vk::raii::DeviceMemory defaultNormalMemory = nullptr;
	vk::raii::ImageView    defaultNormalView   = nullptr;

	std::vector<const char*> requiredDeviceExtension = { vk::KHRSwapchainExtensionName };

	std::vector<std::unique_ptr<Entity>> entities;

	Camera camera;
	float  lastFrameTime = 0.0f;

	ImGuiVulkanUtil* imGui;
};
