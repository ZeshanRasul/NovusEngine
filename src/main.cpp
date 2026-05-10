#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <array>
#include <chrono>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <ktx.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN        // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#include "ECS/components/camera_component.h"
#include "ECS/entity.h"
#include "ECS/components/transform_component.h"

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;
constexpr int MAX_OBJECTS = 3;

const std::string  MODEL_PATH = "../models/swat.gltf";
const std::string  TEXTURE_PATH = "../textures/Swat_Ch15_body_BaseColor.ktx";


const std::vector<char const*> validationLayers = {
	"VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;
	glm::vec4 tangent;

	static vk::VertexInputBindingDescription getBindingDescription()
	{
		return { .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex };
	}

	static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
		return {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
			vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)),
			vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)),
			vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent))
		};
	};

	bool operator==(const Vertex& other) const
	{
		return pos == other.pos && normal == other.normal && texCoord == other.texCoord && tangent == other.tangent;
	}

};

namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1) ^ (hash<glm::vec4>()(vertex.tangent) << 1);
		}
	};
}

struct UniformBufferObject
{
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;

	glm::vec4 lightPositions[4];           // Light positions in world space
	glm::vec4 lightColors[4];              // Light intensities and colors
	glm::vec4 camPos;                      // Camera position for view-dependent effects
	float exposure;                     // HDR exposure control
	float gamma;                        // Gamma correction value (typically 2.2)
	float prefilteredCubeMipLevels;     // IBL prefiltered environment map mip levels
	float scaleIBLAmbient;              // IBL ambient contribution scale

};

class Material {
public:
	explicit Material(std::string name) : name(std::move(name)) {
	}
	~Material() = default;

	[[nodiscard]] const std::string& GetName() const {
		return name;
	}

	// PBR properties (Metallic-Roughness default)
	glm::vec3 albedo = glm::vec3(1.0f);
	float metallic = 0.0;
	float roughness = 0.6;
	float ao = 1.0f;
	glm::vec3 emissive = glm::vec3(0.0f);
	float ior = 1.5f; // Index of refraction
	float emissiveStrength = 1.0f; // KHR_materials_emissive_strength extension
	float alpha = 1.0f; // Base color alpha (from MR baseColorFactor or SpecGloss diffuseFactor)
	float transmissionFactor = 0.0f; // KHR_materials_transmission: 0=opaque, 1=fully transmissive

	// Specular-Glossiness workflow (KHR_materials_pbrSpecularGlossiness)
	bool useSpecularGlossiness = false;
	glm::vec3 specularFactor = glm::vec3(0.04f);
	float glossinessFactor = 1.0f;
	std::string specGlossTexturePath; // Stored separately; also mirrored to metallicRoughnessTexturePath for binding 2

	// Alpha handling (glTF alphaMode and cutoff)
	std::string alphaMode = "OPAQUE"; // "OPAQUE", "MASK", or "BLEND"
	float alphaCutoff = 0.5f; // Used when alphaMode == MASK

	// Texture paths for PBR materials
	std::string albedoTexturePath;
	std::string normalTexturePath;
	std::string metallicRoughnessTexturePath;
	std::string occlusionTexturePath;
	std::string emissiveTexturePath;

	// Hint used by the renderer to select a specialized glass rendering path
	// for architectural glass (windows, lamp glass, etc.). Set by ModelLoader
	// based on material name/properties; defaults to false so non-glass
	// materials continue to use the generic PBR path.
	bool isGlass = false;

	// Hint used by the renderer to preferentially render inner liquid volumes
	// before outer glass shells (e.g., beer/wine in bar glasses). Set by
	// ModelLoader based on material name/properties; defaults to false.
	bool isLiquid = false;

	glm::vec4 baseColorFactor = glm::vec4(1.0f);
	float metallicFactor = 0.0f;
	float roughnessFactor = 1.0f;
	float baseColorTextureIndex = -1.0f;
	float metallicRoughnessTextureIndex = -1.0f;
	float normalTextureIndex = -1.0f;
	float occlusionTextureIndex = -1.0f;
	float emissiveTextureIndex = -1.0f;

private:
	std::string name;
};

// Define a mesh structure to track individual primitives/submeshes
struct Mesh {
	uint32_t firstIndex = 0;      // Starting index in the index buffer
	uint32_t indexCount = 0;      // Number of indices for this mesh
	uint32_t materialIndex = 0;   // Index into materials array (if you have multiple materials)
};

// Define a structure to hold per-object data
struct GameObject {

	// Transform properties
	glm::vec3 position = { 0.0f, 5.25f, 0.0f };
	glm::vec3 rotation = { 0.0f, 0.0f, 0.0f };
	glm::vec3 scale = { 1.0f, 1.0f, 1.0f };

	// Uniform buffer for this object (one per frame in flight)
	std::vector<vk::raii::Buffer> uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	vk::raii::Buffer vertexBuffer = { {} };
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	void* vertexBufferMapped = nullptr;

	vk::raii::Buffer indexBuffer = { {} };
	vk::raii::DeviceMemory indexBufferMemory = nullptr;
	void* indexBufferMapped = nullptr;

	// PBR textures - 5 texture slots for complete PBR rendering
	struct PBRTextures {
		vk::raii::Image        baseColorImage = nullptr;
		vk::raii::DeviceMemory baseColorMemory = nullptr;
		vk::raii::ImageView    baseColorView = nullptr;

		vk::raii::Image        metallicRoughnessImage = nullptr;
		vk::raii::DeviceMemory metallicRoughnessMemory = nullptr;
		vk::raii::ImageView    metallicRoughnessView = nullptr;

		vk::raii::Image        normalImage = nullptr;
		vk::raii::DeviceMemory normalMemory = nullptr;
		vk::raii::ImageView    normalView = nullptr;

		vk::raii::Image        occlusionImage = nullptr;
		vk::raii::DeviceMemory occlusionMemory = nullptr;
		vk::raii::ImageView    occlusionView = nullptr;

		vk::raii::Image        emissiveImage = nullptr;
		vk::raii::DeviceMemory emissiveMemory = nullptr;
		vk::raii::ImageView    emissiveView = nullptr;
  };

    std::vector<Material> materials;
	std::vector<PBRTextures> materialTextures;
	std::vector<std::vector<vk::raii::DescriptorSet>> materialDescriptorSets;

	std::vector<Vertex>    vertices;
	std::vector<uint32_t>  indices;

	// Mesh information for submeshes
	std::vector<Mesh>      meshes;

	// Calculate model matrix based on position, rotation, and scale
	glm::mat4 getModelMatrix() const {
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, position);
		model = glm::rotate(model, rotation.x, glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(model, rotation.y, glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, rotation.z, glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::scale(model, scale);
		return model;
	}
};

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

	HelloTriangleApplication() = default;
	~HelloTriangleApplication() = default;

private:
	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createDescriptorSetLayout();
		createGraphicsPipeline();

		if (!createPBRPipeline()) {
			std::cerr << "Failed to create PBR pipeline" << std::endl;
		}

		createCommandPool();
		createDepthResources();
		createTextureSampler();
		createDefaultTextures();
		setupGameObjects();
		createVertexBuffer(gameObjects[0]);
		createVertexBuffer(gameObjects[0]);
		createVertexBuffer(gameObjects[1]);
		createVertexBuffer(gameObjects[2]);
		createIndexBuffer(gameObjects[0]);
		createIndexBuffer(gameObjects[1]);
		createIndexBuffer(gameObjects[2]);
		createUniformBuffers();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers();
		createSyncObjects();
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Novus Engine", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		camera.setupInputCallbacks(window);
		camera.getViewMatrix();
		camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT);
	}

	void mainLoop() {
		lastFrameTime = 0.0f;
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			static auto startTime = std::chrono::high_resolution_clock::now();
			auto currentTime = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration<float>(currentTime - startTime).count();
			float deltaTime = time - lastFrameTime;
			lastFrameTime = time;
			camera.processInput(window, camera, deltaTime);
			drawFrame();
		}

		device.waitIdle();
	}

	void cleanupSwapChain()
	{
		swapChainImageViews.clear();
		swapChain = nullptr;
	}

	void cleanup() {
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	// Initialize the game objects with different positions, rotations, and scales
	void setupGameObjects() {
		// Object 1 - Flight Helmet (Center)
		gameObjects[0].position = { -3.0f, 0.0f, 0.0f };
		gameObjects[0].rotation = { 0.0f, 0.0f, 0.0f };
		gameObjects[0].scale = { 3.0f, 3.0f, 3.0f };
		gameObjects[0].vertexBuffer = nullptr;
		gameObjects[0].indexBuffer = nullptr;
		loadModel("../models/FlightHelmet.gltf", gameObjects[0]);
        for (size_t i = 0; i < gameObjects[0].materials.size(); ++i) {
			loadPBRTextures(gameObjects[0].materials[i], gameObjects[0].materialTextures[i]);
		}

		// Object 2 - Damaged Helmet (Left)
		gameObjects[2].position = { 3.0f, 0.0f, 0.0f };
		gameObjects[1].rotation = { 0.0f, 0.0f, 0.0f };
		gameObjects[1].scale = { 0.75f, 0.75f, 0.75f };
		gameObjects[1].vertexBuffer = nullptr;
		gameObjects[1].indexBuffer = nullptr;
		loadModel("../models/DamagedHelmet.gltf", gameObjects[1]);
        for (size_t i = 0; i < gameObjects[1].materials.size(); ++i) {
			loadPBRTextures(gameObjects[1].materials[i], gameObjects[1].materialTextures[i]);
		}
		gameObjects[1].materials[0].metallicFactor = 1.0f; // Override metallic factor for a more pronounced metallic look

		// Object 3 - Flight Helmet (Right)
		gameObjects[1].position = { 0.0f, 0.0f, 0.0f };
		gameObjects[2].rotation = { 0.0f, glm::radians(-45.0f), 0.0f };
		gameObjects[2].scale = { 3.0f, 3.0f, 3.0f };
		gameObjects[2].vertexBuffer = nullptr;
		gameObjects[2].indexBuffer = nullptr;
		loadModel("../models/FlightHelmet.gltf", gameObjects[2]);
        for (size_t i = 0; i < gameObjects[2].materials.size(); ++i) {
			loadPBRTextures(gameObjects[2].materials[i], gameObjects[2].materialTextures[i]);
		}
	}

	void recreateSwapChain()
	{
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		device.waitIdle();

		cleanupSwapChain();

		createSwapChain();
		createImageViews();

		createDepthResources();
	}

	std::vector<const char*> getRequiredInstanceExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(vk::EXTDebugUtilsExtensionName);
		}

		return extensions;
	}

	void createInstance() {
		constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Novus Engine",
											  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
											  .pEngineName = "No Engine",
											  .engineVersion = VK_MAKE_VERSION(1, 0, 0),
											  .apiVersion = vk::ApiVersion14 };

		std::vector<char const*> requiredLayers;
		if (enableValidationLayers) {
			requiredLayers.assign(validationLayers.begin(), validationLayers.end());
		}

		auto layerProperties = context.enumerateInstanceLayerProperties();
		if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
			return std::ranges::none_of(layerProperties,
				[requiredLayer](auto const& layerProperty)
				{ return strcmp(layerProperty.layerName, requiredLayer) == 0; });
			}))
		{
			throw std::runtime_error("One or more required layers are not supported!");
		}

		auto requiredExtensions = getRequiredInstanceExtensions();

		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		auto unsupportedPropertyIt =
			std::ranges::find_if(requiredExtensions,
				[&extensionProperties](auto const& requiredExtension) {
					return std::ranges::none_of(extensionProperties,
						[requiredExtension](auto const& extensionProperty) { return strcmp(extensionProperty.extensionName, requiredExtension) == 0; });
				});
		if (unsupportedPropertyIt != requiredExtensions.end())
		{
			throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIt));
		}

		vk::InstanceCreateInfo createInfo{
			.pApplicationInfo = &appInfo,
			.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
			.ppEnabledLayerNames = requiredLayers.data(),
			.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
			.ppEnabledExtensionNames = requiredExtensions.data() };

		instance = vk::raii::Instance(context, createInfo);
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
			vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
		vk::DebugUtilsMessageTypeFlagsEXT     messageTypeFlags(
			vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
		vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{ .messageSeverity = severityFlags,
																			  .messageType = messageTypeFlags,
																			  .pfnUserCallback = &debugCallback };
		debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
	}

	void createSurface() {
		VkSurfaceKHR _surface;
		if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
			throw std::runtime_error("failed to create window surface!");
		}

		surface = vk::raii::SurfaceKHR(instance, _surface);
	}

	bool isDeviceSuitable(vk::raii::PhysicalDevice const& physicalDevice) {
		bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

		auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		bool supportsGraphics =
			std::ranges::any_of(queueFamilies, [](auto const& qfp) { return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });


		auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();
		bool supportsAllRequiredExtensions =
			std::ranges::all_of(requiredDeviceExtension,
				[&availableDeviceExtensions](auto const& requiredDeviceExtension)
				{
					return std::ranges::any_of(availableDeviceExtensions,
						[requiredDeviceExtension](auto const& availableDeviceExtension)
						{ return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0; });
				});

		auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
			vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
		bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
			features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
			features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
			features.template get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>().dynamicRenderingLocalRead;

		return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
	}

	void pickPhysicalDevice() {
		std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
		auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const& physicalDevice) { return isDeviceSuitable(physicalDevice);  });
		if (devIter == physicalDevices.end())
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}

		physicalDevice = *devIter;
	}

	void createLogicalDevice() {
		std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

		queueIndex = ~0;
		for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
		{
			if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) &&
				physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface))
			{
				queueIndex = qfpIndex;
				break;
			}
		}

		if (queueIndex == ~0)
		{
			throw std::runtime_error("Could not find a queue for graphics and present -> terminating");
		}


		vk::PhysicalDeviceFeatures deviceFeatures;

		vk::StructureChain<vk::PhysicalDeviceFeatures2,
			vk::PhysicalDeviceVulkan11Features,
			vk::PhysicalDeviceVulkan13Features,
			vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
			vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>
			featureChain = {
				{.features = {.samplerAnisotropy = true } },                                                          // vk::PhysicalDeviceFeatures2
				{.shaderDrawParameters = true},                              // vk::PhysicalDeviceVulkan11Features
				{.synchronization2 = true, .dynamicRendering = true},        // vk::PhysicalDeviceVulkan13Features
				{.extendedDynamicState = true},                              // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
				{.dynamicRenderingLocalRead = true}                              // vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR
		};

		float queuePriority = 0.5f;
		vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex,
														 .queueCount = 1,
														  .pQueuePriorities = &queuePriority };


		vk::DeviceCreateInfo deviceCreateInfo{
			.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
			.queueCreateInfoCount = 1,
			.pQueueCreateInfos = &deviceQueueCreateInfo,
			.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
			.ppEnabledExtensionNames = requiredDeviceExtension.data()
		};

		device = vk::raii::Device(physicalDevice, deviceCreateInfo);

		queue = vk::raii::Queue(device, queueIndex, 0);
	}

	vk::raii::ImageView createImageView(vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
	{
		vk::ImageViewCreateInfo viewInfo{
			.image = image,
			.viewType = vk::ImageViewType::e2D,
			.format = format,
			.subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1} };
		return vk::raii::ImageView(device, viewInfo);
	}

	void createImageViews()
	{
		assert(swapChainImageViews.empty());

		swapChainImageViews.reserve(swapChainImages.size());
		for (auto& image : swapChainImages)
		{
			swapChainImageViews.emplace_back(createImageView(image, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));
		}
	}

	vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats)
	{
		assert(!availableFormats.empty());

		const auto formatIt = std::ranges::find_if(
			availableFormats,
			[](const auto& format) { return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
			}
		);

		return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
	}

	vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes)
	{
		assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) { return presentMode == vk::PresentModeKHR::eFifo; }));
		return std::ranges::any_of(availablePresentModes,
			[](const vk::PresentModeKHR value) { return vk::PresentModeKHR::eMailbox == value; }) ?
			vk::PresentModeKHR::eMailbox :
			vk::PresentModeKHR::eFifo;
	}

	vk::Extent2D chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		return {
			std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
			std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
		};
	}

	uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
	{
		auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
		if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		{
			minImageCount = surfaceCapabilities.maxImageCount;
		}
		return minImageCount;
	}

	void createSwapChain() {
		auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);

		swapChainExtent = chooseSwapExtent(surfaceCapabilities);
		uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

		std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
		std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
		swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

		vk::SwapchainCreateInfoKHR swapChainCreateInfo{
			.surface = *surface,
			.minImageCount = minImageCount,
			.imageFormat = swapChainSurfaceFormat.format,
			.imageColorSpace = swapChainSurfaceFormat.colorSpace,
			.imageExtent = swapChainExtent,
			.imageArrayLayers = 1,
			.imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
			.imageSharingMode = vk::SharingMode::eExclusive,
			.preTransform = surfaceCapabilities.currentTransform,
			.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
			.presentMode = chooseSwapPresentMode(availablePresentModes),
			.clipped = true,
			.oldSwapchain = nullptr
		};

		swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
		swapChainImages = swapChain.getImages();
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}

		std::vector<char> buffer(file.tellg());

		file.seekg(0, std::ios::beg);
		file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

		file.close();

		return buffer;
	}

	[[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const
	{
		vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size() * sizeof(char), .pCode = reinterpret_cast<const uint32_t*>(code.data()) };

		vk::raii::ShaderModule shaderModule{ device, createInfo };

		return shaderModule;
	}

	void createDescriptorSetLayout()
	{
		// Create descriptor set layout for PBR rendering
		// Binding 0: Uniform buffer (vertex shader)
		// Binding 1: Base color texture (fragment shader)
		// Binding 2: Metallic-roughness texture (fragment shader)
		// Binding 3: Normal map (fragment shader)
		// Binding 4: Occlusion map (fragment shader)
		// Binding 5: Emissive map (fragment shader)
		std::array<vk::DescriptorSetLayoutBinding, 6> bindings{
			{{.binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
			 {.binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
			 {.binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
			 {.binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
			 {.binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment},
			 {.binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment}}
		};

		vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data() };
		descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

	}

	void createGraphicsPipeline()
	{
		auto shaderCode = readFile("../shaders/slang.spv");

		vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

		vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule,  .pName = "vertMain" };
		vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule,  .pName = "fragMain" };

		vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

		vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()), .pDynamicStates = dynamicStates.data() };

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ .vertexBindingDescriptionCount = 1,
															   .pVertexBindingDescriptions = &bindingDescription,
															   .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
															   .pVertexAttributeDescriptions = attributeDescriptions.data()
		};

		vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList };

		vk::Viewport viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f };

		vk::Rect2D scissor{ vk::Offset2D{ 0, 0 }, swapChainExtent };

		vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };

		vk::PipelineRasterizationStateCreateInfo rasterizer{ .depthClampEnable = vk::False,
													.rasterizerDiscardEnable = vk::False,
													.polygonMode = vk::PolygonMode::eFill,
													.cullMode = vk::CullModeFlagBits::eBack,
													.frontFace = vk::FrontFace::eCounterClockwise,
													.depthBiasEnable = vk::False,
													.lineWidth = 1.0f };

		vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

		vk::PipelineDepthStencilStateCreateInfo depthStencil{
			.depthTestEnable = vk::True,
			.depthWriteEnable = vk::True,
			.depthCompareOp = vk::CompareOp::eLess,
			.depthBoundsTestEnable = vk::False,
			.stencilTestEnable = vk::False };

		vk::PipelineColorBlendAttachmentState colorBlendAttachment{
			.blendEnable = vk::True,
			.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
			.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
			.colorBlendOp = vk::BlendOp::eAdd,
			.srcAlphaBlendFactor = vk::BlendFactor::eOne,
			.dstAlphaBlendFactor = vk::BlendFactor::eZero,
			.alphaBlendOp = vk::BlendOp::eAdd,
			.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

		vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0 };

		pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

		vk::Format depthFormat = findDepthFormat();

		vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
			{.stageCount = 2,
			 .pStages = shaderStages,
			 .pVertexInputState = &vertexInputInfo,
			 .pInputAssemblyState = &inputAssembly,
			 .pViewportState = &viewportState,
			 .pRasterizationState = &rasterizer,
			 .pMultisampleState = &multisampling,
			 .pDepthStencilState = &depthStencil,
			 .pColorBlendState = &colorBlending,
			 .pDynamicState = &dynamicState,
			 .layout = pipelineLayout,
			 .renderPass = nullptr},
			{.colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format, .depthAttachmentFormat = depthFormat} };

		graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
	}

	void transition_image_layout(
		vk::Image               image,
		vk::ImageLayout         old_layout,
		vk::ImageLayout         new_layout,
		vk::AccessFlags2        src_access_mask,
		vk::AccessFlags2        dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask,
		vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags    image_aspect_flags)
	{
		vk::ImageMemoryBarrier2 barrier = {
			.srcStageMask = src_stage_mask,
			.srcAccessMask = src_access_mask,
			.dstStageMask = dst_stage_mask,
			.dstAccessMask = dst_access_mask,
			.oldLayout = old_layout,
			.newLayout = new_layout,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = image,
			.subresourceRange = {
				   .aspectMask = image_aspect_flags,
				   .baseMipLevel = 0,
				   .levelCount = 1,
				   .baseArrayLayer = 0,
				   .layerCount = 1} };
		vk::DependencyInfo dependency_info = {
			.dependencyFlags = {},
			.imageMemoryBarrierCount = 1,
			.pImageMemoryBarriers = &barrier };
		commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
	}

	void createCommandPool()
	{
		vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queueIndex };

		commandPool = vk::raii::CommandPool(device, poolInfo);
	}

	vk::raii::CommandBuffer beginSingleTimeCommands()
	{
		vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
		vk::raii::CommandBuffer       commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

		vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
		commandBuffer.begin(beginInfo);

		return std::move(commandBuffer);
	}

	void endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer)
	{
		commandBuffer.end();

		vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
		queue.submit(submitInfo, nullptr);
		queue.waitIdle();
	}

	void createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory)
	{
		vk::ImageCreateInfo imageInfo{
			.imageType = vk::ImageType::e2D,
			.format = format,
			.extent = {width, height, 1},
			.mipLevels = 1,
			.arrayLayers = 1,
			.samples = vk::SampleCountFlagBits::e1,
			.tiling = tiling,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive,
			.initialLayout = vk::ImageLayout::eUndefined };
		image = vk::raii::Image(device, imageInfo);

		vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
		imageMemory = vk::raii::DeviceMemory(device, allocInfo);
		image.bindMemory(imageMemory, 0);
	}

	vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features)
	{
		for (const auto format : candidates)
		{
			vk::FormatProperties props = physicalDevice.getFormatProperties(format);

			if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}
		throw std::runtime_error("failed to find supported format!");
	}

	vk::Format findDepthFormat()
	{
		return findSupportedFormat(
			{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
			vk::ImageTiling::eOptimal,
			vk::FormatFeatureFlagBits::eDepthStencilAttachment
		);
	}

	bool hasStencilComponent(vk::Format format)
	{
		return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
	}

	void createDepthResources()
	{
		vk::Format depthFormat = findDepthFormat();
		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
	}

	// Load a single texture from file path
	void loadTextureFromFile(const std::string& filepath, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory, vk::raii::ImageView& imageView, bool isSRGB = true) {
		// Check if file exists
		std::ifstream file(filepath);
		if (!file.good()) {
			std::cout << "Warning: Texture file not found: " << filepath << " - using placeholder" << std::endl;
			return;
		}
		file.close();

		// Determine file extension
		std::string extension = filepath.substr(filepath.find_last_of('.') + 1);
		std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

		uint32_t texWidth, texHeight;
		vk::DeviceSize imageSize;
		unsigned char* textureData = nullptr;
		int texChannels;

		// Load based on file extension
		if (extension == "ktx") {
			ktxTexture* kTexture;
			KTX_error_code result = ktxTexture_CreateFromNamedFile(
				filepath.c_str(),
				KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
				&kTexture);

			if (result != KTX_SUCCESS) {
				std::cout << "Warning: Failed to load KTX texture: " << filepath << std::endl;
				return;
			}

			texWidth = kTexture->baseWidth;
			texHeight = kTexture->baseHeight;
			imageSize = ktxTexture_GetImageSize(kTexture, 0);
			ktx_uint8_t* ktxTextureData = ktxTexture_GetData(kTexture);

			// Copy KTX data to temporary buffer
			textureData = new unsigned char[imageSize];
			memcpy(textureData, ktxTextureData, imageSize);

			ktxTexture_Destroy(kTexture);
		}
		else if (extension == "png" || extension == "jpg" || extension == "jpeg") {
			// Use stb_image for PNG/JPG
			int texWidth_i, texHeight_i;
			textureData = stbi_load(filepath.c_str(), &texWidth_i, &texHeight_i, &texChannels, STBI_rgb_alpha);

			if (!textureData) {
				std::cout << "Warning: Failed to load image texture: " << filepath << std::endl;
				return;
			}

			texWidth = static_cast<uint32_t>(texWidth_i);
			texHeight = static_cast<uint32_t>(texHeight_i);
			imageSize = texWidth * texHeight * 4; // RGBA
		}
		else {
			std::cout << "Warning: Unsupported texture format: " << extension << " for file: " << filepath << std::endl;
			return;
		}

		// Create staging buffer
		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, textureData, imageSize);
		stagingBufferMemory.unmapMemory();

		// Free texture data
		if (extension == "ktx") {
			delete[] textureData;
		}
		else {
			stbi_image_free(textureData);
		}

		// Determine format - normal maps and data textures should be linear, color should be sRGB
		vk::Format textureFormat = isSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;

		createImage(texWidth, texHeight, textureFormat, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, image, imageMemory);

		vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
		transitionImageLayout(commandBuffer, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		copyBufferToImage(commandBuffer, stagingBuffer, image, texWidth, texHeight);
		transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		endSingleTimeCommands(std::move(commandBuffer));

		// Create image view
		imageView = createImageView(*image, textureFormat, vk::ImageAspectFlagBits::eColor);

		std::cout << "Successfully loaded texture: " << filepath << " (" << texWidth << "x" << texHeight << ")" << std::endl;
	}

   // Load all PBR textures for a material
	void loadPBRTextures(const Material& material, GameObject::PBRTextures& textures) {
		std::cout << "Loading PBR textures for material: " << material.GetName() << std::endl;

		// Base color / albedo (sRGB)
      if (!material.albedoTexturePath.empty()) {
			loadTextureFromFile(material.albedoTexturePath,
				textures.baseColorImage,
				textures.baseColorMemory,
				textures.baseColorView,
				true);
		}

		// Metallic-roughness (linear)
       if (!material.metallicRoughnessTexturePath.empty()) {
			loadTextureFromFile(material.metallicRoughnessTexturePath,
				textures.metallicRoughnessImage,
				textures.metallicRoughnessMemory,
				textures.metallicRoughnessView,
				false);
		}

		// Normal map (linear)
      if (!material.normalTexturePath.empty()) {
			loadTextureFromFile(material.normalTexturePath,
				textures.normalImage,
				textures.normalMemory,
				textures.normalView,
				false);
		}

		// Occlusion (linear)
       if (!material.occlusionTexturePath.empty()) {
			loadTextureFromFile(material.occlusionTexturePath,
				textures.occlusionImage,
				textures.occlusionMemory,
				textures.occlusionView,
				false);
		}

		// Emissive (sRGB)
        if (!material.emissiveTexturePath.empty()) {
			loadTextureFromFile(material.emissiveTexturePath,
				textures.emissiveImage,
				textures.emissiveMemory,
				textures.emissiveView,
				true);
		}
	}

	// Create default fallback textures for missing PBR slots
	void createDefaultTextures() {
		// Create 1x1 white texture for base color/metallic-roughness/occlusion/emissive fallback
		{
          const uint32_t white = 0xFFFFFFFF;
			vk::DeviceSize imageSize = sizeof(uint32_t);

			vk::raii::Buffer stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				stagingBuffer, stagingBufferMemory);

			void* data = stagingBufferMemory.mapMemory(0, imageSize);
			memcpy(data, &white, imageSize);
			stagingBufferMemory.unmapMemory();

			createImage(1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, defaultTextureImage, defaultTextureMemory);

			vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
			transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
			copyBufferToImage(commandBuffer, stagingBuffer, defaultTextureImage, 1, 1);
			transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
			endSingleTimeCommands(std::move(commandBuffer));

			defaultTextureView = createImageView(*defaultTextureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
		}

		// Create 1x1 normal map with (0.5, 0.5, 1.0, 1.0) (flat normal pointing up in tangent space)
		{
			const uint32_t flatNormal = 0xFFFF7F7F; // RGBA: 127, 127, 255, 255 -> (0, 0, 1) in normal space
			vk::DeviceSize imageSize = sizeof(uint32_t);

			vk::raii::Buffer stagingBuffer({});
			vk::raii::DeviceMemory stagingBufferMemory({});
			createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				stagingBuffer, stagingBufferMemory);

			void* data = stagingBufferMemory.mapMemory(0, imageSize);
			memcpy(data, &flatNormal, imageSize);
			stagingBufferMemory.unmapMemory();

			createImage(1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				vk::MemoryPropertyFlagBits::eDeviceLocal, defaultNormalImage, defaultNormalMemory);

			vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
			transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
			copyBufferToImage(commandBuffer, stagingBuffer, defaultNormalImage, 1, 1);
			transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
			endSingleTimeCommands(std::move(commandBuffer));

			defaultNormalView = createImageView(*defaultNormalImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
		}
	}

	void createTextureSampler()
	{
		vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
		vk::SamplerCreateInfo        samplerInfo{ .magFilter = vk::Filter::eLinear,
												 .minFilter = vk::Filter::eLinear,
												 .mipmapMode = vk::SamplerMipmapMode::eLinear,
												 .addressModeU = vk::SamplerAddressMode::eRepeat,
												 .addressModeV = vk::SamplerAddressMode::eRepeat,
												 .addressModeW = vk::SamplerAddressMode::eRepeat,
												 .anisotropyEnable = vk::True,
												 .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
												 .compareEnable = vk::False,
												 .compareOp = vk::CompareOp::eAlways,
												 .borderColor = vk::BorderColor::eIntOpaqueBlack,
												 .unnormalizedCoordinates = vk::False
		};

		textureSampler = vk::raii::Sampler(device, samplerInfo);
	}

	void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory)
	{
		vk::BufferCreateInfo bufferInfo{
			.size = size,
			.usage = usage,
			.sharingMode = vk::SharingMode::eExclusive };
		buffer = vk::raii::Buffer(device, bufferInfo);
		vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
		vk::MemoryAllocateInfo allocInfo{
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
		bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
		buffer.bindMemory(*bufferMemory, 0);
	}

	void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
	{
		vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
		commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{ .size = size });
		endSingleTimeCommands(std::move(commandCopyBuffer));
	}

	void loadModel(std::string modelFilename, GameObject& gameObj)
	{
		tinygltf::Model    model;
		tinygltf::TinyGLTF loader;
		std::string        err;
		std::string        warn;

		bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelFilename);

		if (!warn.empty())
		{
			std::cout << "glTF warning: " << warn << std::endl;
		}

		if (!err.empty())
		{
			std::cout << "glTF error: " << err << std::endl;
		}

		if (!ret)
		{
			throw std::runtime_error("Failed to load glTF model");
		}

		gameObj.vertices.clear();
		gameObj.indices.clear();
		gameObj.meshes.clear();
		gameObj.materials.clear();
		gameObj.materialTextures.clear();
		gameObj.materialDescriptorSets.clear();

		// Extract base directory from model filename for texture loading
		std::string baseDir = modelFilename.substr(0, modelFilename.find_last_of("/\\") + 1);

       for (const auto& mat : model.materials) {
			Material material(mat.name.empty() ? "Material" : mat.name);

			if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4) {
				material.baseColorFactor = glm::vec4(
					mat.pbrMetallicRoughness.baseColorFactor[0],
					mat.pbrMetallicRoughness.baseColorFactor[1],
					mat.pbrMetallicRoughness.baseColorFactor[2],
					mat.pbrMetallicRoughness.baseColorFactor[3]
				);
			}

			material.metallicFactor = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
			material.roughnessFactor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

			if (mat.pbrMetallicRoughness.baseColorTexture.index >= 0) {
				int texIndex = mat.pbrMetallicRoughness.baseColorTexture.index;
				if (texIndex < static_cast<int>(model.textures.size())) {
					int imageIndex = model.textures[texIndex].source;
					if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size())) {
						material.albedoTexturePath = baseDir + model.images[imageIndex].uri;
						material.baseColorTextureIndex = 0.0f;
					}
				}
			}

			if (mat.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0) {
				int texIndex = mat.pbrMetallicRoughness.metallicRoughnessTexture.index;
				if (texIndex < static_cast<int>(model.textures.size())) {
					int imageIndex = model.textures[texIndex].source;
					if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size())) {
						material.metallicRoughnessTexturePath = baseDir + model.images[imageIndex].uri;
						material.metallicRoughnessTextureIndex = 0.0f;
					}
				}
			}

			if (mat.normalTexture.index >= 0) {
				int texIndex = mat.normalTexture.index;
				if (texIndex < static_cast<int>(model.textures.size())) {
					int imageIndex = model.textures[texIndex].source;
					if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size())) {
						material.normalTexturePath = baseDir + model.images[imageIndex].uri;
						material.normalTextureIndex = 0.0f;
					}
				}
			}

			if (mat.occlusionTexture.index >= 0) {
				int texIndex = mat.occlusionTexture.index;
				if (texIndex < static_cast<int>(model.textures.size())) {
					int imageIndex = model.textures[texIndex].source;
					if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size())) {
						material.occlusionTexturePath = baseDir + model.images[imageIndex].uri;
						material.occlusionTextureIndex = 0.0f;
					}
				}
			}

			if (mat.emissiveTexture.index >= 0) {
				int texIndex = mat.emissiveTexture.index;
				if (texIndex < static_cast<int>(model.textures.size())) {
					int imageIndex = model.textures[texIndex].source;
					if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size())) {
						material.emissiveTexturePath = baseDir + model.images[imageIndex].uri;
						material.emissiveTextureIndex = 0.0f;
					}
				}
			}

			std::cout << "Loaded material: " << material.GetName() << std::endl;
			std::cout << "  Base color texture: " << material.albedoTexturePath << std::endl;
			std::cout << "  Metallic-roughness texture: " << material.metallicRoughnessTexturePath << std::endl;
			std::cout << "  Normal texture: " << material.normalTexturePath << std::endl;

			gameObj.materials.push_back(std::move(material));
			gameObj.materialTextures.emplace_back();
		}

		if (gameObj.materials.empty()) {
			gameObj.materials.emplace_back("DefaultMaterial");
			gameObj.materialTextures.emplace_back();
		}

		// Process all meshes in the model (geometry loading - unchanged)
		for (const auto& mesh : model.meshes)
		{
			for (const auto& primitive : mesh.primitives)
			{
				// Track the starting point for this mesh
				Mesh meshInfo;
				meshInfo.firstIndex = static_cast<uint32_t>(gameObj.indices.size());
				meshInfo.materialIndex = primitive.material >= 0 ? primitive.material : 0;

				// Get indices
				const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
				const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
				const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];

				// Get vertex positions
				const tinygltf::Accessor& posAccessor = model.accessors[primitive.attributes.at("POSITION")];
				const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
				const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];

				bool                        hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
				const tinygltf::Accessor* normalAccessor = nullptr;
				const tinygltf::BufferView* normalBufferView = nullptr;
				const tinygltf::Buffer* normalBuffer = nullptr;

				// Get texture coordinates if available
				bool                        hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
				const tinygltf::Accessor* texCoordAccessor = nullptr;
				const tinygltf::BufferView* texCoordBufferView = nullptr;
				const tinygltf::Buffer* texCoordBuffer = nullptr;

				bool hasTangents = primitive.attributes.find("TANGENT") != primitive.attributes.end();
				const tinygltf::Accessor* tangentAccessor = nullptr;
				const tinygltf::BufferView* tangentBufferView = nullptr;
				const tinygltf::Buffer* tangentBuffer = nullptr;

				if (hasTexCoords)
				{
					texCoordAccessor = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
					texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
					texCoordBuffer = &model.buffers[texCoordBufferView->buffer];
				}

				if (hasNormals)
				{
					normalAccessor = &model.accessors[primitive.attributes.at("NORMAL")];
					normalBufferView = &model.bufferViews[normalAccessor->bufferView];
					normalBuffer = &model.buffers[normalBufferView->buffer];
				}

				if (hasTangents)
				{
					tangentAccessor = &model.accessors[primitive.attributes.at("TANGENT")];
					tangentBufferView = &model.bufferViews[tangentAccessor->bufferView];
					tangentBuffer = &model.buffers[tangentBufferView->buffer];
				}

				uint32_t baseVertex = static_cast<uint32_t>(gameObj.vertices.size());

				for (size_t i = 0; i < posAccessor.count; i++)
				{
					Vertex vertex{};

					const float* pos = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * 12]);
					vertex.pos = { pos[0], -pos[1], pos[2] };

					if (hasTexCoords)
					{
						const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * 8]);
						vertex.texCoord = { texCoord[0], texCoord[1] };
					}
					else
					{
						vertex.texCoord = { 0.0f, 0.0f };
					}

					if (hasNormals)
					{
						const float* normal = reinterpret_cast<const float*>(&normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * 12]);
						vertex.normal = { normal[0], -normal[1], normal[2] };

					}
					else
					{
						vertex.normal = { 0.0f, 0.0f, 0.0f };
					}

					if (hasTangents)
					{
						const float* tangent = reinterpret_cast<const float*>(&tangentBuffer->data[tangentBufferView->byteOffset + tangentAccessor->byteOffset + i * 16]);
						vertex.tangent = { tangent[0], -tangent[1], tangent[2], tangent[3] };
					}
					else
					{
						vertex.tangent = { 0.0f, 0.0f, 0.0f, 1.0f };
					}

					gameObj.vertices.push_back(vertex);
				}

				const unsigned char* indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
				size_t               indexCount = indexAccessor.count;
				size_t               indexStride = 0;

				// Determine index stride based on component type
				if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
				{
					indexStride = sizeof(uint16_t);
				}
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
				{
					indexStride = sizeof(uint32_t);
				}
				else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
				{
					indexStride = sizeof(uint8_t);
				}
				else
				{
					throw std::runtime_error("Unsupported index component type");
				}

				gameObj.indices.reserve(gameObj.indices.size() + indexCount);

				for (size_t i = 0; i < indexCount; i++)
				{
					uint32_t index = 0;

					if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
					{
						index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
					}
					else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
					{
						index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
					}
					else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
					{
						index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);
					}

					gameObj.indices.push_back(baseVertex + index);
				}

				// Record mesh information
				meshInfo.indexCount = static_cast<uint32_t>(gameObj.indices.size() - meshInfo.firstIndex);
				gameObj.meshes.push_back(meshInfo);
			}
		}

	}

	void createVertexBuffer(GameObject& gameObj)
	{
		vk::DeviceSize bufferSize = sizeof(gameObj.vertices[0]) * gameObj.vertices.size();

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, gameObj.vertices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, gameObj.vertexBuffer, gameObj.vertexBufferMemory);

		copyBuffer(stagingBuffer, gameObj.vertexBuffer, bufferSize);
	}

	void createIndexBuffer(GameObject& gameObj)
	{
		vk::DeviceSize bufferSize = sizeof(gameObj.indices[0]) * gameObj.indices.size();

		vk::raii::Buffer stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);

		void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
		memcpy(dataStaging, gameObj.indices.data(), bufferSize);
		stagingBufferMemory.unmapMemory();

		createBuffer(bufferSize, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, gameObj.indexBuffer, gameObj.indexBufferMemory);
		copyBuffer(stagingBuffer, gameObj.indexBuffer, bufferSize);
	}

	void createUniformBuffers()
	{
		// For each game object
		for (auto& gameObject : gameObjects) {
			gameObject.uniformBuffers.clear();
			gameObject.uniformBuffersMemory.clear();
			gameObject.uniformBuffersMapped.clear();

			// Create uniform buffers for each frame in flight
			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
				vk::raii::Buffer buffer({});
				vk::raii::DeviceMemory bufferMem({});
				createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
					vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					buffer, bufferMem);
				gameObject.uniformBuffers.emplace_back(std::move(buffer));
				gameObject.uniformBuffersMemory.emplace_back(std::move(bufferMem));
				gameObject.uniformBuffersMapped.emplace_back(gameObject.uniformBuffersMemory[i].mapMemory(0, bufferSize));
			}
		}
	}

	void updateUniformBuffers() {
		// Camera and projection matrices (shared by all objects)
		glm::mat4 view = camera.getViewMatrix();
		glm::mat4 proj = camera.getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 2000.0f);
		//	proj[1][1] *= -1; // Flip Y for Vulkan

			// Update uniform buffers for each object
		for (auto& gameObject : gameObjects) {
			// Apply continuous rotation to the object
		//	gameObject.rotation.y += 0.001f; // Slow rotation around Y axis

			// Get the model matrix for this object
			glm::mat4 initialRotation = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 model = gameObject.getModelMatrix();

			// Create and update the UBO
			UniformBufferObject ubo{
				.model = model,
				.view = view,
				.proj = proj
			};

			// Copy the UBO data to the mapped memory
			memcpy(gameObject.uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
		}
	}

	void createDescriptorPool()
	{
        uint32_t materialCount = 0;
		for (const auto& gameObject : gameObjects)
		{
			materialCount += static_cast<uint32_t>(std::max<size_t>(1, gameObject.materials.size()));
		}

		std::array<vk::DescriptorPoolSize, 2> poolSize{ {{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = materialCount * MAX_FRAMES_IN_FLIGHT},
														{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 5 * materialCount * MAX_FRAMES_IN_FLIGHT}} };
		vk::DescriptorPoolCreateInfo          poolInfo{ .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                              .maxSets = materialCount * MAX_FRAMES_IN_FLIGHT,
													   .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
													   .pPoolSizes = poolSize.data() };

		descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
	}

   void createDescriptorSets() {
		for (auto& gameObject : gameObjects) {
			gameObject.materialDescriptorSets.clear();
			gameObject.materialDescriptorSets.resize(gameObject.materials.size());

			for (size_t materialIndex = 0; materialIndex < gameObject.materials.size(); ++materialIndex) {
				auto& materialTextures = gameObject.materialTextures[materialIndex];
				auto& descriptorSetsForMaterial = gameObject.materialDescriptorSets[materialIndex];

				std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
				vk::DescriptorSetAllocateInfo allocInfo{
					.descriptorPool = *descriptorPool,
					.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
					.pSetLayouts = layouts.data()
				};

				descriptorSetsForMaterial = device.allocateDescriptorSets(allocInfo);

				for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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

					vk::DescriptorImageInfo baseColorInfo{ .sampler = *textureSampler, .imageView = baseColorView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
					vk::DescriptorImageInfo metallicRoughnessInfo{ .sampler = *textureSampler, .imageView = metallicRoughnessView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
					vk::DescriptorImageInfo normalInfo{ .sampler = *textureSampler, .imageView = normalView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
					vk::DescriptorImageInfo occlusionInfo{ .sampler = *textureSampler, .imageView = occlusionView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
					vk::DescriptorImageInfo emissiveInfo{ .sampler = *textureSampler, .imageView = emissiveView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

					std::array descriptorWrites{
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &bufferInfo },
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &baseColorInfo },
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &metallicRoughnessInfo },
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &normalInfo },
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &occlusionInfo },
						vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &emissiveInfo }
					};

					device.updateDescriptorSets(descriptorWrites, {});
				}
			}
		}
	}
	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void createCommandBuffers()
	{
		vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
		commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
	}

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

	void copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height)
	{
		vk::BufferImageCopy region{ .bufferOffset = 0,
						   .bufferRowLength = 0,
						   .bufferImageHeight = 0,
						   .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
						   .imageOffset = {0, 0, 0},
						   .imageExtent = {width, height, 1} };
		commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
	}

	void pushMaterialProperties(vk::CommandBuffer commandBuffer, const GameObject* model, uint32_t materialIndex) {
		// Get material from the model
     const Material& material = model->materials[materialIndex < model->materials.size() ? materialIndex : 0];

		// Define push constants
		PushConstantBlock pushConstants{};
		pushConstants.baseColorFactor = material.baseColorFactor;
		pushConstants.metallicFactor = material.metallicFactor;
		pushConstants.roughnessFactor = material.roughnessFactor;
		pushConstants.baseColorTextureSet = material.baseColorTextureIndex;
		pushConstants.physicalDescriptorTextureSet = material.metallicRoughnessTextureIndex;
		pushConstants.normalTextureSet = material.normalTextureIndex;
		pushConstants.occlusionTextureSet = material.occlusionTextureIndex;
		pushConstants.emissiveTextureSet = material.emissiveTextureIndex;
		pushConstants.alphaMask = 0.0f;
		pushConstants.alphaMaskCutoff = material.alphaCutoff;

		// Push constants to shader using vk::raii
		commandBuffer.pushConstants(
			*pbrPipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			sizeof(PushConstantBlock),
			&pushConstants
		);
	}


	//void updateUniformBuffer(uint32_t currentFrame) {
	//	// Update uniform buffer for each game object
	//	for (auto& gameObject : gameObjects) {
	//		UniformBufferObject ubo{};

	//		// Update model matrix from game object transform
	//		ubo.model = gameObject.getModelMatrix();

	//		// Update view and projection matrices from camera
	//		ubo.view = camera.getViewMatrix();
	//		float aspectRatio = swapChainExtent.width / static_cast<float>(swapChainExtent.height);
	//		ubo.proj = camera.getProjectionMatrix(aspectRatio);

	//		// Set up lights (4 point lights around the scene)
	//		ubo.lightPositions[0] = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);
	//		ubo.lightPositions[1] = glm::vec4(-10.0f, 10.0f, 10.0f, 1.0f);
	//		ubo.lightPositions[2] = glm::vec4(10.0f, 10.0f, -10.0f, 1.0f);
	//		ubo.lightPositions[3] = glm::vec4(-10.0f, 10.0f, -10.0f, 1.0f);

	//		// Set light colors (white lights with different intensities)
	//		ubo.lightColors[0] = glm::vec4(0.0f, 0.0f, 300.0f, 1.0f);
	//		ubo.lightColors[1] = glm::vec4(300.0f, 0.0f, 0.0f, 1.0f);
	//		ubo.lightColors[2] = glm::vec4(0.0f, 300.0f, 0.0f, 1.0f);
	//		ubo.lightColors[3] = glm::vec4(300.0f, 300.0f, 300.0f, 1.0f);

	//		// Set camera position
	//		ubo.camPos = glm::vec4(camera.getPosition(), 1.0f);

	//		// Set HDR and gamma parameters
	//		ubo.exposure = 1.0f;
	//		ubo.gamma = 2.2f;
	//		ubo.prefilteredCubeMipLevels = 1.0f;
	//		ubo.scaleIBLAmbient = 1.0f;

	//		// Copy data to mapped uniform buffer
	//		memcpy(gameObject.uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	//	}
	//}

	void recordCommandBuffer(uint32_t imageIndex)
	{
		auto& commandBuffer = commandBuffers[frameIndex];
		commandBuffer.begin({});

		transition_image_layout(swapChainImages[imageIndex], vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
			vk::AccessFlags2{}, vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::ImageAspectFlagBits::eColor);

		transition_image_layout(
			*depthImage,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eDepthAttachmentOptimal,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::ImageAspectFlagBits::eDepth);


		vk::ClearValue              clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
		vk::ClearValue				clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

		vk::RenderingAttachmentInfo attachmentInfo = {
			.imageView = swapChainImageViews[imageIndex],
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eStore,
			.clearValue = clearColor };

		vk::RenderingAttachmentInfo depthAttachmentInfo = {
			.imageView = depthImageView,
			.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
			.loadOp = vk::AttachmentLoadOp::eClear,
			.storeOp = vk::AttachmentStoreOp::eDontCare,
			.clearValue = clearDepth };

		vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = {0, 0}, .extent = swapChainExtent},
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo };

		commandBuffer.beginRendering(renderingInfo);

		//	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);

		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));


		//// Draw each object with its own descriptor set
		//for (const auto& gameObject : gameObjects) {
		//	// Bind vertex and index buffers (shared by all objects)
		//	commandBuffers[frameIndex].bindVertexBuffers(0, *gameObject.vertexBuffer, { 0 });
		//	commandBuffers[frameIndex].bindIndexBuffer(*gameObject.indexBuffer, 0, vk::IndexType::eUint32);
		//	// Bind the descriptor set for this object
		//	commandBuffers[frameIndex].bindDescriptorSets(
		//		vk::PipelineBindPoint::eGraphics,
		//		*pipelineLayout,
		//		0,
		//		*gameObject.descriptorSets[frameIndex],
		//		nullptr
		//	);

		//	// Draw the object
		//	commandBuffers[frameIndex].drawIndexed(gameObject.indices.size(), 1, 0, 0, 0);
		//}


		// Bind the PBR pipeline

		// For each model in the scene
		for (auto& model : gameObjects) {
			// Bind vertex and index buffers once per model
			updateUniformBuffer(frameIndex, &model, &camera);
			vk::Buffer vertexBuffers[] = { model.vertexBuffer };
			vk::DeviceSize offsets[] = { 0 };
			commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
			commandBuffer.bindIndexBuffer(*model.indexBuffer, 0, vk::IndexType::eUint32);

			// For each mesh in the model
			for (const auto& mesh : model.meshes) {
				// Push material properties for this mesh
				pushMaterialProperties(commandBuffer, &model, mesh.materialIndex);
				uint32_t descriptorMaterialIndex = mesh.materialIndex < model.materialDescriptorSets.size() ? mesh.materialIndex : 0;

				// Bind descriptor sets
				commandBuffer.bindDescriptorSets(
					vk::PipelineBindPoint::eGraphics,
					*pbrPipelineLayout,
					0,
                  *model.materialDescriptorSets[descriptorMaterialIndex][frameIndex],
					nullptr
				);

				// Draw this specific mesh using its index range
				commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
			}
		}

		commandBuffer.endRendering();

		transition_image_layout(
			swapChainImages[imageIndex],
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::ePresentSrcKHR,
			vk::AccessFlagBits2::eColorAttachmentWrite,             // srcAccessMask
			{},                                                     // dstAccessMask
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,     // srcStage
			vk::PipelineStageFlagBits2::eBottomOfPipe,               // dstStage
			vk::ImageAspectFlagBits::eColor
		);

		commandBuffer.end();



	}

	void createSyncObjects()
	{
		assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
			inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
		}
	}

	void drawFrame()
	{
		auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
		if (fenceResult != vk::Result::eSuccess)
		{
			throw std::runtime_error("Failed to wait for draw fence!");
		}

		auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

		if (result == vk::Result::eErrorOutOfDateKHR)
		{
			recreateSwapChain();
			return;
		}
		if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
		{
			assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		device.resetFences(*inFlightFences[frameIndex]);

		commandBuffers[frameIndex].reset();
		recordCommandBuffer(imageIndex);

		queue.waitIdle();

		vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
		const vk::SubmitInfo   submitInfo{ .waitSemaphoreCount = 1,
										  .pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
										  .pWaitDstStageMask = &waitDestinationStageMask,
										  .commandBufferCount = 1,
										  .pCommandBuffers = &*commandBuffers[frameIndex],
										  .signalSemaphoreCount = 1,
										  .pSignalSemaphores = &*renderFinishedSemaphores[imageIndex] };

		queue.submit(submitInfo, *inFlightFences[frameIndex]);

		const vk::PresentInfoKHR presentInfoKHR{
			.waitSemaphoreCount = 1,
			.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
			.swapchainCount = 1,
			.pSwapchains = &*swapChain,
			.pImageIndices = &imageIndex };

		result = queue.presentKHR(presentInfoKHR);

		if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapChain();
		}
		else
		{
			assert(result == vk::Result::eSuccess);
		}

		frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*)
	{
		if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError || severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
		{
			std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
		}

		return vk::False;
	}

	bool createPBRPipeline() {
		try {
			// Load our compiled PBR shader from disk
			// The .spv file contains both vertex and fragment shader code compiled by slangc
			auto shaderCode = readFile("../shaders/pbr.spv");

			// Create a shader module - this is Vulkan's container for shader bytecode
			// The shader module acts as a wrapper around the SPIR-V bytecode that GPU drivers understand
			vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

			// Configure the vertex shader stage
			// This tells Vulkan which shader stage this module serves and its entry point function
			vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
			vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex)
				.setModule(*shaderModule)
				.setPName("vertMain");  // Must match the vertex shader function name

			// Configure the fragment shader stage
			// Same module, different entry point - this is how combined shaders work
			vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
			fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment)
				.setModule(*shaderModule)
				.setPName("fragMain");  // Must match the fragment shader function name

			std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

			// Configure how vertex data is structured and fed to the vertex shader
			vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

			// Use the same vertex binding and attribute descriptions as the Vertex struct
			auto bindingDescription = Vertex::getBindingDescription();
			auto attributeDescriptions = Vertex::getAttributeDescriptions();

			// Connect the binding and attribute descriptions to the vertex input state
			vertexInputInfo.setVertexBindingDescriptionCount(1)
				.setPVertexBindingDescriptions(&bindingDescription)
				.setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()))
				.setPVertexAttributeDescriptions(attributeDescriptions.data());

			// Configure input assembly - how vertices become triangles
			vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
			inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList)  // Every 3 vertices form a triangle
				.setPrimitiveRestartEnable(false);                    // Don't use primitive restart indices

			// Configure viewport and scissor state
			vk::PipelineViewportStateCreateInfo viewportState;
			viewportState.setViewportCount(1)       // Single viewport (most common case)
				.setScissorCount(1);        // Single scissor rectangle

			// Define which pipeline state can be changed dynamically
			// This improves performance by avoiding pipeline recreation for common changes
			std::vector<vk::DynamicState> dynamicStates = {
				vk::DynamicState::eViewport,        // Viewport can change (window resize, camera changes)
				vk::DynamicState::eScissor          // Scissor rectangle can change (UI clipping, effects)
			};

			vk::PipelineDynamicStateCreateInfo dynamicState;
			dynamicState.setDynamicStateCount(static_cast<uint32_t>(dynamicStates.size()))
				.setPDynamicStates(dynamicStates.data());

			// Configure rasterization - how triangles become pixels
			vk::PipelineRasterizationStateCreateInfo rasterizer;
			rasterizer.setDepthClampEnable(false)                           // Don't clamp depth values (standard behavior)
				.setRasterizerDiscardEnable(false)                     // Don't discard primitives before rasterization
				.setPolygonMode(vk::PolygonMode::eFill)                // Fill triangles (not wireframe or points)
				.setLineWidth(1.0f)                                    // Line width (only relevant for wireframe)
				.setCullMode(vk::CullModeFlagBits::eBack)              // Cull back-facing triangles
				.setFrontFace(vk::FrontFace::eCounterClockwise)        // Counter-clockwise vertices = front-facing
				.setDepthBiasEnable(false);                            // No depth bias (used for shadow mapping)

			// Configure multisampling - anti-aliasing settings
			vk::PipelineMultisampleStateCreateInfo multisampling;
			multisampling.setSampleShadingEnable(false)                     // Disable per-sample shading
				.setRasterizationSamples(vk::SampleCountFlagBits::e1); // No multisampling (1 sample per pixel)

			// Configure depth and stencil testing
			vk::PipelineDepthStencilStateCreateInfo depthStencil;
			depthStencil.setDepthTestEnable(true)                           // Enable depth testing for proper occlusion
				.setDepthWriteEnable(true)                           // Write depth values to depth buffer
				.setDepthCompareOp(vk::CompareOp::eLess)             // Fragment passes if its depth is less (closer)
				.setDepthBoundsTestEnable(false)                     // Don't use depth bounds testing
				.setStencilTestEnable(false);                        // Don't use stencil testing

			// Configure color blending - how new pixels combine with existing ones
			vk::PipelineColorBlendAttachmentState colorBlendAttachment;
			colorBlendAttachment.setColorWriteMask(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |     // Write all color channels
				vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
				.setBlendEnable(true)                                    // Enable alpha blending
				.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)      // New fragment's alpha
				.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)  // One minus new fragment's alpha
				.setColorBlendOp(vk::BlendOp::eAdd)                      // Add source and destination
				.setSrcAlphaBlendFactor(vk::BlendFactor::eOne)           // Preserve new alpha
				.setDstAlphaBlendFactor(vk::BlendFactor::eZero)          // Ignore old alpha
				.setAlphaBlendOp(vk::BlendOp::eAdd);                     // Add alpha values

			vk::PipelineColorBlendStateCreateInfo colorBlending;
			colorBlending.setLogicOpEnable(false)                                     // Don't use logical operations
				.setAttachmentCount(1)                                        // Single color attachment
				.setPAttachments(&colorBlendAttachment);

			// Configure push constants for fast material property updates
			vk::PushConstantRange pushConstantRange;
			pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eFragment)      // Only fragment shader uses these
				.setOffset(0)                                             // Start at beginning
				.setSize(sizeof(PushConstantBlock));                      // Size of our material data

			// Create the pipeline layout - defines resource organization
			vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
			pipelineLayoutInfo.setSetLayoutCount(1)                                  // Single descriptor set
				.setPSetLayouts(&*descriptorSetLayout)                  // Our texture/uniform bindings
				.setPushConstantRangeCount(1)                           // One push constant block
				.setPPushConstantRanges(&pushConstantRange);

			// Create the pipeline layout object
			pbrPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

			// Assemble the complete graphics pipeline
			vk::GraphicsPipelineCreateInfo pipelineInfo;
			pipelineInfo.setStageCount(static_cast<uint32_t>(shaderStages.size()))   // Number of shader stages
				.setPStages(shaderStages.data())                               // Shader stage configurations
				.setPVertexInputState(&vertexInputInfo)                        // Vertex format
				.setPInputAssemblyState(&inputAssembly)                        // Primitive topology
				.setPViewportState(&viewportState)                             // Viewport configuration
				.setPRasterizationState(&rasterizer)                           // Rasterization settings
				.setPMultisampleState(&multisampling)                          // Anti-aliasing settings
				.setPDepthStencilState(&depthStencil)                          // Depth/stencil testing
				.setPColorBlendState(&colorBlending)                           // Blending configuration
				.setPDynamicState(&dynamicState)                               // Dynamic state settings
				.setLayout(*pbrPipelineLayout)                                 // Resource layout
				.setRenderPass(nullptr)                                        // Using dynamic rendering
				.setSubpass(0)                                                 // Subpass index
				.setBasePipelineHandle(nullptr);                               // No base pipeline

			// Configure for dynamic rendering (modern Vulkan approach)
			vk::PipelineRenderingCreateInfo renderingInfo;
			renderingInfo.setColorAttachmentCount(1)                                 // Single color target
				.setPColorAttachmentFormats(&swapChainSurfaceFormat.format)           // Match swapchain format
				.setDepthAttachmentFormat(findDepthFormat());                // Depth buffer format
			pipelineInfo.setPNext(&renderingInfo);

			// Create the final graphics pipeline
			pbrPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);

			return true;
		}
		catch (const std::exception& e) {
			std::cerr << "Error creating PBR pipeline: " << e.what() << std::endl;
			return false;
		}
	}

	// Update uniform buffer
	void updateUniformBuffer(uint32_t currentFrame, GameObject* entity, Camera* camera) {
		// Get the transform component from the entity
	//	auto transform = entity->GetComponent<TransformComponent>();
		//if (!transform) {
		//	std::cerr << "Entity does not have a transform component" << std::endl;
		//	return;
		//}

		// Create the uniform buffer object
		UniformBufferObject ubo{};

		// Set the model matrix from the entity's transform
		ubo.model = glm::translate(glm::mat4(1.0f), entity->position);
		ubo.model = glm::rotate(ubo.model, glm::radians(entity->rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		ubo.model = glm::rotate(ubo.model, glm::radians(entity->rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.model = glm::rotate(ubo.model, glm::radians(entity->rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.model = glm::scale(ubo.model, glm::vec3(entity->scale));

		// Set the view and projection matrices from the camera
		if (camera) {
			ubo.view = camera->getViewMatrix();
			ubo.proj = camera->getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 2000.0f);
		}
		else {
			// Default view and projection matrices if no camera is provided
			ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
			ubo.proj[1][1] *= -1; // Flip Y coordinate for Vulkan
		}

        // Set up lights in a tighter studio arrangement around the actual scene bounds.
		// The objects span roughly x:[-5,1], so keep lights close enough to affect all of them.
		ubo.lightPositions[0] = glm::vec4(-1.5f, 3.5f, 3.0f, 1.0f);  // front key
		ubo.lightPositions[1] = glm::vec4(-4.5f, 2.5f, 2.0f, 1.0f);  // left fill
		ubo.lightPositions[2] = glm::vec4(1.5f, 2.0f, -2.5f, 1.0f);  // right rim
		ubo.lightPositions[3] = glm::vec4(-2.0f, 4.5f, -1.0f, 1.0f); // top back

		// Use balanced white lights first so material response is easier to judge.
		ubo.lightColors[0] = glm::vec4(2.0f, 0.0f, 0.0f, 1.0f);
		ubo.lightColors[1] = glm::vec4(0.0f, 0.0f, 4.0f, 1.0f);
		ubo.lightColors[2] = glm::vec4(0.0f, 4.0f, 0.0f, 1.0f);
		ubo.lightColors[3] = glm::vec4(6.0f, 0.1f, 0.0f, 1.0f);

		// Set camera position for view-dependent effects
		ubo.camPos = glm::vec4(camera ? camera->getPosition() : glm::vec3(2.0f, 2.0f, 2.0f), 1.0f);

		// Set PBR parameters - increased exposure for brighter rendering
		ubo.exposure = 1.0f;
		ubo.gamma = 2.2f;
		ubo.prefilteredCubeMipLevels = 1.0f;
      ubo.scaleIBLAmbient = 0.02f;

		// Copy the uniform buffer object to the device memory using vk::raii
		// With vk::raii, we can use the mapped memory directly
		memcpy(entity->uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

private:
	GLFWwindow* window;

	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice physicalDevice = nullptr;
	vk::raii::Device device = nullptr;
	vk::raii::Queue queue = nullptr;
	uint32_t queueIndex = ~0;

	vk::raii::SurfaceKHR surface = nullptr;
	vk::raii::SwapchainKHR swapChain = nullptr;
	std::vector<vk::Image> swapChainImages;
	vk::Extent2D swapChainExtent;
	vk::SurfaceFormatKHR swapChainSurfaceFormat;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	bool framebufferResized = false;

	vk::raii::DescriptorPool descriptorPool = nullptr;
	vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline graphicsPipeline = nullptr;

	vk::raii::CommandPool commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t frameIndex = 0;

	vk::raii::Buffer vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer       indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;

	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*>                 uniformBuffersMapped;

	vk::raii::Image        textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView = nullptr;
	vk::raii::Sampler      textureSampler = nullptr;

	vk::raii::Image depthImage = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView depthImageView = nullptr;

	// Default fallback textures
	vk::raii::Image defaultTextureImage = nullptr;
	vk::raii::DeviceMemory defaultTextureMemory = nullptr;
	vk::raii::ImageView defaultTextureView = nullptr;

	vk::raii::Image defaultNormalImage = nullptr;
	vk::raii::DeviceMemory defaultNormalMemory = nullptr;
	vk::raii::ImageView defaultNormalView = nullptr;

	std::vector<const char*> requiredDeviceExtension = { vk::KHRSwapchainExtensionName };

	std::array<GameObject, MAX_OBJECTS> gameObjects;



	// ENGINE REFACTOR

	Camera camera;
	float lastFrameTime = 0.0f;

	// Engine - Renderer
	vk::raii::PipelineLayout pbrPipelineLayout = nullptr;
	vk::raii::Pipeline pbrPipeline = nullptr;

	// Push constant block for PBR material properties
	struct PushConstantBlock {
		glm::vec4 baseColorFactor;
		float metallicFactor;
		float roughnessFactor;
		int baseColorTextureSet;
		int physicalDescriptorTextureSet;
		int normalTextureSet;
		int occlusionTextureSet;
		int emissiveTextureSet;
		float alphaMask;
		float alphaMaskCutoff;
	};


};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}