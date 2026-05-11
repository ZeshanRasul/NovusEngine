#include "renderer/renderer.h"
#include <ktx.h>

#include "../vulkan/command_pool.h"
#include "../vulkan/command_buffer.h"
#include "../vulkan/buffer.h"
#include "../vulkan/texture.h"
#include "../vulkan/image.h"
#include "../vulkan/image_view.h"
#include "../vulkan/pipeline.h"
#include "../model/model.h"
#include "../vulkan/uniform_buffer.h"
#include "../ECS/entity.h"
// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

void Renderer::run()
{
	initWindow();
	initVulkan();
	mainLoop();
	cleanup();
}

// ---------------------------------------------------------------------------
// Initialisation / shutdown
// ---------------------------------------------------------------------------

void Renderer::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Novus Engine", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

	camera.setupInputCallbacks(window);
	glfwSetWindowUserPointer(window, &camera); // Set the user pointer for the InputSystem callbacks
	InputSystem::Initialize(window, &camera);
	camera.setPosition(glm::vec3(0.0f, 20.0f, 43.0f));
	camera.getViewMatrix();
	camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f);
}

void Renderer::initVulkan()
{
	deviceInit();
	createSwapChain();
	createSwapChainImageViews();
	createDescriptorSetLayout();

	if (!createPBRPipeline())
	{
		std::cerr << "Failed to create PBR pipeline" << std::endl;
	}

	CommandPool::init(device, queueIndex, commandPool);
	createDepthResources();
	createTextureSampler();
	createDefaultTextures();
	setupGameObjects();
	for (auto& entityPtr : entities)
	{
		createVertexBuffer(*entityPtr->GetComponent<RenderableComponent>());
		createIndexBuffer(*entityPtr->GetComponent<RenderableComponent>());
	}
	UniformBuffer::createUniformBuffers(entities, device, physicalDevice, MAX_FRAMES_IN_FLIGHT);
	createDescriptorPool();
	createDescriptorSets();
	CommandBuffer::init(device, queueIndex, commandPool, commandBuffers, MAX_FRAMES_IN_FLIGHT);
	createSyncObjects();

	imGui = new ImGuiVulkanUtil(
		device,
		physicalDevice,
		queue,
		queueIndex
	);

	imGui->init(swapChainExtent.width, swapChainExtent.height);
	imGui->initResources(); // No renderPass needed with dynamic rendering
}

void Renderer::mainLoop()
{
	lastFrameTime = 0.0f;
	while (!glfwWindowShouldClose(window))
	{
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto        currentTime = std::chrono::high_resolution_clock::now();
		float       time = std::chrono::duration<float>(currentTime - startTime).count();
		float       deltaTime = time - lastFrameTime;
		lastFrameTime = time;
		InputSystem::Update(deltaTime);

		camera.processInput(window, camera, deltaTime);
		drawFrame();
	}

	device.waitIdle();
}

void Renderer::cleanupSwapChain()
{
	swapChainImageViews.clear();
	swapChain = nullptr;
}

void Renderer::cleanup()
{
	glfwDestroyWindow(window);
	glfwTerminate();
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void Renderer::framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/)
{
	auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void Renderer::mousePosCallback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddMousePosEvent(static_cast<float>(xpos), static_cast<float>(ypos));

	if (!io.WantCaptureMouse)
		Camera::mouseCallback(window, xpos, ypos);
}

void Renderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
	ImGuiIO& io = ImGui::GetIO();
	if (button >= 0 && button < ImGuiMouseButton_COUNT)
		io.AddMouseButtonEvent(button, action == GLFW_PRESS);
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddMouseWheelEvent(static_cast<float>(xoffset), static_cast<float>(yoffset));

	if (!io.WantCaptureMouse)
		Camera::scrollCallback(window, xoffset, yoffset);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL
Renderer::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT  severity,
	vk::DebugUtilsMessageTypeFlagsEXT          type,
	const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void*)
{
	if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
		severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
	{
		std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
	}
	return vk::False;
}

void Renderer::renderImgui()
{
	if (imGui->newFrame()) {
	}

	// Create a window for camera controls
	ImGui::Begin("Camera Controls");

	// Add a button to reset camera position
	if (ImGui::Button("Reset Camera")) {
		camera.setPosition(glm::vec3(0.0f, 20.0f, 23.0f));
		camera.setYaw(-90.0f);
		camera.setPitch(0.0f);
	}

	// Add sliders for camera settings
	float movementSpeed = camera.getMovementSpeed();
	if (ImGui::SliderFloat("Movement Speed", &movementSpeed, 1.0f, 10.0f)) {
		camera.setMovementSpeed(movementSpeed);
	}

	float sensitivity = camera.getMouseSensitivity();
	if (ImGui::SliderFloat("Mouse Sensitivity", &sensitivity, 0.1f, 1.0f)) {
		camera.setMouseSensitivity(sensitivity);
	}

	float zoom = camera.getZoom();
	if (ImGui::SliderFloat("Zoom", &zoom, 1.0f, 45.0f)) {
		camera.setZoom(zoom);
	}

	ImGui::End();

	// End the frame
	ImGui::EndFrame();

	// Render to generate draw data
	ImGui::Render();
	imGui->updateBuffers();
}

// ---------------------------------------------------------------------------
// Instance / device
// ---------------------------------------------------------------------------

std::vector<const char*> Renderer::getRequiredInstanceExtensions()
{
	uint32_t glfwExtensionCount = 0;
	auto     glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
		extensions.push_back(vk::EXTDebugUtilsExtensionName);

	return extensions;
}

bool Renderer::deviceInit()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();

	return true;
}

void Renderer::createInstance()
{
	constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Novus Engine",
										   .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
										   .pEngineName = "No Engine",
										   .engineVersion = VK_MAKE_VERSION(1, 0, 0),
										   .apiVersion = vk::ApiVersion14 };

	std::vector<char const*> requiredLayers;
	if (enableValidationLayers)
		requiredLayers.assign(validationLayers.begin(), validationLayers.end());

	auto layerProperties = context.enumerateInstanceLayerProperties();
	if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
		return std::ranges::none_of(layerProperties, [requiredLayer](auto const& lp) {
			return strcmp(lp.layerName, requiredLayer) == 0;
			});
		}))
	{
		throw std::runtime_error("One or more required layers are not supported!");
	}

	auto requiredExtensions = getRequiredInstanceExtensions();

	auto extensionProperties = context.enumerateInstanceExtensionProperties();
	auto unsupportedIt = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& re) {
		return std::ranges::none_of(extensionProperties, [re](auto const& ep) {
			return strcmp(ep.extensionName, re) == 0;
			});
		});
	if (unsupportedIt != requiredExtensions.end())
		throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedIt));

	vk::InstanceCreateInfo createInfo{
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
		.ppEnabledLayerNames = requiredLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data() };

	instance = vk::raii::Instance(context, createInfo);
}

void Renderer::setupDebugMessenger()
{
	if (!enableValidationLayers)
		return;

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback };
	debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void Renderer::createSurface()
{
	VkSurfaceKHR _surface;
	if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		throw std::runtime_error("failed to create window surface!");
	surface = vk::raii::SurfaceKHR(instance, _surface);
}

bool Renderer::isDeviceSuitable(vk::raii::PhysicalDevice const& pd)
{
	bool supportsVulkan1_3 = pd.getProperties().apiVersion >= vk::ApiVersion13;

	auto queueFamilies = pd.getQueueFamilyProperties();
	bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
		return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
		});

	auto availableExts = pd.enumerateDeviceExtensionProperties();
	bool supportsAllRequiredExts = std::ranges::all_of(requiredDeviceExtension, [&availableExts](auto const& req) {
		return std::ranges::any_of(availableExts, [req](auto const& avail) {
			return strcmp(avail.extensionName, req) == 0;
			});
		});

	auto features = pd.template getFeatures2<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
	bool supportsRequiredFeatures =
		features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
		features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
		features.template get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>().dynamicRenderingLocalRead;

	return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExts && supportsRequiredFeatures;
}

void Renderer::pickPhysicalDevice()
{
	std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const& pd) { return isDeviceSuitable(pd); });
	if (devIter == physicalDevices.end())
		throw std::runtime_error("failed to find a suitable GPU!");
	physicalDevice = *devIter;
}

void Renderer::createLogicalDevice()
{
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

	queueIndex = ~0u;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
	{
		if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
			physicalDevice.getSurfaceSupportKHR(i, *surface))
		{
			queueIndex = i;
			break;
		}
	}

	if (queueIndex == ~0u)
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");

	vk::StructureChain<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>
		featureChain = {
			{.features = {.samplerAnisotropy = true } },
			{.shaderDrawParameters = true },
			{.synchronization2 = true, .dynamicRendering = true },
			{.extendedDynamicState = true },
			{.dynamicRenderingLocalRead = true }
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

// ---------------------------------------------------------------------------
// Swap chain
// ---------------------------------------------------------------------------

vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats)
{
	assert(!availableFormats.empty());
	const auto it = std::ranges::find_if(availableFormats, [](const auto& f) {
		return f.format == vk::Format::eB8G8R8A8Unorm && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
		});
	return it != availableFormats.end() ? *it : availableFormats[0];
}

vk::PresentModeKHR Renderer::chooseSwapPresentMode(std::vector<vk::PresentModeKHR> const& availablePresentModes)
{
	assert(std::ranges::any_of(availablePresentModes, [](auto pm) { return pm == vk::PresentModeKHR::eFifo; }));
	return std::ranges::any_of(availablePresentModes, [](const vk::PresentModeKHR v) {
		return vk::PresentModeKHR::eMailbox == v;
		}) ?
		vk::PresentModeKHR::eMailbox :
			vk::PresentModeKHR::eFifo;
}

vk::Extent2D Renderer::chooseSwapExtent(vk::SurfaceCapabilitiesKHR const& capabilities)
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		return capabilities.currentExtent;

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	return {
		std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
		std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
	};
}

uint32_t Renderer::chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities)
{
	auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
	if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
		minImageCount = surfaceCapabilities.maxImageCount;
	return minImageCount;
}

void Renderer::createSwapChain()
{
	auto surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);

	swapChainExtent = chooseSwapExtent(surfaceCapabilities);
	uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

	std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
	std::vector<vk::PresentModeKHR>   availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
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

void Renderer::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();
	cleanupSwapChain();
	createSwapChain();
	createSwapChainImageViews();
	createDepthResources();
}

// ---------------------------------------------------------------------------
// Image views
// ---------------------------------------------------------------------------

void Renderer::createSwapChainImageViews()
{
	assert(swapChainImageViews.empty());
	swapChainImageViews.reserve(swapChainImages.size());
	for (auto& image : swapChainImages)
		swapChainImageViews.emplace_back(ImageView::createImageView(device, image, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));
}

// ---------------------------------------------------------------------------
// Shaders / pipelines
// ---------------------------------------------------------------------------

std::vector<char> Renderer::readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("failed to open file!");

	std::vector<char> buffer(file.tellg());
	file.seekg(0, std::ios::beg);
	file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
	file.close();
	return buffer;
}

vk::raii::ShaderModule Renderer::createShaderModule(const std::vector<char>& code) const
{
	vk::ShaderModuleCreateInfo createInfo{ .codeSize = code.size() * sizeof(char),
										   .pCode = reinterpret_cast<const uint32_t*>(code.data()) };
	return vk::raii::ShaderModule{ device, createInfo };
}

void Renderer::createDescriptorSetLayout()
{
	std::array<vk::DescriptorSetLayoutBinding, 6> bindings{ {
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

void Renderer::createGraphicsPipeline()
{
	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "../shaders/slang.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "../shaders/slang.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};

	config.vertexBindings = { bindingDescription };
	config.vertexAttributes = {
		attributeDescriptions.begin(),
		attributeDescriptions.end()
	};

	config.descriptorSetLayouts = { *descriptorSetLayout };
	config.colorAttachmentFormats = { swapChainSurfaceFormat.format };
	config.depthAttachmentFormat = findDepthFormat();

	auto pipelineBundle = Pipeline::createPipeline(device, config);
	pipelineLayout = std::move(pipelineBundle.layout);
	graphicsPipeline = std::move(pipelineBundle.pipeline);
}

bool Renderer::createPBRPipeline()
{
	try
	{
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		vk::PushConstantRange pushConstantRange{
			.stageFlags = vk::ShaderStageFlagBits::eFragment,
			.offset = 0,
			.size = sizeof(PushConstantBlock)
		};

		Pipeline::PipelineConfig config{};
		config.shaderStages = {
			{ "../shaders/pbr.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
			{ "../shaders/pbr.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
		};

		config.vertexBindings = { bindingDescription };
		config.vertexAttributes = {
			attributeDescriptions.begin(),
			attributeDescriptions.end()
		};

		config.descriptorSetLayouts = { *descriptorSetLayout };
		config.pushConstantRanges = { pushConstantRange };
		config.colorAttachmentFormats = { swapChainSurfaceFormat.format };
		config.depthAttachmentFormat = findDepthFormat();

		auto pipelineBundle = Pipeline::createPipeline(device, config);
		pbrPipelineLayout = std::move(pipelineBundle.layout);
		pbrPipeline = std::move(pipelineBundle.pipeline);

		return true;
	}
	catch (std::exception const& e)
	{
		std::cerr << "Error creating PBR pipeline: " << e.what() << std::endl;
		return false;
	}
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
	auto& commandBuffer = commandBuffers[frameIndex];
	commandBuffer.begin({});

	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);

	transition_image_layout(*depthImage,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = depthImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};

	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

	for (auto& entityPtr : entities)
	{
		auto* renderable = entityPtr->GetComponent<RenderableComponent>();
		auto* transform = entityPtr->GetComponent<TransformComponent>();
		if (!renderable || !transform)
			continue;

		UniformBuffer::updateUniformBuffer(frameIndex, renderable, transform, &camera, swapChainExtent);
		vk::Buffer     vertexBuffers[] = { renderable->vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*renderable->indexBuffer, 0, vk::IndexType::eUint32);

		for (const auto& mesh : renderable->meshes)
		{
			pushMaterialProperties(commandBuffer, renderable, mesh.materialIndex);
			uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable->materialDescriptorSets.size() ? mesh.materialIndex : 0;

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*pbrPipelineLayout,
				0,
				*renderable->materialDescriptorSets[descriptorMaterialIndex][frameIndex],
				nullptr);

			commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
		}
	}

	commandBuffer.endRendering();

	// ↓ ImGui draws on top, image still in eColorAttachmentOptimal
	imGui->drawFrame(commandBuffer, *swapChainImageViews[imageIndex]);

	// ↓ then transition to present
	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe,
		vk::ImageAspectFlagBits::eColor);

	commandBuffer.end();
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
{
	vk::raii::CommandBuffer commandCopyBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
	commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{ .size = size });
	CommandBuffer::endSingleTimeCommands(std::move(commandCopyBuffer), queue);
}

void Renderer::createVertexBuffer(RenderableComponent& gameObj)
{
	vk::DeviceSize         bufferSize = sizeof(gameObj.vertices[0]) * gameObj.vertices.size();
	vk::raii::Buffer       stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	Buffer::createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, gameObj.vertices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	Buffer::createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		gameObj.vertexBuffer, gameObj.vertexBufferMemory);

	copyBuffer(stagingBuffer, gameObj.vertexBuffer, bufferSize);
}

void Renderer::createIndexBuffer(RenderableComponent& gameObj)
{
	vk::DeviceSize         bufferSize = sizeof(gameObj.indices[0]) * gameObj.indices.size();
	if (bufferSize == 0) return;

	vk::raii::Buffer       stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	Buffer::createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, gameObj.indices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	Buffer::createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		gameObj.indexBuffer, gameObj.indexBufferMemory);

	copyBuffer(stagingBuffer, gameObj.indexBuffer, bufferSize);
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

void Renderer::transition_image_layout(vk::Image               image,
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
			.layerCount = 1
		}
	};
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};
	commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
}

// ---------------------------------------------------------------------------
// Depth
// ---------------------------------------------------------------------------

vk::Format Renderer::findSupportedFormat(const std::vector<vk::Format>& candidates,
	vk::ImageTiling tiling, vk::FormatFeatureFlags features)
{
	for (const auto format : candidates)
	{
		vk::FormatProperties props = physicalDevice.getFormatProperties(format);

		if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features)
			return format;
		if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features)
			return format;
	}
	throw std::runtime_error("failed to find supported format!");
}

vk::Format Renderer::findDepthFormat()
{
	return findSupportedFormat(
		{ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint },
		vk::ImageTiling::eOptimal,
		vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool Renderer::hasStencilComponent(vk::Format format)
{
	return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

void Renderer::createDepthResources()
{
	vk::Format depthFormat = findDepthFormat();
	Image::createImage(device, physicalDevice, swapChainExtent.width, swapChainExtent.height, depthFormat,
		vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
		vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
	depthImageView = ImageView::createImageView(device, depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

void Renderer::loadPBRTextures(const Material& material, RenderableComponent::PBRTextures& textures)
{
	std::cout << "Loading PBR textures for material: " << material.GetName() << std::endl;

	if (!material.albedoTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.albedoTexturePath, textures.baseColorImage, textures.baseColorMemory, textures.baseColorView, true);

	if (!material.metallicRoughnessTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.metallicRoughnessTexturePath, textures.metallicRoughnessImage, textures.metallicRoughnessMemory, textures.metallicRoughnessView, false);

	if (!material.normalTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.normalTexturePath, textures.normalImage, textures.normalMemory, textures.normalView, false);

	if (!material.occlusionTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.occlusionTexturePath, textures.occlusionImage, textures.occlusionMemory, textures.occlusionView, false);

	if (!material.emissiveTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.emissiveTexturePath, textures.emissiveImage, textures.emissiveMemory, textures.emissiveView, true);
}

void Renderer::createDefaultTextures()
{
	// White 1x1
	{
		const uint32_t white = 0xFFFFFFFF;
		vk::DeviceSize imageSize = sizeof(uint32_t);

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		Buffer::createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, &white, imageSize);
		stagingBufferMemory.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, defaultTextureImage, defaultTextureMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuffer, defaultTextureImage, 1, 1);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

		defaultTextureView = ImageView::createImageView(device, *defaultTextureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}

	// Flat normal 1x1
	{
		const uint32_t flatNormal = 0xFFFF7F7F;
		vk::DeviceSize imageSize = sizeof(uint32_t);

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		Buffer::createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, &flatNormal, imageSize);
		stagingBufferMemory.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, defaultNormalImage, defaultNormalMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuffer, defaultNormalImage, 1, 1);
		Image::transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

		defaultNormalView = ImageView::createImageView(device, *defaultNormalImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}
}

void Renderer::createTextureSampler()
{
	vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
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

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

void Renderer::createDescriptorPool()
{
	uint32_t materialCount = 0;
	for (auto& entityPtr : entities)
	{
		auto* renderable = entityPtr->GetComponent<RenderableComponent>();
		if (renderable)
			materialCount += static_cast<uint32_t>(std::max<size_t>(1, renderable->materials.size()));
	}

	std::array<vk::DescriptorPoolSize, 2> poolSize{ {
		{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = materialCount * MAX_FRAMES_IN_FLIGHT },
		{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 5 * materialCount * MAX_FRAMES_IN_FLIGHT }
	} };

	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = materialCount * MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = static_cast<uint32_t>(poolSize.size()),
		.pPoolSizes = poolSize.data()
	};

	descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
}

void Renderer::createDescriptorSets()
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

			std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
			vk::DescriptorSetAllocateInfo allocInfo{
				.descriptorPool = *descriptorPool,
				.descriptorSetCount = static_cast<uint32_t>(layouts.size()),
				.pSetLayouts = layouts.data()
			};

			descriptorSetsForMaterial = device.allocateDescriptorSets(allocInfo);

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
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

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------

void Renderer::createSyncObjects()
{
	assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

	for (size_t i = 0; i < swapChainImages.size(); i++)
		renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
	{
		presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
		inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
	}
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

void Renderer::pushMaterialProperties(vk::CommandBuffer commandBuffer,
	const RenderableComponent* model, uint32_t materialIndex)
{
	const Material& material = model->materials[materialIndex < model->materials.size() ? materialIndex : 0];

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

	commandBuffer.pushConstants(*pbrPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstantBlock), &pushConstants);
}

void Renderer::drawFrame()
{
	auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
	if (fenceResult != vk::Result::eSuccess)
		throw std::runtime_error("Failed to wait for draw fence!");

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

	renderImgui();

	device.resetFences(*inFlightFences[frameIndex]);
	commandBuffers[frameIndex].reset();
	recordCommandBuffer(imageIndex);

	queue.waitIdle();

	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*commandBuffers[frameIndex],
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
	};

	queue.submit(submitInfo, *inFlightFences[frameIndex]);

	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
		.swapchainCount = 1,
		.pSwapchains = &*swapChain,
		.pImageIndices = &imageIndex
	};

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

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

void Renderer::setupGameObjects()
{
	auto makeEntity = [&](const std::string& name,
		glm::vec3 position, glm::vec3 rotation, glm::vec3 scale,
		const std::string& modelPath) -> Entity& {
			entities.push_back(std::make_unique<Entity>(name));
			Entity& entity = *entities.back();

			auto* transform = entity.AddComponent<TransformComponent>();
			transform->SetPosition(position);
			transform->SetRotation(glm::quat(rotation));
			transform->SetScale(scale);

			auto* renderable = entity.AddComponent<RenderableComponent>();
			Model::loadModel(modelPath, *renderable);
			for (size_t i = 0; i < renderable->materials.size(); ++i)
				loadPBRTextures(renderable->materials[i], renderable->materialTextures[i]);

			return entity;
		};

	makeEntity("FlightHelmet_Left",
		{ -3.0f, 10.5f, -100.0f }, { 0.0f, 0.0f, 0.0f }, { 33.0f, 33.0f, 33.0f },
		"../models/FlightHelmet.gltf");

	{
		Entity& e = makeEntity("DamagedHelmet",
			{ 3.0f, 52.0f, -100.0f }, { 0.0f, 0.0f, 0.0f }, { 11.5f, 11.5f, 11.5f },
			"../models/DamagedHelmet.gltf");
		e.GetComponent<RenderableComponent>()->materials[0].metallicFactor = 1.0f;
	}

	makeEntity("Sponza",
		{ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 3.0f, 3.0f, 3.0f },
		"../models/Sponza.gltf");
}

