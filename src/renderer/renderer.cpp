#include "renderer/renderer.h"

#include <ktx.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

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
    camera.getViewMatrix();
    camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT);
}

void Renderer::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createGraphicsPipeline();

    if (!createPBRPipeline())
    {
        std::cerr << "Failed to create PBR pipeline" << std::endl;
    }

    createCommandPool();
    createDepthResources();
    createTextureSampler();
    createDefaultTextures();
    setupGameObjects();
    for (auto& entityPtr : entities)
    {
        createVertexBuffer(*entityPtr->GetComponent<RenderableComponent>());
        createIndexBuffer(*entityPtr->GetComponent<RenderableComponent>());
    }
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

void Renderer::mainLoop()
{
    lastFrameTime = 0.0f;
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        static auto startTime   = std::chrono::high_resolution_clock::now();
        auto        currentTime = std::chrono::high_resolution_clock::now();
        float       time        = std::chrono::duration<float>(currentTime - startTime).count();
        float       deltaTime   = time - lastFrameTime;
        lastFrameTime           = time;
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
    auto app               = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
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

// ---------------------------------------------------------------------------
// Instance / device
// ---------------------------------------------------------------------------

std::vector<const char*> Renderer::getRequiredInstanceExtensions()
{
    uint32_t glfwExtensionCount = 0;
    auto     glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
        extensions.push_back(vk::EXTDebugUtilsExtensionName);

    return extensions;
}

void Renderer::createInstance()
{
    constexpr vk::ApplicationInfo appInfo{ .pApplicationName   = "Novus Engine",
                                           .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                           .pEngineName        = "No Engine",
                                           .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
                                           .apiVersion         = vk::ApiVersion14 };

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
    auto unsupportedIt       = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& re) {
        return std::ranges::none_of(extensionProperties, [re](auto const& ep) {
            return strcmp(ep.extensionName, re) == 0;
        });
    });
    if (unsupportedIt != requiredExtensions.end())
        throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedIt));

    vk::InstanceCreateInfo createInfo{
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
        .ppEnabledLayerNames     = requiredLayers.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
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

    auto queueFamilies   = pd.getQueueFamilyProperties();
    bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
        return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
    });

    auto availableExts         = pd.enumerateDeviceExtensionProperties();
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
            { .features = { .samplerAnisotropy = true } },
            { .shaderDrawParameters = true },
            { .synchronization2 = true, .dynamicRendering = true },
            { .extendedDynamicState = true },
            { .dynamicRenderingLocalRead = true }
        };

    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex,
                                                      .queueCount       = 1,
                                                      .pQueuePriorities = &queuePriority };

    vk::DeviceCreateInfo deviceCreateInfo{
        .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
        .queueCreateInfoCount    = 1,
        .pQueueCreateInfos       = &deviceQueueCreateInfo,
        .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtension.size()),
        .ppEnabledExtensionNames = requiredDeviceExtension.data()
    };

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    queue  = vk::raii::Queue(device, queueIndex, 0);
}

// ---------------------------------------------------------------------------
// Swap chain
// ---------------------------------------------------------------------------

vk::SurfaceFormatKHR Renderer::chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const& availableFormats)
{
    assert(!availableFormats.empty());
    const auto it = std::ranges::find_if(availableFormats, [](const auto& f) {
        return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
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

    swapChainExtent       = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats      = physicalDevice.getSurfaceFormatsKHR(surface);
    std::vector<vk::PresentModeKHR>   availablePresentModes = physicalDevice.getSurfacePresentModesKHR(surface);
    swapChainSurfaceFormat                                    = chooseSwapSurfaceFormat(availableFormats);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{
        .surface          = *surface,
        .minImageCount    = minImageCount,
        .imageFormat      = swapChainSurfaceFormat.format,
        .imageColorSpace  = swapChainSurfaceFormat.colorSpace,
        .imageExtent      = swapChainExtent,
        .imageArrayLayers = 1,
        .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = vk::SharingMode::eExclusive,
        .preTransform     = surfaceCapabilities.currentTransform,
        .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
        .presentMode      = chooseSwapPresentMode(availablePresentModes),
        .clipped          = true,
        .oldSwapchain     = nullptr
    };

    swapChain       = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
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
    createImageViews();
    createDepthResources();
}

// ---------------------------------------------------------------------------
// Image views
// ---------------------------------------------------------------------------

vk::raii::ImageView Renderer::createImageView(vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
    vk::ImageViewCreateInfo viewInfo{
        .image            = image,
        .viewType         = vk::ImageViewType::e2D,
        .format           = format,
        .subresourceRange = { .aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 } };
    return vk::raii::ImageView(device, viewInfo);
}

void Renderer::createImageViews()
{
    assert(swapChainImageViews.empty());
    swapChainImageViews.reserve(swapChainImages.size());
    for (auto& image : swapChainImages)
        swapChainImageViews.emplace_back(createImageView(image, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor));
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
                                           .pCode    = reinterpret_cast<const uint32_t*>(code.data()) };
    return vk::raii::ShaderModule{ device, createInfo };
}

void Renderer::createDescriptorSetLayout()
{
    std::array<vk::DescriptorSetLayoutBinding, 6> bindings{ {
        { .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        { .binding = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 2, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 3, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 4, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 5, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
    } };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{ .bindingCount = static_cast<uint32_t>(bindings.size()),
                                                  .pBindings    = bindings.data() };
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
}

void Renderer::createGraphicsPipeline()
{
    auto shaderCode = readFile("../shaders/slang.spv");

    vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain" };
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{ .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain" };

    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    std::vector<vk::DynamicState>     dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dynamicState{ .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
                                                      .pDynamicStates    = dynamicStates.data() };

    auto bindingDescription   = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &bindingDescription,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
        .pVertexAttributeDescriptions    = attributeDescriptions.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList };

    vk::Viewport viewport{ 0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f };
    vk::Rect2D   scissor{ vk::Offset2D{ 0, 0 }, swapChainExtent };

    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .depthClampEnable        = vk::False,
        .rasterizerDiscardEnable = vk::False,
        .polygonMode             = vk::PolygonMode::eFill,
        .cullMode                = vk::CullModeFlagBits::eBack,
        .frontFace               = vk::FrontFace::eCounterClockwise,
        .depthBiasEnable         = vk::False,
        .lineWidth               = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{ .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable       = vk::True,
        .depthWriteEnable      = vk::True,
        .depthCompareOp        = vk::CompareOp::eLess,
        .depthBoundsTestEnable = vk::False,
        .stencilTestEnable     = vk::False
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable         = vk::True,
        .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
        .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
        .colorBlendOp        = vk::BlendOp::eAdd,
        .srcAlphaBlendFactor = vk::BlendFactor::eOne,
        .dstAlphaBlendFactor = vk::BlendFactor::eZero,
        .alphaBlendOp        = vk::BlendOp::eAdd,
        .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending{ .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0 };
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::Format depthFormat = findDepthFormat();

    vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
        { .stageCount          = 2,
          .pStages             = shaderStages,
          .pVertexInputState   = &vertexInputInfo,
          .pInputAssemblyState = &inputAssembly,
          .pViewportState      = &viewportState,
          .pRasterizationState = &rasterizer,
          .pMultisampleState   = &multisampling,
          .pDepthStencilState  = &depthStencil,
          .pColorBlendState    = &colorBlending,
          .pDynamicState       = &dynamicState,
          .layout              = pipelineLayout,
          .renderPass          = nullptr },
        { .colorAttachmentCount = 1, .pColorAttachmentFormats = &swapChainSurfaceFormat.format, .depthAttachmentFormat = depthFormat }
    };

    graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
}

bool Renderer::createPBRPipeline()
{
    try
    {
        auto shaderCode = readFile("../shaders/pbr.spv");

        vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo;
        vertShaderStageInfo.setStage(vk::ShaderStageFlagBits::eVertex).setModule(*shaderModule).setPName("vertMain");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo;
        fragShaderStageInfo.setStage(vk::ShaderStageFlagBits::eFragment).setModule(*shaderModule).setPName("fragMain");

        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
        auto                                   bindingDescription    = Vertex::getBindingDescription();
        auto                                   attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputInfo.setVertexBindingDescriptionCount(1)
            .setPVertexBindingDescriptions(&bindingDescription)
            .setVertexAttributeDescriptionCount(static_cast<uint32_t>(attributeDescriptions.size()))
            .setPVertexAttributeDescriptions(attributeDescriptions.data());

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly;
        inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList).setPrimitiveRestartEnable(false);

        vk::PipelineViewportStateCreateInfo viewportState;
        viewportState.setViewportCount(1).setScissorCount(1);

        std::vector<vk::DynamicState>      dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState;
        dynamicState.setDynamicStateCount(static_cast<uint32_t>(dynamicStates.size())).setPDynamicStates(dynamicStates.data());

        vk::PipelineRasterizationStateCreateInfo rasterizer;
        rasterizer.setDepthClampEnable(false)
            .setRasterizerDiscardEnable(false)
            .setPolygonMode(vk::PolygonMode::eFill)
            .setLineWidth(1.0f)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eCounterClockwise)
            .setDepthBiasEnable(false);

        vk::PipelineMultisampleStateCreateInfo multisampling;
        multisampling.setSampleShadingEnable(false).setRasterizationSamples(vk::SampleCountFlagBits::e1);

        vk::PipelineDepthStencilStateCreateInfo depthStencil;
        depthStencil.setDepthTestEnable(true)
            .setDepthWriteEnable(true)
            .setDepthCompareOp(vk::CompareOp::eLess)
            .setDepthBoundsTestEnable(false)
            .setStencilTestEnable(false);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment;
        colorBlendAttachment
            .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
            .setBlendEnable(true)
            .setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
            .setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
            .setColorBlendOp(vk::BlendOp::eAdd)
            .setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
            .setDstAlphaBlendFactor(vk::BlendFactor::eZero)
            .setAlphaBlendOp(vk::BlendOp::eAdd);

        vk::PipelineColorBlendStateCreateInfo colorBlending;
        colorBlending.setLogicOpEnable(false).setAttachmentCount(1).setPAttachments(&colorBlendAttachment);

        vk::PushConstantRange pushConstantRange;
        pushConstantRange.setStageFlags(vk::ShaderStageFlagBits::eFragment).setOffset(0).setSize(sizeof(PushConstantBlock));

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
        pipelineLayoutInfo.setSetLayoutCount(1)
            .setPSetLayouts(&*descriptorSetLayout)
            .setPushConstantRangeCount(1)
            .setPPushConstantRanges(&pushConstantRange);

        pbrPipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo;
        pipelineInfo.setStageCount(static_cast<uint32_t>(shaderStages.size()))
            .setPStages(shaderStages.data())
            .setPVertexInputState(&vertexInputInfo)
            .setPInputAssemblyState(&inputAssembly)
            .setPViewportState(&viewportState)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampling)
            .setPDepthStencilState(&depthStencil)
            .setPColorBlendState(&colorBlending)
            .setPDynamicState(&dynamicState)
            .setLayout(*pbrPipelineLayout)
            .setRenderPass(nullptr)
            .setSubpass(0)
            .setBasePipelineHandle(nullptr);

        vk::PipelineRenderingCreateInfo renderingInfo;
        renderingInfo.setColorAttachmentCount(1)
            .setPColorAttachmentFormats(&swapChainSurfaceFormat.format)
            .setDepthAttachmentFormat(findDepthFormat());
        pipelineInfo.setPNext(&renderingInfo);

        pbrPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error creating PBR pipeline: " << e.what() << std::endl;
        return false;
    }
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

void Renderer::createCommandPool()
{
    vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                        .queueFamilyIndex = queueIndex };
    commandPool = vk::raii::CommandPool(device, poolInfo);
}

void Renderer::createCommandBuffers()
{
    vk::CommandBufferAllocateInfo allocInfo{ .commandPool        = commandPool,
                                             .level              = vk::CommandBufferLevel::ePrimary,
                                             .commandBufferCount = MAX_FRAMES_IN_FLIGHT };
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
}

vk::raii::CommandBuffer Renderer::beginSingleTimeCommands()
{
    vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
    vk::raii::CommandBuffer       commandBuffer = std::move(vk::raii::CommandBuffers(device, allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
    commandBuffer.begin(beginInfo);

    return commandBuffer;
}

void Renderer::endSingleTimeCommands(vk::raii::CommandBuffer&& commandBuffer)
{
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandBuffer };
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
}

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
        .imageView   = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eStore,
        .clearValue  = clearColor
    };

    vk::RenderingAttachmentInfo depthAttachmentInfo = {
        .imageView   = depthImageView,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp      = vk::AttachmentLoadOp::eClear,
        .storeOp     = vk::AttachmentStoreOp::eDontCare,
        .clearValue  = clearDepth
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea             = { .offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount             = 1,
        .colorAttachmentCount   = 1,
        .pColorAttachments      = &attachmentInfo,
        .pDepthAttachment       = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

    for (auto& entityPtr : entities)
    {
        auto* renderable = entityPtr->GetComponent<RenderableComponent>();
        auto* transform  = entityPtr->GetComponent<TransformComponent>();
        if (!renderable || !transform)
            continue;

        updateUniformBuffer(frameIndex, renderable, transform, &camera);
        vk::Buffer     vertexBuffers[] = { renderable->vertexBuffer };
        vk::DeviceSize offsets[]       = { 0 };
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

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void Renderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                            vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory)
{
    vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
    buffer = vk::raii::Buffer(device, bufferInfo);

    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
    };
    bufferMemory = vk::raii::DeviceMemory(device, allocInfo);
    buffer.bindMemory(*bufferMemory, 0);
}

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
{
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{ .size = size });
    endSingleTimeCommands(std::move(commandCopyBuffer));
}

void Renderer::createVertexBuffer(RenderableComponent& gameObj)
{
    vk::DeviceSize         bufferSize = sizeof(gameObj.vertices[0]) * gameObj.vertices.size();
    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, gameObj.vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 gameObj.vertexBuffer, gameObj.vertexBufferMemory);

    copyBuffer(stagingBuffer, gameObj.vertexBuffer, bufferSize);
}

void Renderer::createIndexBuffer(RenderableComponent& gameObj)
{
    vk::DeviceSize         bufferSize = sizeof(gameObj.indices[0]) * gameObj.indices.size();
    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, gameObj.indices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eDeviceLocal,
                 gameObj.indexBuffer, gameObj.indexBufferMemory);

    copyBuffer(stagingBuffer, gameObj.indexBuffer, bufferSize);
}

void Renderer::createUniformBuffers()
{
    for (auto& entityPtr : entities)
    {
        auto& gameObject = *entityPtr->GetComponent<RenderableComponent>();
        gameObject.uniformBuffers.clear();
        gameObject.uniformBuffersMemory.clear();
        gameObject.uniformBuffersMapped.clear();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vk::DeviceSize         bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer       buffer({});
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

void Renderer::updateUniformBuffers()
{
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 proj = camera.getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 2000.0f);

    for (auto& entityPtr : entities)
    {
        auto* renderable = entityPtr->GetComponent<RenderableComponent>();
        auto* transform  = entityPtr->GetComponent<TransformComponent>();

        glm::mat4 model = transform ? transform->GetTransformMatrix() : glm::mat4(1.0f);

        UniformBufferObject ubo{ .model = model, .view = view, .proj = proj };
        memcpy(renderable->uniformBuffersMapped[frameIndex], &ubo, sizeof(ubo));
    }
}

void Renderer::updateUniformBuffer(uint32_t currentFrame, RenderableComponent* renderable,
                                   TransformComponent* transform, Camera* cam)
{
    UniformBufferObject ubo{};

    ubo.model = transform ? transform->GetTransformMatrix() : glm::mat4(1.0f);

    if (cam)
    {
        ubo.view = cam->getViewMatrix();
        ubo.proj = cam->getProjectionMatrix(static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 2000.0f);
    }
    else
    {
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / static_cast<float>(swapChainExtent.height), 0.1f, 100.0f);
        ubo.proj[1][1] *= -1;
    }

    ubo.lightPositions[0] = glm::vec4(-1.5f, 3.5f,  3.0f, 1.0f);
    ubo.lightPositions[1] = glm::vec4(-4.5f, 2.5f,  2.0f, 1.0f);
    ubo.lightPositions[2] = glm::vec4( 1.5f, 2.0f, -2.5f, 1.0f);
    ubo.lightPositions[3] = glm::vec4(-2.0f, 4.5f, -1.0f, 1.0f);

    ubo.lightColors[0] = glm::vec4(2.0f, 0.0f, 0.0f, 1.0f);
    ubo.lightColors[1] = glm::vec4(0.0f, 0.0f, 4.0f, 1.0f);
    ubo.lightColors[2] = glm::vec4(0.0f, 4.0f, 0.0f, 1.0f);
    ubo.lightColors[3] = glm::vec4(6.0f, 0.1f, 0.0f, 1.0f);

    ubo.camPos                  = glm::vec4(cam ? cam->getPosition() : glm::vec3(2.0f, 2.0f, 2.0f), 1.0f);
    ubo.exposure                = 1.0f;
    ubo.gamma                   = 2.2f;
    ubo.prefilteredCubeMipLevels = 1.0f;
    ubo.scaleIBLAmbient         = 0.02f;

    memcpy(renderable->uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

void Renderer::createImage(uint32_t width, uint32_t height, vk::Format format,
                           vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                           vk::MemoryPropertyFlags properties,
                           vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory)
{
    vk::ImageCreateInfo imageInfo{
        .imageType     = vk::ImageType::e2D,
        .format        = format,
        .extent        = { width, height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = vk::SampleCountFlagBits::e1,
        .tiling        = tiling,
        .usage         = usage,
        .sharingMode   = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };
    image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = memRequirements.size,
        .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)
    };
    imageMemory = vk::raii::DeviceMemory(device, allocInfo);
    image.bindMemory(imageMemory, 0);
}

void Renderer::copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer,
                                 vk::raii::Image& image, uint32_t width, uint32_t height)
{
    vk::BufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .imageOffset       = { 0, 0, 0 },
        .imageExtent       = { width, height, 1 }
    };
    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

void Renderer::transitionImageLayout(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image,
                                     vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::ImageMemoryBarrier barrier{
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
        .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
        .image               = image,
        .subresourceRange    = { .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 }
    };

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
    {
        barrier.srcAccessMask = {};
        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
        sourceStage           = vk::PipelineStageFlagBits::eTopOfPipe;
        destinationStage      = vk::PipelineStageFlagBits::eTransfer;
    }
    else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
    {
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
        sourceStage           = vk::PipelineStageFlagBits::eTransfer;
        destinationStage      = vk::PipelineStageFlagBits::eFragmentShader;
    }
    else
    {
        throw std::invalid_argument("unsupported layout transition!");
    }

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, barrier);
}

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
        .srcStageMask        = src_stage_mask,
        .srcAccessMask       = src_access_mask,
        .dstStageMask        = dst_stage_mask,
        .dstAccessMask       = dst_access_mask,
        .oldLayout           = old_layout,
        .newLayout           = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = {
            .aspectMask     = image_aspect_flags,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1
        }
    };
    vk::DependencyInfo dependency_info = {
        .dependencyFlags        = {},
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers   = &barrier
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
    createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

void Renderer::loadTextureFromFile(const std::string& filepath,
                                   vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory,
                                   vk::raii::ImageView& imageView, bool isSRGB)
{
    std::ifstream file(filepath);
    if (!file.good())
    {
        std::cout << "Warning: Texture file not found: " << filepath << " - using placeholder" << std::endl;
        return;
    }
    file.close();

    std::string extension = filepath.substr(filepath.find_last_of('.') + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    uint32_t       texWidth, texHeight;
    vk::DeviceSize imageSize;
    unsigned char* textureData = nullptr;
    int            texChannels;

    if (extension == "ktx")
    {
        ktxTexture* kTexture;
        KTX_error_code result = ktxTexture_CreateFromNamedFile(
            filepath.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &kTexture);

        if (result != KTX_SUCCESS)
        {
            std::cout << "Warning: Failed to load KTX texture: " << filepath << std::endl;
            return;
        }

        texWidth   = kTexture->baseWidth;
        texHeight  = kTexture->baseHeight;
        imageSize  = ktxTexture_GetImageSize(kTexture, 0);
        auto* ktxData = ktxTexture_GetData(kTexture);

        textureData = new unsigned char[imageSize];
        memcpy(textureData, ktxData, imageSize);
        ktxTexture_Destroy(kTexture);
    }
    else if (extension == "png" || extension == "jpg" || extension == "jpeg")
    {
        int texWidth_i, texHeight_i;
        textureData = stbi_load(filepath.c_str(), &texWidth_i, &texHeight_i, &texChannels, STBI_rgb_alpha);

        if (!textureData)
        {
            std::cout << "Warning: Failed to load image texture: " << filepath << std::endl;
            return;
        }

        texWidth  = static_cast<uint32_t>(texWidth_i);
        texHeight = static_cast<uint32_t>(texHeight_i);
        imageSize = texWidth * texHeight * 4;
    }
    else
    {
        std::cout << "Warning: Unsupported texture format: " << extension << " for file: " << filepath << std::endl;
        return;
    }

    vk::raii::Buffer       stagingBuffer({});
    vk::raii::DeviceMemory stagingBufferMemory({});
    createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 stagingBuffer, stagingBufferMemory);

    void* data = stagingBufferMemory.mapMemory(0, imageSize);
    memcpy(data, textureData, imageSize);
    stagingBufferMemory.unmapMemory();

    if (extension == "ktx")
        delete[] textureData;
    else
        stbi_image_free(textureData);

    vk::Format textureFormat = isSRGB ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm;

    createImage(texWidth, texHeight, textureFormat, vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                vk::MemoryPropertyFlagBits::eDeviceLocal, image, imageMemory);

    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    transitionImageLayout(commandBuffer, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(commandBuffer, stagingBuffer, image, texWidth, texHeight);
    transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
    endSingleTimeCommands(std::move(commandBuffer));

    imageView = createImageView(*image, textureFormat, vk::ImageAspectFlagBits::eColor);

    std::cout << "Successfully loaded texture: " << filepath << " (" << texWidth << "x" << texHeight << ")" << std::endl;
}

void Renderer::loadPBRTextures(const Material& material, RenderableComponent::PBRTextures& textures)
{
    std::cout << "Loading PBR textures for material: " << material.GetName() << std::endl;

    if (!material.albedoTexturePath.empty())
        loadTextureFromFile(material.albedoTexturePath, textures.baseColorImage, textures.baseColorMemory, textures.baseColorView, true);

    if (!material.metallicRoughnessTexturePath.empty())
        loadTextureFromFile(material.metallicRoughnessTexturePath, textures.metallicRoughnessImage, textures.metallicRoughnessMemory, textures.metallicRoughnessView, false);

    if (!material.normalTexturePath.empty())
        loadTextureFromFile(material.normalTexturePath, textures.normalImage, textures.normalMemory, textures.normalView, false);

    if (!material.occlusionTexturePath.empty())
        loadTextureFromFile(material.occlusionTexturePath, textures.occlusionImage, textures.occlusionMemory, textures.occlusionView, false);

    if (!material.emissiveTexturePath.empty())
        loadTextureFromFile(material.emissiveTexturePath, textures.emissiveImage, textures.emissiveMemory, textures.emissiveView, true);
}

void Renderer::createDefaultTextures()
{
    // White 1x1
    {
        const uint32_t white     = 0xFFFFFFFF;
        vk::DeviceSize imageSize = sizeof(uint32_t);

        vk::raii::Buffer       stagingBuffer({});
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

    // Flat normal 1x1
    {
        const uint32_t flatNormal = 0xFFFF7F7F;
        vk::DeviceSize imageSize  = sizeof(uint32_t);

        vk::raii::Buffer       stagingBuffer({});
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

void Renderer::createTextureSampler()
{
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{
        .magFilter               = vk::Filter::eLinear,
        .minFilter               = vk::Filter::eLinear,
        .mipmapMode              = vk::SamplerMipmapMode::eLinear,
        .addressModeU            = vk::SamplerAddressMode::eRepeat,
        .addressModeV            = vk::SamplerAddressMode::eRepeat,
        .addressModeW            = vk::SamplerAddressMode::eRepeat,
        .anisotropyEnable        = vk::True,
        .maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
        .compareEnable           = vk::False,
        .compareOp               = vk::CompareOp::eAlways,
        .borderColor             = vk::BorderColor::eIntOpaqueBlack,
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
        { .type = vk::DescriptorType::eUniformBuffer, .descriptorCount = materialCount * MAX_FRAMES_IN_FLIGHT },
        { .type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 5 * materialCount * MAX_FRAMES_IN_FLIGHT }
    } };

    vk::DescriptorPoolCreateInfo poolInfo{
        .flags       = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets     = materialCount * MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSize.size()),
        .pPoolSizes  = poolSize.data()
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
            auto& materialTextures          = gameObject.materialTextures[materialIndex];
            auto& descriptorSetsForMaterial = gameObject.materialDescriptorSets[materialIndex];

            std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *descriptorSetLayout);
            vk::DescriptorSetAllocateInfo allocInfo{
                .descriptorPool     = *descriptorPool,
                .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
                .pSetLayouts        = layouts.data()
            };

            descriptorSetsForMaterial = device.allocateDescriptorSets(allocInfo);

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            {
                vk::DescriptorBufferInfo bufferInfo{
                    .buffer = *gameObject.uniformBuffers[i],
                    .offset = 0,
                    .range  = sizeof(UniformBufferObject)
                };

                vk::ImageView baseColorView        = (*materialTextures.baseColorView != VK_NULL_HANDLE)        ? vk::ImageView(*materialTextures.baseColorView)        : vk::ImageView(*defaultTextureView);
                vk::ImageView metallicRoughnessView = (*materialTextures.metallicRoughnessView != VK_NULL_HANDLE) ? vk::ImageView(*materialTextures.metallicRoughnessView) : vk::ImageView(*defaultTextureView);
                vk::ImageView normalView            = (*materialTextures.normalView != VK_NULL_HANDLE)            ? vk::ImageView(*materialTextures.normalView)            : vk::ImageView(*defaultNormalView);
                vk::ImageView occlusionView         = (*materialTextures.occlusionView != VK_NULL_HANDLE)         ? vk::ImageView(*materialTextures.occlusionView)         : vk::ImageView(*defaultTextureView);
                vk::ImageView emissiveView          = (*materialTextures.emissiveView != VK_NULL_HANDLE)          ? vk::ImageView(*materialTextures.emissiveView)          : vk::ImageView(*defaultTextureView);

                vk::DescriptorImageInfo baseColorInfo       { .sampler = *textureSampler, .imageView = baseColorView,        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                vk::DescriptorImageInfo metallicRoughnessInfo{ .sampler = *textureSampler, .imageView = metallicRoughnessView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                vk::DescriptorImageInfo normalInfo           { .sampler = *textureSampler, .imageView = normalView,            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                vk::DescriptorImageInfo occlusionInfo        { .sampler = *textureSampler, .imageView = occlusionView,         .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
                vk::DescriptorImageInfo emissiveInfo         { .sampler = *textureSampler, .imageView = emissiveView,          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

                std::array descriptorWrites{
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer,        .pBufferInfo = &bufferInfo },
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo  = &baseColorInfo },
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo  = &metallicRoughnessInfo },
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 3, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo  = &normalInfo },
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 4, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo  = &occlusionInfo },
                    vk::WriteDescriptorSet{ .dstSet = *descriptorSetsForMaterial[i], .dstBinding = 5, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo  = &emissiveInfo }
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
    pushConstants.baseColorFactor              = material.baseColorFactor;
    pushConstants.metallicFactor               = material.metallicFactor;
    pushConstants.roughnessFactor              = material.roughnessFactor;
    pushConstants.baseColorTextureSet          = material.baseColorTextureIndex;
    pushConstants.physicalDescriptorTextureSet = material.metallicRoughnessTextureIndex;
    pushConstants.normalTextureSet             = material.normalTextureIndex;
    pushConstants.occlusionTextureSet          = material.occlusionTextureIndex;
    pushConstants.emissiveTextureSet           = material.emissiveTextureIndex;
    pushConstants.alphaMask                    = 0.0f;
    pushConstants.alphaMaskCutoff              = material.alphaCutoff;

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

    device.resetFences(*inFlightFences[frameIndex]);
    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    queue.waitIdle();

    vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    const vk::SubmitInfo submitInfo{
        .waitSemaphoreCount   = 1,
        .pWaitSemaphores      = &*presentCompleteSemaphores[frameIndex],
        .pWaitDstStageMask    = &waitDestinationStageMask,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &*commandBuffers[frameIndex],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores    = &*renderFinishedSemaphores[imageIndex]
    };
    queue.submit(submitInfo, *inFlightFences[frameIndex]);

    const vk::PresentInfoKHR presentInfoKHR{
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &*renderFinishedSemaphores[imageIndex],
        .swapchainCount     = 1,
        .pSwapchains        = &*swapChain,
        .pImageIndices      = &imageIndex
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
        loadModel(modelPath, *renderable);
        for (size_t i = 0; i < renderable->materials.size(); ++i)
            loadPBRTextures(renderable->materials[i], renderable->materialTextures[i]);

        return entity;
    };

    makeEntity("FlightHelmet_Left",
               { -3.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 3.0f, 3.0f, 3.0f },
               "../models/FlightHelmet.gltf");

    {
        Entity& e = makeEntity("DamagedHelmet",
                               { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.75f, 0.75f },
                               "../models/DamagedHelmet.gltf");
        e.GetComponent<RenderableComponent>()->materials[0].metallicFactor = 1.0f;
    }

    makeEntity("Sponza",
               { 0.0f, 0.0f, 0.0f }, { 0.0f, glm::radians(-45.0f), 0.0f }, { 3.0f, 3.0f, 3.0f },
               "../models/Sponza.gltf");
}

void Renderer::loadModel(std::string modelFilename, RenderableComponent& gameObj)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err;
    std::string        warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, modelFilename);

    if (!warn.empty())
        std::cout << "glTF warning: " << warn << std::endl;
    if (!err.empty())
        std::cout << "glTF error: " << err << std::endl;
    if (!ret)
        throw std::runtime_error("Failed to load glTF model");

    gameObj.vertices.clear();
    gameObj.indices.clear();
    gameObj.meshes.clear();
    gameObj.materials.clear();
    gameObj.materialTextures.clear();
    gameObj.materialDescriptorSets.clear();

    std::string baseDir = modelFilename.substr(0, modelFilename.find_last_of("/\\") + 1);

    for (const auto& mat : model.materials)
    {
        Material material(mat.name.empty() ? "Material" : mat.name);

        if (mat.pbrMetallicRoughness.baseColorFactor.size() == 4)
        {
            material.baseColorFactor = glm::vec4(
                mat.pbrMetallicRoughness.baseColorFactor[0],
                mat.pbrMetallicRoughness.baseColorFactor[1],
                mat.pbrMetallicRoughness.baseColorFactor[2],
                mat.pbrMetallicRoughness.baseColorFactor[3]);
        }

        material.metallicFactor  = static_cast<float>(mat.pbrMetallicRoughness.metallicFactor);
        material.roughnessFactor = static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor);

        auto resolveTexture = [&](int texIndex, std::string& outPath, float& outIndex) {
            if (texIndex >= 0 && texIndex < static_cast<int>(model.textures.size()))
            {
                int imageIndex = model.textures[texIndex].source;
                if (imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size()))
                {
                    outPath  = baseDir + model.images[imageIndex].uri;
                    outIndex = 0.0f;
                }
            }
        };

        resolveTexture(mat.pbrMetallicRoughness.baseColorTexture.index,       material.albedoTexturePath,             material.baseColorTextureIndex);
        resolveTexture(mat.pbrMetallicRoughness.metallicRoughnessTexture.index, material.metallicRoughnessTexturePath, material.metallicRoughnessTextureIndex);
        resolveTexture(mat.normalTexture.index,    material.normalTexturePath,    material.normalTextureIndex);
        resolveTexture(mat.occlusionTexture.index, material.occlusionTexturePath, material.occlusionTextureIndex);
        resolveTexture(mat.emissiveTexture.index,  material.emissiveTexturePath,  material.emissiveTextureIndex);

        std::cout << "Loaded material: " << material.GetName() << std::endl;
        std::cout << "  Base color texture: " << material.albedoTexturePath << std::endl;
        std::cout << "  Metallic-roughness texture: " << material.metallicRoughnessTexturePath << std::endl;
        std::cout << "  Normal texture: " << material.normalTexturePath << std::endl;

        gameObj.materials.push_back(std::move(material));
        gameObj.materialTextures.emplace_back();
    }

    if (gameObj.materials.empty())
    {
        gameObj.materials.emplace_back("DefaultMaterial");
        gameObj.materialTextures.emplace_back();
    }

    for (const auto& mesh : model.meshes)
    {
        for (const auto& primitive : mesh.primitives)
        {
            Mesh meshInfo;
            meshInfo.firstIndex   = static_cast<uint32_t>(gameObj.indices.size());
            meshInfo.materialIndex = primitive.material >= 0 ? primitive.material : 0;

            const tinygltf::Accessor&   indexAccessor   = model.accessors[primitive.indices];
            const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer&     indexBuffer     = model.buffers[indexBufferView.buffer];

            const tinygltf::Accessor&   posAccessor   = model.accessors[primitive.attributes.at("POSITION")];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer&     posBuffer     = model.buffers[posBufferView.buffer];

            bool hasNormals  = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            bool hasTangents = primitive.attributes.find("TANGENT") != primitive.attributes.end();

            const tinygltf::Accessor*   texCoordAccessor  = nullptr;
            const tinygltf::BufferView* texCoordBufferView = nullptr;
            const tinygltf::Buffer*     texCoordBuffer     = nullptr;

            const tinygltf::Accessor*   normalAccessor  = nullptr;
            const tinygltf::BufferView* normalBufferView = nullptr;
            const tinygltf::Buffer*     normalBuffer     = nullptr;

            const tinygltf::Accessor*   tangentAccessor  = nullptr;
            const tinygltf::BufferView* tangentBufferView = nullptr;
            const tinygltf::Buffer*     tangentBuffer     = nullptr;

            if (hasTexCoords)
            {
                texCoordAccessor  = &model.accessors[primitive.attributes.at("TEXCOORD_0")];
                texCoordBufferView = &model.bufferViews[texCoordAccessor->bufferView];
                texCoordBuffer     = &model.buffers[texCoordBufferView->buffer];
            }

            if (hasNormals)
            {
                normalAccessor  = &model.accessors[primitive.attributes.at("NORMAL")];
                normalBufferView = &model.bufferViews[normalAccessor->bufferView];
                normalBuffer     = &model.buffers[normalBufferView->buffer];
            }

            if (hasTangents)
            {
                tangentAccessor  = &model.accessors[primitive.attributes.at("TANGENT")];
                tangentBufferView = &model.bufferViews[tangentAccessor->bufferView];
                tangentBuffer     = &model.buffers[tangentBufferView->buffer];
            }

            uint32_t baseVertex = static_cast<uint32_t>(gameObj.vertices.size());

            for (size_t i = 0; i < posAccessor.count; i++)
            {
                Vertex vertex{};

                const float* pos = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset + i * 12]);
                vertex.pos       = { pos[0], -pos[1], pos[2] };

                if (hasTexCoords)
                {
                    const float* texCoord = reinterpret_cast<const float*>(&texCoordBuffer->data[texCoordBufferView->byteOffset + texCoordAccessor->byteOffset + i * 8]);
                    vertex.texCoord       = { texCoord[0], texCoord[1] };
                }
                else
                {
                    vertex.texCoord = { 0.0f, 0.0f };
                }

                if (hasNormals)
                {
                    const float* normal = reinterpret_cast<const float*>(&normalBuffer->data[normalBufferView->byteOffset + normalAccessor->byteOffset + i * 12]);
                    vertex.normal       = { normal[0], -normal[1], normal[2] };
                }
                else
                {
                    vertex.normal = { 0.0f, 0.0f, 0.0f };
                }

                if (hasTangents)
                {
                    const float* tangent = reinterpret_cast<const float*>(&tangentBuffer->data[tangentBufferView->byteOffset + tangentAccessor->byteOffset + i * 16]);
                    vertex.tangent       = { tangent[0], -tangent[1], tangent[2], tangent[3] };
                }
                else
                {
                    vertex.tangent = { 0.0f, 0.0f, 0.0f, 1.0f };
                }

                gameObj.vertices.push_back(vertex);
            }

            const unsigned char* indexData   = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            size_t               indexCount  = indexAccessor.count;
            size_t               indexStride = 0;

            if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                indexStride = sizeof(uint16_t);
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                indexStride = sizeof(uint32_t);
            else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                indexStride = sizeof(uint8_t);
            else
                throw std::runtime_error("Unsupported index component type");

            gameObj.indices.reserve(gameObj.indices.size() + indexCount);

            for (size_t i = 0; i < indexCount; i++)
            {
                uint32_t index = 0;

                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    index = *reinterpret_cast<const uint16_t*>(indexData + i * indexStride);
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                    index = *reinterpret_cast<const uint32_t*>(indexData + i * indexStride);
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                    index = *reinterpret_cast<const uint8_t*>(indexData + i * indexStride);

                gameObj.indices.push_back(baseVertex + index);
            }

            meshInfo.indexCount = static_cast<uint32_t>(gameObj.indices.size() - meshInfo.firstIndex);
            gameObj.meshes.push_back(meshInfo);
        }
    }
}
