#include "imgui_vulkan_util.h"
#include <fstream>
#include <stdexcept>

std::vector<char> ImGuiVulkanUtil::readFile(const std::string& filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("failed to open file: " + filename);
	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	return buffer;
}

vk::raii::ShaderModule ImGuiVulkanUtil::createShaderModule(const std::vector<char>& code)
{
	vk::ShaderModuleCreateInfo createInfo{
		.codeSize = code.size(),
		.pCode    = reinterpret_cast<const uint32_t*>(code.data())
	};
	return vk::raii::ShaderModule(*device, createInfo);
}

ImGuiVulkanUtil::ImGuiVulkanUtil(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice,
	vk::raii::Queue& graphicsQueue, uint32_t graphicsQueueFamily)
	: device(&device), physicalDevice(&physicalDevice),
	graphicsQueue(&graphicsQueue), graphicsQueueFamily(graphicsQueueFamily)
{
	vk::CommandPoolCreateInfo poolInfo{ .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
										.queueFamilyIndex = graphicsQueueFamily };
	commandPool = this->device->createCommandPool(poolInfo);

	// Set up dynamic rendering info
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachmentFormats = &colorFormat;
}

ImGuiVulkanUtil::~ImGuiVulkanUtil() {
	// Wait for device to finish operations before destroying resources
	// NOTE: waitIdle() is acceptable in destructors/cleanup code but should NEVER be used
	// in the main rendering loop as it causes severe performance issues. For frame
	// synchronization, use fences and semaphores instead.
	if (device) {
     try {
			device->waitIdle();
		}
		catch (...) {
			// Best-effort teardown: ignore device-lost or other errors during destruction.
		}
	}

	// All resources are automatically cleaned up by their destructors
	// No manual cleanup needed

	// ImGui context is destroyed separately
}

void ImGuiVulkanUtil::init(float width, float height) {
	// Initialize ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Configure ImGui
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable keyboard controls
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;      // Enable docking

	// Set display size
	io.DisplaySize = ImVec2(width, height);
	io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

 // Set up style
	vulkanStyle = ImGui::GetStyle();
	vulkanStyle.WindowRounding = 5.0f;
	vulkanStyle.ChildRounding = 4.0f;
	vulkanStyle.FrameRounding = 4.0f;
	vulkanStyle.PopupRounding = 4.0f;
	vulkanStyle.GrabRounding = 4.0f;
	vulkanStyle.ScrollbarRounding = 6.0f;
	vulkanStyle.TabRounding = 4.0f;
	vulkanStyle.WindowBorderSize = 1.0f;
	vulkanStyle.FrameBorderSize = 0.0f;
	vulkanStyle.Colors[ImGuiCol_Text] = ImVec4(0.95f, 0.96f, 0.98f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.54f, 0.60f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_WindowBg] = ImVec4(0.10f, 0.11f, 0.13f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ChildBg] = ImVec4(0.11f, 0.12f, 0.15f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_PopupBg] = ImVec4(0.12f, 0.13f, 0.16f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_Border] = ImVec4(0.26f, 0.29f, 0.35f, 0.70f);
	vulkanStyle.Colors[ImGuiCol_FrameBg] = ImVec4(0.16f, 0.18f, 0.22f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.21f, 0.24f, 0.30f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.25f, 0.29f, 0.36f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_TitleBg] = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.12f, 0.14f, 0.18f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.12f, 0.14f, 0.18f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.28f, 0.31f, 0.37f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.34f, 0.38f, 0.45f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.40f, 0.45f, 0.53f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_CheckMark] = ImVec4(0.41f, 0.65f, 0.95f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_SliderGrab] = ImVec4(0.35f, 0.57f, 0.90f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.45f, 0.69f, 0.98f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_Button] = ImVec4(0.19f, 0.22f, 0.28f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.29f, 0.37f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_ButtonActive] = ImVec4(0.30f, 0.35f, 0.44f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_Header] = ImVec4(0.18f, 0.22f, 0.30f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.25f, 0.31f, 0.40f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_HeaderActive] = ImVec4(0.31f, 0.37f, 0.47f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_Separator] = ImVec4(0.23f, 0.26f, 0.32f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.17f, 0.22f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_TabHovered] = ImVec4(0.24f, 0.29f, 0.37f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_TabSelected] = ImVec4(0.20f, 0.25f, 0.33f, 1.00f);
	vulkanStyle.Colors[ImGuiCol_DockingPreview] = ImVec4(0.30f, 0.50f, 0.86f, 0.55f);
	vulkanStyle.Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.08f, 0.09f, 0.11f, 1.00f);

	// Apply default style
	setStyle(0);
}

void ImGuiVulkanUtil::setStyle(uint32_t index) {
	ImGuiStyle& style = ImGui::GetStyle();

	switch (index) {
	case 0:
		// Custom Vulkan style
		style = vulkanStyle;
		break;
	case 1:
		// Classic style
		ImGui::StyleColorsClassic();
		break;
	case 2:
		// Dark style
		ImGui::StyleColorsDark();
		break;
	case 3:
		// Light style
		ImGui::StyleColorsLight();
		break;
	}
}

void ImGuiVulkanUtil::createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory)
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
	image = vk::raii::Image(*device, imageInfo);

	vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{
		.allocationSize = memRequirements.size,
		.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties) };
	imageMemory = vk::raii::DeviceMemory(*device, allocInfo);
	image.bindMemory(imageMemory, 0);
}

vk::raii::ImageView ImGuiVulkanUtil::createImageView(vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
	vk::ImageViewCreateInfo viewInfo{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1} };
	return vk::raii::ImageView(*device, viewInfo);
}


void ImGuiVulkanUtil::initResources() {
	// Extract font atlas data from ImGui's internal font system
	// ImGui generates a texture atlas containing all glyphs needed for text rendering
	ImGuiIO& io = ImGui::GetIO();
	unsigned char* fontData;                    // Raw pixel data from font atlas
	int texWidth, texHeight;                    // Dimensions of the generated font atlas
	io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);

	// Calculate total memory requirements for GPU transfer
	// Each pixel contains 4 bytes (RGBA) requiring precise memory allocation
	vk::DeviceSize uploadSize = texWidth * texHeight * 4 * sizeof(char);
	// Define image dimensions and create extent structure
	// Vulkan requires explicit specification of all image dimensions
	vk::Extent3D fontExtent{
		static_cast<uint32_t>(texWidth),        // Image width in pixels
		static_cast<uint32_t>(texHeight),       // Image height in pixels
		1                                       // Single layer (not a 3D texture or array)
	};

	// Create optimized GPU image for font texture storage
	// This image will be sampled by shaders during UI rendering
	createImage(static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, fontImage, fontImageMemory);

	// Create image view for shader access
	// The image view defines how shaders interpret the raw image data
	fontImageView = createImageView(*fontImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);

	vk::raii::Buffer       stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	createBuffer(uploadSize, vk::BufferUsageFlagBits::eTransferSrc,
				 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				 stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, uploadSize);
	memcpy(dataStaging, fontData, uploadSize);
	stagingBufferMemory.unmapMemory();

	vk::raii::CommandBuffer cmd = beginSingleTimeCommands();
	transitionImageLayout(cmd, fontImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(cmd, stagingBuffer, fontImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
	transitionImageLayout(cmd, fontImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	endSingleTimeCommands(std::move(cmd));

	// Configure texture sampling parameters for optimal text rendering
// These settings directly impact text quality and performance
	vk::SamplerCreateInfo samplerInfo{};
	samplerInfo.magFilter = vk::Filter::eLinear;                    // Smooth scaling when magnified
	samplerInfo.minFilter = vk::Filter::eLinear;                    // Smooth scaling when minified
	samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;        // Smooth transitions between mip levels
	samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;  // Prevent texture wrapping
	samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;  // Clean edge handling
	samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;  // 3D consistency
	samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;   // White border for clamped areas

	sampler = device->createSampler(samplerInfo);                   // Create the GPU sampler object

	// Create descriptor pool for shader resource binding
	// Descriptors provide the interface between shaders and GPU resources
    vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eCombinedImageSampler, 64 };

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;     // Allow individual descriptor set freeing
 poolInfo.maxSets = 64;                                                     // Maximum number of descriptor sets
	poolInfo.poolSizeCount = 1;                                                // Number of pool size specifications
	poolInfo.pPoolSizes = &poolSize;                                           // Pool size configuration

	descriptorPool = device->createDescriptorPool(poolInfo);                   // Create descriptor pool

	// Create descriptor set layout defining shader resource interface
	// This layout must match the binding declarations in the ImGui shaders
	vk::DescriptorSetLayoutBinding binding{};
	binding.descriptorType = vk::DescriptorType::eCombinedImageSampler;        // Combined texture and sampler
	binding.descriptorCount = 1;                                               // Single texture binding
	binding.stageFlags = vk::ShaderStageFlagBits::eFragment;                   // Used in fragment shader
	binding.binding = 0;                                                       // Shader binding point 0

	vk::DescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.bindingCount = 1;                                               // Number of bindings in layout
	layoutInfo.pBindings = &binding;                                           // Binding configuration array

	descriptorSetLayout = device->createDescriptorSetLayout(layoutInfo);       // Create layout object

	// Allocate descriptor set from pool using the defined layout
	// This creates the actual binding that connects GPU resources to shaders
	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.descriptorPool = *descriptorPool;                                // Source pool for allocation
	allocInfo.descriptorSetCount = 1;                                          // Number of sets to allocate
	vk::DescriptorSetLayout layouts[] = { *descriptorSetLayout };                // Layout template array
	allocInfo.pSetLayouts = layouts;                                           // Layout configuration

	descriptorSet = std::move(device->allocateDescriptorSets(allocInfo).front()); // Allocate and store set

	// Update descriptor set with actual font texture and sampler resources
	// This final step connects the physical GPU resources to the shader binding points
	vk::DescriptorImageInfo imageInfo{};
	imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;           // Expected image layout
	imageInfo.imageView = fontImageView;                           // Font texture view
	imageInfo.sampler = *sampler;                                              // Texture sampler

	vk::WriteDescriptorSet writeSet{};
	writeSet.dstSet = *descriptorSet;                                          // Target descriptor set
	writeSet.descriptorCount = 1;                                              // Number of resources to bind
	writeSet.descriptorType = vk::DescriptorType::eCombinedImageSampler;       // Resource type
	writeSet.pImageInfo = &imageInfo;                                          // Image resource information
	writeSet.dstBinding = 0;                                                   // Binding point in shader

	device->updateDescriptorSets(writeSet, {});                   // Execute the binding update
	// Create pipeline cache
	vk::PipelineCacheCreateInfo pipelineCacheInfo{};
	pipelineCache = device->createPipelineCache(pipelineCacheInfo);

	// Create pipeline layout
	vk::PushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eVertex;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(PushConstBlock);

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.setLayoutCount = 1;
	vk::DescriptorSetLayout setLayouts[] = { *descriptorSetLayout };
	pipelineLayoutInfo.pSetLayouts = setLayouts;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

	pipelineLayout = device->createPipelineLayout(pipelineLayoutInfo);

	// Load the compiled imgui shader (vertex + fragment entry points in one SPIR-V blob)
	auto shaderCode = readFile("shaders/imgui.spv");
	vk::raii::ShaderModule shaderModule = createShaderModule(shaderCode);

	vk::PipelineShaderStageCreateInfo vertStageInfo{
		.stage  = vk::ShaderStageFlagBits::eVertex,
		.module = shaderModule,
		.pName  = "vertMain"
	};
	vk::PipelineShaderStageCreateInfo fragStageInfo{
		.stage  = vk::ShaderStageFlagBits::eFragment,
		.module = shaderModule,
		.pName  = "fragMain"
	};
	vk::PipelineShaderStageCreateInfo shaderStages[] = { vertStageInfo, fragStageInfo };

	// ImDrawVert: pos(float2), uv(float2), col(uint32 packed as R8G8B8A8_UNORM)
	vk::VertexInputBindingDescription bindingDesc{
		.binding   = 0,
		.stride    = sizeof(ImDrawVert),
		.inputRate = vk::VertexInputRate::eVertex
	};
	std::array<vk::VertexInputAttributeDescription, 3> attrDescs{{
		{ .location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat,  .offset = offsetof(ImDrawVert, pos) },
		{ .location = 1, .binding = 0, .format = vk::Format::eR32G32Sfloat,  .offset = offsetof(ImDrawVert, uv)  },
		{ .location = 2, .binding = 0, .format = vk::Format::eR8G8B8A8Unorm, .offset = offsetof(ImDrawVert, col) }
	}};

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
		.vertexBindingDescriptionCount   = 1,
		.pVertexBindingDescriptions      = &bindingDesc,
		.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()),
		.pVertexAttributeDescriptions    = attrDescs.data()
	};

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{ .topology = vk::PrimitiveTopology::eTriangleList };

	vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

	vk::PipelineRasterizationStateCreateInfo rasterizer{
		.depthClampEnable        = vk::False,
		.rasterizerDiscardEnable = vk::False,
		.polygonMode             = vk::PolygonMode::eFill,
		.cullMode                = vk::CullModeFlagBits::eNone,
		.frontFace               = vk::FrontFace::eCounterClockwise,
		.depthBiasEnable         = vk::False,
		.lineWidth               = 1.0f
	};

	vk::PipelineMultisampleStateCreateInfo multisampling{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
		.sampleShadingEnable  = vk::False
	};

	// Alpha blending for UI transparency
	vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		.blendEnable         = vk::True,
		.srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
		.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		.colorBlendOp        = vk::BlendOp::eAdd,
		.srcAlphaBlendFactor = vk::BlendFactor::eOne,
		.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
		.alphaBlendOp        = vk::BlendOp::eAdd,
		.colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
							   vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
	};

	vk::PipelineColorBlendStateCreateInfo colorBlending{
		.logicOpEnable   = vk::False,
		.attachmentCount = 1,
		.pAttachments    = &colorBlendAttachment
	};

	// No depth testing for UI overlay
	vk::PipelineDepthStencilStateCreateInfo depthStencil{
		.depthTestEnable  = vk::False,
		.depthWriteEnable = vk::False,
		.stencilTestEnable = vk::False
	};

	std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
	vk::PipelineDynamicStateCreateInfo dynamicState{
		.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
		.pDynamicStates    = dynamicStates.data()
	};

	// Chain pipeline create info with dynamic rendering info (no render pass needed)
	vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineChain = {
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
		  .layout              = *pipelineLayout,
		  .renderPass          = nullptr },
		{ .colorAttachmentCount    = 1,
		  .pColorAttachmentFormats = &colorFormat }
	};

	pipeline = vk::raii::Pipeline(*device, nullptr, pipelineChain.get<vk::GraphicsPipelineCreateInfo>());
}


bool ImGuiVulkanUtil::newFrame() {
	// Start a new ImGui frame
	ImGui::NewFrame();

	return true;
}

ImTextureID ImGuiVulkanUtil::registerTexture(vk::ImageView imageView, vk::Sampler textureSampler, vk::ImageLayout imageLayout)
{
	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.descriptorPool = *descriptorPool;
	allocInfo.descriptorSetCount = 1;
	vk::DescriptorSetLayout layout = *descriptorSetLayout;
	allocInfo.pSetLayouts = &layout;

	auto allocatedSets = device->allocateDescriptorSets(allocInfo);
	userTextureDescriptorSets.push_back(std::move(allocatedSets.front()));
	vk::raii::DescriptorSet& userSet = userTextureDescriptorSets.back();

	vk::DescriptorImageInfo imageInfo{};
	imageInfo.sampler = textureSampler;
	imageInfo.imageView = imageView;
	imageInfo.imageLayout = imageLayout;

	vk::WriteDescriptorSet writeSet{};
	writeSet.dstSet = *userSet;
	writeSet.dstBinding = 0;
	writeSet.descriptorCount = 1;
	writeSet.descriptorType = vk::DescriptorType::eCombinedImageSampler;
	writeSet.pImageInfo = &imageInfo;

	device->updateDescriptorSets(writeSet, {});
	return reinterpret_cast<ImTextureID>(static_cast<VkDescriptorSet>(*userSet));
}

void ImGuiVulkanUtil::clearRegisteredTextures()
{
	userTextureDescriptorSets.clear();
}

void ImGuiVulkanUtil::updateBuffers() {
	ImDrawData* drawData = ImGui::GetDrawData();
	if (!drawData || drawData->CmdListsCount == 0) {
		return;
	}

	// Calculate required buffer sizes
	vk::DeviceSize vertexBufferSize = drawData->TotalVtxCount * sizeof(ImDrawVert);
	vk::DeviceSize indexBufferSize = drawData->TotalIdxCount * sizeof(ImDrawIdx);

	// Resize buffers if needed
	if (drawData->TotalVtxCount > static_cast<int>(vertexCount)) {
		createBuffer(vertexBufferSize,
					 vk::BufferUsageFlagBits::eVertexBuffer,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 vertexBuffer, vertexBufferMemory);
		vertexCount = drawData->TotalVtxCount;
	}

	if (drawData->TotalIdxCount > static_cast<int>(indexCount)) {
		createBuffer(indexBufferSize,
					 vk::BufferUsageFlagBits::eIndexBuffer,
					 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
					 indexBuffer, indexBufferMemory);
		indexCount = drawData->TotalIdxCount;
	}

	// Upload data to buffers
	ImDrawVert* vtxDst = static_cast<ImDrawVert*>(vertexBufferMemory.mapMemory(0, vertexBufferSize));
	ImDrawIdx*  idxDst = static_cast<ImDrawIdx*>(indexBufferMemory.mapMemory(0, indexBufferSize));

	for (int n = 0; n < drawData->CmdListsCount; n++) {
		const ImDrawList* cmdList = drawData->CmdLists[n];
		memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
		memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));
		vtxDst += cmdList->VtxBuffer.Size;
		idxDst += cmdList->IdxBuffer.Size;
	}

	vertexBufferMemory.unmapMemory();
	indexBufferMemory.unmapMemory();
}

void ImGuiVulkanUtil::drawFrame(vk::raii::CommandBuffer& commandBuffer, vk::ImageView swapchainImageView) {
	ImDrawData* drawData = ImGui::GetDrawData();
	if (!drawData || drawData->CmdListsCount == 0) {
		return;
	}

	// Begin dynamic rendering
	vk::RenderingAttachmentInfo colorAttachment{};
	colorAttachment.imageView   = swapchainImageView;
	colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	colorAttachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	colorAttachment.storeOp     = vk::AttachmentStoreOp::eStore;

	vk::RenderingInfo renderingInfo{};
	renderingInfo.renderArea = vk::Rect2D{ {0, 0}, {static_cast<uint32_t>(drawData->DisplaySize.x),
												   static_cast<uint32_t>(drawData->DisplaySize.y)} };
	renderingInfo.layerCount = 1;
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	commandBuffer.beginRendering(renderingInfo);

	// Bind the pipeline used for ImGui
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);

	// Configure viewport for UI pixel coordinates
	vk::Viewport viewport{};
	viewport.width = drawData->DisplaySize.x;
	viewport.height = drawData->DisplaySize.y;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;
	commandBuffer.setViewport(0, viewport);

	// Convert from ImGui coordinates into NDC via a simple scale/translate
	pushConstBlock.scale = glm::vec2(2.0f / drawData->DisplaySize.x, 2.0f / drawData->DisplaySize.y);
	pushConstBlock.translate = glm::vec2(-1.0f);

	commandBuffer.pushConstants<PushConstBlock>(
		*pipelineLayout,                    // Must be dereferenced if using vk::raii::PipelineLayout
		vk::ShaderStageFlagBits::eVertex,   // Or appropriate stage flags
		0,                                   // offset
		pushConstBlock                             // Your data
	);

	// We already filled these buffers this frame
	vk::Buffer vertexBuffers[] = { vertexBuffer };
	vk::DeviceSize offsets[] = { 0 };
	commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
	commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);

	int vertexOffset = 0;
	int indexOffset = 0;

	for (int i = 0; i < drawData->CmdListsCount; i++) {
		const ImDrawList* cmdList = drawData->CmdLists[i];

		for (int j = 0; j < cmdList->CmdBuffer.Size; j++) {
			const ImDrawCmd* pcmd = &cmdList->CmdBuffer[j];
			if (pcmd->UserCallback) {
				pcmd->UserCallback(cmdList, pcmd);
				continue;
			}

			// Clip per draw call
			vk::Rect2D scissor{};
			scissor.offset.x = std::max(static_cast<int32_t>(pcmd->ClipRect.x), 0);
			scissor.offset.y = std::max(static_cast<int32_t>(pcmd->ClipRect.y), 0);
			scissor.extent.width = static_cast<uint32_t>(pcmd->ClipRect.z - pcmd->ClipRect.x);
			scissor.extent.height = static_cast<uint32_t>(pcmd->ClipRect.w - pcmd->ClipRect.y);
			commandBuffer.setScissor(0, scissor);

            // Bind texture referenced by this draw command (font or user-registered)
			vk::DescriptorSet textureSet = pcmd->GetTexID()
				? reinterpret_cast<VkDescriptorSet>(pcmd->GetTexID())
				: static_cast<VkDescriptorSet>(*descriptorSet);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                *pipelineLayout, 0, textureSet, {});

			// Issue indexed draw for this UI batch
			commandBuffer.drawIndexed(pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
			indexOffset += pcmd->ElemCount;
		}

		vertexOffset += cmdList->VtxBuffer.Size;
	}

	// Close the rendering scope for the UI overlay
	commandBuffer.endRendering();
}

void ImGuiVulkanUtil::handleKey(int key, int scancode, int action, int mods) {
	ImGuiIO& io = ImGui::GetIO();

	// This example uses GLFW key codes and actions, but you can adapt this
	// to work with any windowing library's input system

	// Map the platform-specific key action to ImGui's key state
	// In GLFW: GLFW_PRESS = 1, GLFW_RELEASE = 0
	const int KEY_PRESSED = 1;  // Generic key pressed value
	const int KEY_RELEASED = 0; // Generic key released value

	//if (action == KEY_PRESSED)
	//	io.KeysDown[key] = true;
	//if (action == KEY_RELEASED)
	//	io.KeysDown[key] = false;

	//// Update modifier keys
	//// These key codes are GLFW-specific, but you would use your windowing library's
	//// equivalent key codes for other libraries
	//const int KEY_LEFT_CTRL = 341;   // GLFW_KEY_LEFT_CONTROL
	//const int KEY_RIGHT_CTRL = 345;  // GLFW_KEY_RIGHT_CONTROL
	//const int KEY_LEFT_SHIFT = 340;  // GLFW_KEY_LEFT_SHIFT
	//const int KEY_RIGHT_SHIFT = 344; // GLFW_KEY_RIGHT_SHIFT
	//const int KEY_LEFT_ALT = 342;    // GLFW_KEY_LEFT_ALT
	//const int KEY_RIGHT_ALT = 346;   // GLFW_KEY_RIGHT_ALT
	//const int KEY_LEFT_SUPER = 343;  // GLFW_KEY_LEFT_SUPER
	//const int KEY_RIGHT_SUPER = 347; // GLFW_KEY_RIGHT_SUPER

	//io.KeyCtrl = io.KeysDown[KEY_LEFT_CTRL] || io.KeysDown[KEY_RIGHT_CTRL];
	//io.KeyShift = io.KeysDown[KEY_LEFT_SHIFT] || io.KeysDown[KEY_RIGHT_SHIFT];
	//io.KeyAlt = io.KeysDown[KEY_LEFT_ALT] || io.KeysDown[KEY_RIGHT_ALT];
	//io.KeySuper = io.KeysDown[KEY_LEFT_SUPER] || io.KeysDown[KEY_RIGHT_SUPER];
}

bool ImGuiVulkanUtil::getWantKeyCapture() {
	return ImGui::GetIO().WantCaptureKeyboard;
}

void ImGuiVulkanUtil::charPressed(uint32_t key) {
	ImGuiIO& io = ImGui::GetIO();
	io.AddInputCharacter(key);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

uint32_t ImGuiVulkanUtil::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
	vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice->getMemoryProperties();
	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			return i;
	}
	throw std::runtime_error("failed to find suitable memory type!");
}

void ImGuiVulkanUtil::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
								   vk::MemoryPropertyFlags properties,
								   vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory)
{
	vk::BufferCreateInfo bufferInfo{ .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive };
	buffer = vk::raii::Buffer(*device, bufferInfo);
	vk::MemoryRequirements memReq = buffer.getMemoryRequirements();
	vk::MemoryAllocateInfo allocInfo{ .allocationSize  = memReq.size,
									  .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties) };
	bufferMemory = vk::raii::DeviceMemory(*device, allocInfo);
	buffer.bindMemory(*bufferMemory, 0);
}

void ImGuiVulkanUtil::copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size)
{
	vk::raii::CommandBuffer cmd = beginSingleTimeCommands();
	cmd.copyBuffer(*src, *dst, vk::BufferCopy{ .size = size });
	endSingleTimeCommands(std::move(cmd));
}

void ImGuiVulkanUtil::copyBufferToImage(vk::raii::CommandBuffer& commandBuffer, const vk::raii::Buffer& buffer,
										vk::raii::Image& image, uint32_t width, uint32_t height)
{
	vk::BufferImageCopy region{
		.bufferOffset      = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
		.imageSubresource  = { .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.imageOffset       = { 0, 0, 0 },
		.imageExtent       = { width, height, 1 }
	};
	commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

vk::raii::CommandBuffer ImGuiVulkanUtil::beginSingleTimeCommands()
{
	vk::CommandBufferAllocateInfo allocInfo{ .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1 };
	vk::raii::CommandBuffer cmd = std::move(vk::raii::CommandBuffers(*device, allocInfo).front());
	cmd.begin(vk::CommandBufferBeginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
	return cmd;
}

void ImGuiVulkanUtil::endSingleTimeCommands(vk::raii::CommandBuffer&& cmd)
{
	cmd.end();
	vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmd };
	graphicsQueue->submit(submitInfo, nullptr);
	graphicsQueue->waitIdle();
}
