#include "imgui_vulkan_util.h"

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
		device->waitIdle();
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
	vulkanStyle.Colors[ImGuiCol_TitleBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.6f);
	vulkanStyle.Colors[ImGuiCol_TitleBgActive] = ImVec4(1.0f, 0.0f, 0.0f, 0.8f);
	vulkanStyle.Colors[ImGuiCol_MenuBarBg] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	vulkanStyle.Colors[ImGuiCol_Header] = ImVec4(1.0f, 0.0f, 0.0f, 0.4f);
	vulkanStyle.Colors[ImGuiCol_CheckMark] = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);

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
	createImage(1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal, fontImage, fontImageMemory);

	// Create image view for shader access
	// The image view defines how shaders interpret the raw image data
	createImageView(*fontImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);

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
	vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eCombinedImageSampler, 1 };

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;     // Allow individual descriptor set freeing
	poolInfo.maxSets = 2;                                                      // Maximum number of descriptor sets
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

	// Create the graphics pipeline with dynamic rendering
	// ... (shader loading, pipeline state setup, etc.)

	// For brevity, we're omitting the full pipeline creation code here
	// In a real implementation, you would:
	// 1. Load the vertex and fragment shaders
	// 2. Set up all the pipeline state (vertex input, input assembly, rasterization, etc.)
	// 3. Include the renderingInfo in the pipeline creation to enable dynamic rendering
}


bool ImGuiVulkanUtil::newFrame() {
	// Start a new ImGui frame
	ImGui::NewFrame();

	// Create your UI elements here
	// For example:
	ImGui::Begin("Vulkan ImGui Demo");
	ImGui::Text("Hello, Vulkan!");
	if (ImGui::Button("Click me!")) {
		// Handle button click
	}
	ImGui::End();

	// End the frame
	ImGui::EndFrame();

	// Render to generate draw data
	ImGui::Render();

	// Check if buffers need updating
	ImDrawData* drawData = ImGui::GetDrawData();
	if (drawData && drawData->CmdListsCount > 0) {
		if (drawData->TotalVtxCount > vertexCount || drawData->TotalIdxCount > indexCount) {
			needsUpdateBuffers = true;
			return true;
		}
	}

	return false;
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
