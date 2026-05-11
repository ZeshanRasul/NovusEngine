#include "texture.h"
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include "command_buffer.h"
#include "buffer.h"

void Texture::loadTextureFromFile(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice, vk::raii::Queue& queue, vk::raii::CommandPool& commandPool, const std::string& filepath, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory, vk::raii::ImageView& imageView, bool isSRGB)
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

		texWidth = kTexture->baseWidth;
		texHeight = kTexture->baseHeight;
		imageSize = ktxTexture_GetImageSize(kTexture, 0);
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

		texWidth = static_cast<uint32_t>(texWidth_i);
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
	Buffer::createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
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

	// Delay staging buffer cleanup until command execution is complete
	// In a real application, you would queue these for deletion after idle
	// But for our simplified start-up use single time commands already waitIdle
	// Just make sure endSingleTimeCommands actually waits
	vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
	transitionImageLayout(commandBuffer, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
	copyBufferToImage(commandBuffer, stagingBuffer, image, texWidth, texHeight);
	transitionImageLayout(commandBuffer, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
	CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

	imageView = createImageView(*image, textureFormat, vk::ImageAspectFlagBits::eColor);

	std::cout << "Successfully loaded texture: " << filepath << " (" << texWidth << "x" << texHeight << ")" << std::endl;
}
