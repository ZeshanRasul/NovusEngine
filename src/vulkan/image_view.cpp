#include "image_view.h"

vk::raii::ImageView ImageView::createImageView(vk::raii::Device& device, vk::Image const& image, vk::Format format, vk::ImageAspectFlags aspectFlags)
{
	vk::ImageViewCreateInfo viewInfo{
		.image = image,
		.viewType = vk::ImageViewType::e2D,
		.format = format,
		.subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 } };
	return vk::raii::ImageView(device, viewInfo);
}
