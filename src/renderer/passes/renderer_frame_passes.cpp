#include "renderer/renderer.h"

void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
    auto& commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});

    if (renderEnableShadows)
    {
        for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade)
        {
            transition_image_layout(*shadowImages[cascade],
                shadowImageLayouts[cascade],
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::AccessFlags2{},
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::PipelineStageFlagBits2::eTopOfPipe,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::ImageAspectFlagBits::eDepth);
            shadowImageLayouts[cascade] = vk::ImageLayout::eDepthAttachmentOptimal;

            recordShadowPass(commandBuffer, cascade);
            recordAssimpShadowPass(commandBuffer, cascade);
            commandBuffer.endRendering();

            transition_image_layout(*shadowImages[cascade],
                vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                vk::AccessFlagBits2::eShaderSampledRead,
                vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::ImageAspectFlagBits::eDepth);
            shadowImageLayouts[cascade] = vk::ImageLayout::eShaderReadOnlyOptimal;
        }
    }

    beginMainPass(commandBuffer, imageIndex);
    recordScenePass(commandBuffer);
    recordColliderDebugPass(commandBuffer);
    commandBuffer.endRendering();

    if (renderEnablePostProcessing)
    {
        if (renderEnableBloom && bloomEnabled)
            recordBloomPasses(commandBuffer);

        if (renderEnableFxaa)
            recordFxaaPass(commandBuffer, imageIndex);
        else
            recordSceneCopyPass(commandBuffer, imageIndex);
    }
    else
    {
        recordSceneCopyPass(commandBuffer, imageIndex);
    }

    recordImguiPass(commandBuffer, imageIndex);

    endMainPass(commandBuffer, imageIndex);

    commandBuffer.end();
}

void Renderer::beginMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    transition_image_layout(*fxaaImage,
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
        .imageView = fxaaImageView,
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
}

void Renderer::recordSceneCopyPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    transition_image_layout(*fxaaImage,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eTransferRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);

    vk::ImageCopy copyRegion{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .srcOffset = { 0, 0, 0 },
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
        .dstOffset = { 0, 0, 0 },
        .extent = { swapChainExtent.width, swapChainExtent.height, 1 }
    };
    commandBuffer.copyImage(*fxaaImage, vk::ImageLayout::eTransferSrcOptimal,
        swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
        copyRegion);

    transition_image_layout(*viewportPreviewImage,
        viewportPreviewImageLayout,
        vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);
    viewportPreviewImageLayout = vk::ImageLayout::eTransferDstOptimal;

    commandBuffer.copyImage(*fxaaImage, vk::ImageLayout::eTransferSrcOptimal,
        *viewportPreviewImage, vk::ImageLayout::eTransferDstOptimal,
        copyRegion);

    transition_image_layout(*viewportPreviewImage,
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    viewportPreviewImageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(*fxaaImage,
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferRead,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);

    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eTransferDstOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferWrite,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
}

void Renderer::endMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::ePresentSrcKHR,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        {},
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eBottomOfPipe,
        vk::ImageAspectFlagBits::eColor);
}
