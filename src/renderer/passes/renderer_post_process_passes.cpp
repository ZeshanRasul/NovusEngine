#include "renderer/renderer.h"

void Renderer::recordBloomPasses(vk::raii::CommandBuffer& commandBuffer)
{
    if (!bloomEnabled)
        return;

    transition_image_layout(*fxaaImage,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);

    transition_image_layout(*bloomImageA,
        bloomImageALayout,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    bloomImageALayout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::RenderingAttachmentInfo bloomAttachmentA{
        .imageView = bloomImageAView,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };

    vk::RenderingInfo bloomRenderInfoA{
     .renderArea = {.offset = { 0, 0 }, .extent = bloomExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &bloomAttachmentA
    };

    vk::RenderingInfo bloomRenderInfoB{
        .renderArea = {.offset = { 0, 0 }, .extent = bloomExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = nullptr
    };
    commandBuffer.beginRendering(bloomRenderInfoA);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomExtractPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
        static_cast<float>(bloomExtent.width),
        static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *bloomExtractPipelineLayout, 0,
        *bloomExtractDescriptorSets[frameIndex], nullptr);
    BloomExtractPushConstantsCPU extractPc{
          .threshold = bloomThreshold,
          .softKnee = bloomSoftKnee,
          .prefilterScale = bloomPrefilterScale
    };
    commandBuffer.pushConstants<BloomExtractPushConstantsCPU>(
        *bloomExtractPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        extractPc);
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(*bloomImageA,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eColor);
    bloomImageALayout = vk::ImageLayout::eShaderReadOnlyOptimal;

    transition_image_layout(*bloomImageB,
        bloomImageBLayout,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
    bloomImageBLayout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::RenderingAttachmentInfo bloomAttachmentB{
         .imageView = bloomImageBView,
         .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
         .loadOp = vk::AttachmentLoadOp::eClear,
         .storeOp = vk::AttachmentStoreOp::eStore,
         .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };
    bloomRenderInfoB.pColorAttachments = &bloomAttachmentB;

    const int blurPassPairs = std::max(1, bloomBlurPasses);
    for (int i = 0; i < blurPassPairs; ++i)
    {
        if (i > 0)
        {
            transition_image_layout(*bloomImageB,
                vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::ImageLayout::eColorAttachmentOptimal,
                vk::AccessFlagBits2::eShaderSampledRead,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::ImageAspectFlagBits::eColor);
            bloomImageBLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        commandBuffer.beginRendering(bloomRenderInfoB);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomBlurPipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
            static_cast<float>(bloomExtent.width),
            static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *bloomBlurPipelineLayout, 0,
            *bloomBlurFromADescriptorSets[frameIndex], nullptr);
        BloomBlurPushConstantsCPU blurPcX{
            .direction = {
                1.0f / static_cast<float>(bloomExtent.width),
                0.0f
            },
            .blurScale = bloomBlurScale
        };
        commandBuffer.pushConstants<BloomBlurPushConstantsCPU>(
            *bloomBlurPipelineLayout,
            vk::ShaderStageFlagBits::eFragment,
            0,
            blurPcX);
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();

        transition_image_layout(*bloomImageB,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor);
        bloomImageBLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

        transition_image_layout(*bloomImageA,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::ImageAspectFlagBits::eColor);
        bloomImageALayout = vk::ImageLayout::eColorAttachmentOptimal;

        commandBuffer.beginRendering(bloomRenderInfoA);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomBlurPipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
            static_cast<float>(bloomExtent.width),
            static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
            *bloomBlurPipelineLayout, 0,
            *bloomBlurFromBDescriptorSets[frameIndex], nullptr);
        BloomBlurPushConstantsCPU blurPcY{
            .direction = {
                0.0f,
                1.0f / static_cast<float>(bloomExtent.height)
            },
            .blurScale = bloomBlurScale
        };
        commandBuffer.pushConstants<BloomBlurPushConstantsCPU>(
            *bloomBlurPipelineLayout,
            vk::ShaderStageFlagBits::eFragment,
            0,
            blurPcY);
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();

        transition_image_layout(*bloomImageA,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::AccessFlagBits2::eShaderSampledRead,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eFragmentShader,
            vk::ImageAspectFlagBits::eColor);
        bloomImageALayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
}

void Renderer::recordFxaaPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    transition_image_layout(*depthImage,
        vk::ImageLayout::eDepthAttachmentOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits2::eShaderSampledRead,
        vk::PipelineStageFlagBits2::eLateFragmentTests,
        vk::PipelineStageFlagBits2::eFragmentShader,
        vk::ImageAspectFlagBits::eDepth);

    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);

    vk::RenderingAttachmentInfo colorAttachment{
        .imageView = swapChainImageViews[imageIndex],
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
    };

    vk::RenderingInfo renderingInfo{
        .renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *fxaaPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
        static_cast<float>(swapChainExtent.width),
        static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, swapChainExtent));

    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
        *fxaaPipelineLayout, 0,
        *fxaaDescriptorSets[frameIndex], nullptr);

    FxaaPushConstantsCPU pc{
        .rcpFrame = {
            1.0f / static_cast<float>(swapChainExtent.width),
            1.0f / static_cast<float>(swapChainExtent.height)
        },
        .exposure = fxaaExposure,
      .gamma = fxaaGamma,
      .bloomIntensity = bloomEnabled ? bloomIntensity : 0.0f,
        .debugMode = postProcessDebugMode
    };

    commandBuffer.pushConstants<FxaaPushConstantsCPU>(
        *fxaaPipelineLayout,
        vk::ShaderStageFlagBits::eFragment,
        0,
        pc
    );

    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transition_image_layout(*viewportPreviewImage,
        viewportPreviewImageLayout,
        vk::ImageLayout::eTransferDstOptimal,
        vk::AccessFlags2{},
        vk::AccessFlagBits2::eTransferWrite,
        vk::PipelineStageFlagBits2::eTopOfPipe,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::ImageAspectFlagBits::eColor);
    viewportPreviewImageLayout = vk::ImageLayout::eTransferDstOptimal;

    transition_image_layout(swapChainImages[imageIndex],
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
    commandBuffer.copyImage(swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal,
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

    transition_image_layout(swapChainImages[imageIndex],
        vk::ImageLayout::eTransferSrcOptimal,
        vk::ImageLayout::eColorAttachmentOptimal,
        vk::AccessFlagBits2::eTransferRead,
        vk::AccessFlagBits2::eColorAttachmentWrite,
        vk::PipelineStageFlagBits2::eTransfer,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        vk::ImageAspectFlagBits::eColor);
}

void Renderer::recordImguiPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
    imGui->drawFrame(commandBuffer, *swapChainImageViews[imageIndex]);
}
