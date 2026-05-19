#include "renderer/renderer.h"

#include "vulkan/material.h"
#include "pass_common.h"

void Renderer::recordShadowPass(vk::raii::CommandBuffer& commandBuffer, uint32_t cascadeIndex)
{
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    vk::RenderingAttachmentInfo depthAttachmentInfo = {
    .imageView = shadowImageViews[cascadeIndex],
    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eStore,
    .clearValue = clearDepth
    };

    vk::RenderingInfo renderingInfo = {
        .renderArea = {.offset = { 0, 0 }, .extent = vk::Extent2D(2048, 2048)},
        .layerCount = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments = nullptr,
        .pDepthAttachment = &depthAttachmentInfo
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline);
    commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(2048), static_cast<float>(2048), 0.0f, 1.0f));
    commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(2048, 2048)));

    int cascadeIndexInt = static_cast<int>(cascadeIndex);
    commandBuffer.pushConstants<int>(*shadowPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, cascadeIndexInt);

    auto& registry = mEnttScene.getRegistry();
    std::array<glm::vec4, MAX_POINT_LIGHTS> pointLightPositions{};
    std::array<glm::vec4, MAX_POINT_LIGHTS> pointLightColors{};
    
	collectPointLights(registry, pointLightPositions, pointLightColors);

    int lightIndex = 0;
    for (auto [lightEntity, light, lightTransform] : registry.view<PointLightComponent, TransformComponent>().each())
    {
        (void)lightEntity;
        if (lightIndex >= static_cast<int>(MAX_POINT_LIGHTS))
            break;
        if (!light.enabled)
            continue;

        pointLightPositions[lightIndex] = glm::vec4(lightTransform.GetPosition(), 1.0f);
        pointLightColors[lightIndex] = glm::vec4(light.color * light.intensity, 1.0f);
        ++lightIndex;
    }
    for (; lightIndex < static_cast<int>(MAX_POINT_LIGHTS); ++lightIndex) {
        pointLightColors[lightIndex] = glm::vec4(0.0f);
    }

    for (auto [entity, renderable, transform] : registry.view<RenderableComponent, TransformComponent>().each())
    {
        (void)entity;

        UniformBuffer::updateUniformBuffer(frameIndex, &renderable, &transform, &camera, swapChainExtent, shadowSettings, pointLightPositions, pointLightColors);
        vk::Buffer     vertexBuffers[] = { renderable.vertexBuffer };
        vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(*renderable.indexBuffer, 0, vk::IndexType::eUint32);

        for (const auto& mesh : renderable.meshes)
        {
            uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable.materialDescriptorSets.size() ? mesh.materialIndex : 0;

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *shadowPipelineLayout,
                0,
                *renderable.materialDescriptorSets[descriptorMaterialIndex][frameIndex],
                nullptr);

            commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
        }
    }
}

void Renderer::recordColliderDebugPass(vk::raii::CommandBuffer& commandBuffer)
{
    if (!physicsDrawColliderDebug || *colliderDebugPipeline == VK_NULL_HANDLE)
        return;

    rebuildColliderDebugLines();
    if (colliderDebugVertices.empty())
        return;

    ensureColliderDebugVertexCapacity(colliderDebugVertices.size());

    void* mapped = colliderDebugVertexBufferMemory.mapMemory(
        0, colliderDebugVertices.size() * sizeof(DebugLineVertex));
    std::memcpy(mapped, colliderDebugVertices.data(),
        colliderDebugVertices.size() * sizeof(DebugLineVertex));
    colliderDebugVertexBufferMemory.unmapMemory();

    const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
    DebugLinePushConstants pc{};
    pc.viewProj = camera.getProjectionMatrix(aspect, 0.1f, 3000.0f) * camera.getViewMatrix();

    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *colliderDebugPipeline);
    commandBuffer.pushConstants<DebugLinePushConstants>(
        *colliderDebugPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pc);

    vk::Buffer vb = *colliderDebugVertexBuffer;
    vk::DeviceSize off = 0;
    commandBuffer.bindVertexBuffers(0, vb, off);
    commandBuffer.draw(static_cast<uint32_t>(colliderDebugVertices.size()), 1, 0, 0);
}

void Renderer::recordScenePass(vk::raii::CommandBuffer& commandBuffer)
{
    auto& registry = mEnttScene.getRegistry();
    std::array<glm::vec4, MAX_POINT_LIGHTS> pointLightPositions{};
    std::array<glm::vec4, MAX_POINT_LIGHTS> pointLightColors{};
	collectPointLights(registry, pointLightPositions, pointLightColors);

    int lightIndex = 0;
    for (auto [lightEntity, light, lightTransform] : registry.view<PointLightComponent, TransformComponent>().each())
    {
        (void)lightEntity;
        if (lightIndex >= static_cast<int>(MAX_POINT_LIGHTS))
            break;
        if (!light.enabled)
            continue;

        pointLightPositions[lightIndex] = glm::vec4(lightTransform.GetPosition(), 1.0f);
        pointLightColors[lightIndex] = glm::vec4(light.color * light.intensity, 1.0f);
        ++lightIndex;
    }

    for (; lightIndex < static_cast<int>(MAX_POINT_LIGHTS); ++lightIndex) {
        pointLightColors[lightIndex] = glm::vec4(0.0f);
    }

    for (auto [ecsEntity, renderable, transform] : registry.view<RenderableComponent, TransformComponent>().each())
    {
        (void)ecsEntity;

        UniformBuffer::updateUniformBuffer(frameIndex, &renderable, &transform, &camera, swapChainExtent, shadowSettings, pointLightPositions, pointLightColors);
        vk::Buffer     vertexBuffers[] = { renderable.vertexBuffer };
        vk::DeviceSize offsets[] = { 0 };
        commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffer.bindIndexBuffer(*renderable.indexBuffer, 0, vk::IndexType::eUint32);

        for (const auto& mesh : renderable.meshes)
        {
            const Material& material = renderable.materials[mesh.materialIndex < renderable.materials.size() ? mesh.materialIndex : 0];
            MaterialPushConstants::push(commandBuffer, *pbrPipelineLayout, material);
            uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable.materialDescriptorSets.size() ? mesh.materialIndex : 0;

            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                *pbrPipelineLayout,
                0,
                *renderable.materialDescriptorSets[descriptorMaterialIndex][frameIndex],
                nullptr);

            commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
        }
    }

    recordAssimpSkinnedPass(commandBuffer);
}
