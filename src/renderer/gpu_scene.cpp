#include "gpu_scene.h"

#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

#include "../ECS/components/renderable_component.h"
#include "../ECS/components/transform_component.h"
#include "../vulkan/utils.h"

// ============================================================================
// One-shot command buffer helpers
// ============================================================================

static vk::raii::CommandBuffer beginSingleShot(vk::raii::Device& device, vk::raii::CommandPool& pool)
{
    vk::CommandBufferAllocateInfo ai{
        .commandPool        = *pool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto bufs = device.allocateCommandBuffers(ai);
    vk::raii::CommandBuffer cb = std::move(bufs[0]);
    cb.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    return cb;
}

static void endSingleShot(vk::raii::CommandBuffer& cb, vk::raii::Queue& queue)
{
    cb.end();
    vk::SubmitInfo si{ .commandBufferCount = 1, .pCommandBuffers = &*cb };
    queue.submit(si, nullptr);
    queue.waitIdle();
}

// ============================================================================
// uploadViaStaging — copies host data to a device-local buffer
// ============================================================================
void GPUScene::uploadViaStaging(vk::raii::Device&          device,
                                vk::raii::PhysicalDevice&  physicalDevice,
                                vk::raii::Queue&           queue,
                                vk::raii::CommandPool&     commandPool,
                                const void*                data,
                                vk::DeviceSize             size,
                                vk::raii::Buffer&          dstBuffer,
                                vk::raii::DeviceMemory&    dstMemory,
                                vk::BufferUsageFlags       extraUsage)
{
    // Destination (device-local)
    {
        vk::BufferCreateInfo bi{ .size = size,
            .usage       = vk::BufferUsageFlagBits::eTransferDst | extraUsage,
            .sharingMode = vk::SharingMode::eExclusive };
        dstBuffer = vk::raii::Buffer(device, bi);

        auto mr = dstBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        dstMemory = vk::raii::DeviceMemory(device, mai);
        dstBuffer.bindMemory(*dstMemory, 0);
    }

    // Staging (host-visible)
    vk::raii::Buffer       staging    = nullptr;
    vk::raii::DeviceMemory stagingMem = nullptr;
    {
        vk::BufferCreateInfo bi{ .size = size,
            .usage       = vk::BufferUsageFlagBits::eTransferSrc,
            .sharingMode = vk::SharingMode::eExclusive };
        staging = vk::raii::Buffer(device, bi);

        auto mr = staging.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        stagingMem = vk::raii::DeviceMemory(device, mai);
        staging.bindMemory(*stagingMem, 0);
    }

    void* mapped = stagingMem.mapMemory(0, size);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    stagingMem.unmapMemory();

    auto cb = beginSingleShot(device, commandPool);
    cb.copyBuffer(*staging, *dstBuffer, vk::BufferCopy{ .size = size });
    endSingleShot(cb, queue);
}

// ============================================================================
// computeLocalBoundingSphere (Ritter approximation in mesh-local space)
// ============================================================================
glm::vec4 GPUScene::computeLocalBoundingSphere(const std::vector<Vertex>& verts,
                                               uint32_t                   firstVertex,
                                               uint32_t                   vertexCount)
{
    if (vertexCount == 0)
        return glm::vec4(0.0f);

    glm::vec3 centre(0.0f);
    for (uint32_t i = 0; i < vertexCount; ++i)
        centre += verts[firstVertex + i].pos;
    centre /= static_cast<float>(vertexCount);

    float radius = 0.0f;
    for (uint32_t i = 0; i < vertexCount; ++i)
        radius = glm::max(radius, glm::length(verts[firstVertex + i].pos - centre));

    return glm::vec4(centre, radius);
}

// ============================================================================
// createBindlessDescriptors
// ============================================================================
void GPUScene::createBindlessDescriptors(vk::raii::Device& device)
{
    vk::DescriptorSetLayoutBinding binding{
        .binding         = 0,
        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = GPU_MAX_BINDLESS_TEXTURES,
        .stageFlags      = vk::ShaderStageFlagBits::eFragment
    };

    vk::DescriptorBindingFlags bindingFlags =
        vk::DescriptorBindingFlagBits::ePartiallyBound |
        vk::DescriptorBindingFlagBits::eUpdateAfterBind;

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        .bindingCount  = 1,
        .pBindingFlags = &bindingFlags
    };

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .pNext        = &flagsInfo,
        .flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        .bindingCount = 1,
        .pBindings    = &binding
    };
    bindlessLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::DescriptorPoolSize poolSize{
        .type            = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = GPU_MAX_BINDLESS_TEXTURES
    };
    vk::DescriptorPoolCreateInfo poolInfo{
        .flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize
    };
    bindlessPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool     = *bindlessPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &*bindlessLayout
    };
    auto sets   = device.allocateDescriptorSets(allocInfo);
    bindlessSet = std::move(sets[0]);
}

uint32_t GPUScene::registerTexture(vk::raii::Device& device,
                                   vk::ImageView     imageView,
                                   vk::Sampler       sampler)
{
    uint32_t idx = nextBindlessIdx++;
    vk::DescriptorImageInfo imgInfo{
        .sampler     = sampler,
        .imageView   = imageView,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
    };
    vk::WriteDescriptorSet write{
        .dstSet          = *bindlessSet,
        .dstBinding      = 0,
        .dstArrayElement = idx,
        .descriptorCount = 1,
        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo      = &imgInfo
    };
    device.updateDescriptorSets(write, nullptr);
    return idx;
}

// ============================================================================
// createSSBOs
// ============================================================================
void GPUScene::createSSBOs(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice)
{
    const vk::DeviceSize objSize = sizeof(ObjectData)   * GPU_MAX_DRAWS;
    const vk::DeviceSize matSize = sizeof(MaterialData) * GPU_MAX_DRAWS;

    // Object data — host-visible (updated each frame for moving objects)
    {
        vk::BufferCreateInfo bi{ .size = objSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer };
        objectDataBuffer = vk::raii::Buffer(device, bi);
        auto mr = objectDataBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        objectDataMemory = vk::raii::DeviceMemory(device, mai);
        objectDataBuffer.bindMemory(*objectDataMemory, 0);
        objectDataMapped = objectDataMemory.mapMemory(0, objSize);
    }

    // Material data — device-local (immutable after build)
    {
        vk::BufferCreateInfo bi{ .size = matSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eTransferDst };
        materialDataBuffer = vk::raii::Buffer(device, bi);
        auto mr = materialDataBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        materialDataMemory = vk::raii::DeviceMemory(device, mai);
        materialDataBuffer.bindMemory(*materialDataMemory, 0);
    }
}

// ============================================================================
// createCullBuffers
// ============================================================================
void GPUScene::createCullBuffers(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice)
{
    const vk::DeviceSize cmdSize = sizeof(vk::DrawIndexedIndirectCommand) * GPU_MAX_DRAWS;

    // Cull output — device-local, written by compute, read by indirect draw
    {
        vk::BufferCreateInfo bi{ .size = cmdSize,
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eIndirectBuffer |
                     vk::BufferUsageFlagBits::eTransferDst };
        cullOutputBuffer = vk::raii::Buffer(device, bi);
        auto mr = cullOutputBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        cullOutputMemory = vk::raii::DeviceMemory(device, mai);
        cullOutputBuffer.bindMemory(*cullOutputMemory, 0);
    }

    // Draw count — device-local atomic counter (fillBuffer'd to 0 each frame)
    {
        vk::BufferCreateInfo bi{ .size = sizeof(uint32_t),
            .usage = vk::BufferUsageFlagBits::eStorageBuffer |
                     vk::BufferUsageFlagBits::eIndirectBuffer |
                     vk::BufferUsageFlagBits::eTransferDst |
                     vk::BufferUsageFlagBits::eTransferSrc };
        drawCountBuffer = vk::raii::Buffer(device, bi);
        auto mr = drawCountBuffer.getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eDeviceLocal)
        };
        drawCountMemory = vk::raii::DeviceMemory(device, mai);
        drawCountBuffer.bindMemory(*drawCountMemory, 0);
    }

    // Readback buffers — host-visible, one per frame-in-flight
    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i)
    {
        vk::BufferCreateInfo bi{ .size = sizeof(uint32_t),
            .usage = vk::BufferUsageFlagBits::eTransferDst };
        drawCountReadbackBuffers[i] = vk::raii::Buffer(device, bi);
        auto mr = drawCountReadbackBuffers[i].getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        drawCountReadbackMemories[i] = vk::raii::DeviceMemory(device, mai);
        drawCountReadbackBuffers[i].bindMemory(*drawCountReadbackMemories[i], 0);
        drawCountReadbackMapped[i] = drawCountReadbackMemories[i].mapMemory(0, sizeof(uint32_t));
        // Initialise to 0 so the display doesn't show garbage before the first frame.
        *static_cast<uint32_t*>(drawCountReadbackMapped[i]) = 0;
    }
}

// ============================================================================
// createFrameUBOs
// ============================================================================
void GPUScene::createFrameUBOs(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice)
{
    for (uint32_t i = 0; i < GPU_FRAMES_IN_FLIGHT; ++i)
    {
        vk::BufferCreateInfo bi{ .size = sizeof(FrameData),
            .usage = vk::BufferUsageFlagBits::eUniformBuffer };
        frameUBOBuffers[i] = vk::raii::Buffer(device, bi);

        auto mr = frameUBOBuffers[i].getMemoryRequirements();
        vk::MemoryAllocateInfo mai{
            .allocationSize  = mr.size,
            .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        frameUBOMemories[i] = vk::raii::DeviceMemory(device, mai);
        frameUBOBuffers[i].bindMemory(*frameUBOMemories[i], 0);
        frameUBOMapped[i] = frameUBOMemories[i].mapMemory(0, sizeof(FrameData));
    }
}

// ============================================================================
// build — single-pass scene build
// ============================================================================
void GPUScene::build(entt::registry&           registry,
                     vk::raii::Device&          device,
                     vk::raii::PhysicalDevice&  physicalDevice,
                     vk::raii::Queue&           queue,
                     vk::raii::CommandPool&     commandPool,
                     vk::raii::Sampler&         defaultSampler,
                     vk::ImageView              defaultTextureView,
                     vk::ImageView              defaultNormalView)
{
    valid = false;
    nextBindlessIdx = 0;
    vertices.clear();
    indices.clear();
    drawCommands.clear();
    objectDataCPU.clear();
    materialDataCPU.clear();
    entityRanges.clear();

    // 1. Bindless descriptor infrastructure (must exist before registerTexture)
    createBindlessDescriptors(device);

    // 2. Single-pass: geometry + materials + draw commands
    for (auto [entity, renderable, transform] : registry.view<RenderableComponent, TransformComponent>().each())
    {
        if (renderable.meshes.empty() || renderable.vertices.empty()) continue;

        EntityDrawRange range;
        range.entity              = entity;
        range.firstDraw           = static_cast<uint32_t>(drawCommands.size());
        range.drawCount           = 0;
        range.localBoundingSphere = computeLocalBoundingSphere(
            renderable.vertices, 0, static_cast<uint32_t>(renderable.vertices.size()));

        const uint32_t globalVertexBase = static_cast<uint32_t>(vertices.size());
        const uint32_t globalIndexBase  = static_cast<uint32_t>(indices.size());

        // Append geometry
        vertices.insert(vertices.end(), renderable.vertices.begin(), renderable.vertices.end());
        for (uint32_t idx : renderable.indices)
            indices.push_back(idx); // vertexOffset in the draw command handles the base

        // Build one MaterialData per material slot and register its textures
        const uint32_t materialBase = static_cast<uint32_t>(materialDataCPU.size());
        for (uint32_t mi = 0; mi < static_cast<uint32_t>(renderable.materials.size()); ++mi)
        {
            const Material& mat = renderable.materials[mi];
            MaterialData md{};
            md.baseColorFactor  = mat.baseColorFactor;
            md.metallicFactor   = mat.metallicFactor;
            md.roughnessFactor  = mat.roughnessFactor;
            md.alphaMask        = (mat.alphaMode == "MASK") ? 1.0f : 0.0f;
            md.alphaMaskCutoff  = mat.alphaCutoff;

            // Helper: register a texture view or fall back to the default
            auto regTex = [&](bool hasTexture, vk::ImageView view, vk::ImageView fallback) -> int32_t
            {
                vk::ImageView chosen = (hasTexture && view) ? view : fallback;
                if (!chosen) return -1;
                return static_cast<int32_t>(registerTexture(device, chosen, *defaultSampler));
            };

            if (mi < renderable.materialTextures.size())
            {
                const auto& tex = renderable.materialTextures[mi];
                md.baseColorTexIdx = regTex(
                    mat.baseColorTextureIndex >= 0.0f, *tex.baseColorView, defaultTextureView);
                md.metallicRoughnessTexIdx = regTex(
                    mat.metallicRoughnessTextureIndex >= 0.0f, *tex.metallicRoughnessView, VK_NULL_HANDLE);
                md.normalTexIdx = regTex(
                    mat.normalTextureIndex >= 0.0f, *tex.normalView, defaultNormalView);
                md.occlusionTexIdx = regTex(
                    mat.occlusionTextureIndex >= 0.0f, *tex.occlusionView, VK_NULL_HANDLE);
                md.emissiveTexIdx = regTex(
                    mat.emissiveTextureIndex >= 0.0f, *tex.emissiveView, VK_NULL_HANDLE);
            }
            else
            {
                md.baseColorTexIdx = regTex(false, VK_NULL_HANDLE, defaultTextureView);
                md.metallicRoughnessTexIdx = -1;
                md.normalTexIdx            = regTex(false, VK_NULL_HANDLE, defaultNormalView);
                md.occlusionTexIdx         = -1;
                md.emissiveTexIdx          = -1;
            }

            materialDataCPU.push_back(md);
        }

        // Build one draw command + object data per submesh
        const glm::mat4 model = transform.GetTransformMatrix();
        glm::vec3 lCentre = glm::vec3(range.localBoundingSphere);
        float     lRadius = range.localBoundingSphere.w;
        glm::vec3 wCentre = glm::vec3(model * glm::vec4(lCentre, 1.0f));
        float sx = glm::length(glm::vec3(model[0]));
        float sy = glm::length(glm::vec3(model[1]));
        float sz = glm::length(glm::vec3(model[2]));
        float wRadius = lRadius * glm::max(glm::max(sx, sy), sz);

        for (const auto& mesh : renderable.meshes)
        {
            uint32_t matIdx = materialBase +
                (mesh.materialIndex < static_cast<uint32_t>(renderable.materials.size())
                    ? mesh.materialIndex : 0u);

            ObjectData obj{};
            obj.modelMatrix    = model;
            obj.boundingSphere = glm::vec4(wCentre, wRadius);
            obj.materialIndex  = matIdx;

            // firstInstance carries the object index — vertex shader uses SV_StartInstanceLocation
            vk::DrawIndexedIndirectCommand cmd{
                .indexCount    = mesh.indexCount,
                .instanceCount = 1,
                .firstIndex    = globalIndexBase + mesh.firstIndex,
                .vertexOffset  = static_cast<int32_t>(globalVertexBase),
                .firstInstance = static_cast<uint32_t>(objectDataCPU.size())
            };

            objectDataCPU.push_back(obj);
            drawCommands.push_back(cmd);
            ++range.drawCount;
        }

        entityRanges.push_back(range);
    }

    if (drawCommands.empty()) return;

    // 3. Create GPU buffers
    createSSBOs(device, physicalDevice);
    createCullBuffers(device, physicalDevice);
    createFrameUBOs(device, physicalDevice);

    // 4. Upload immutable geometry
    uploadViaStaging(device, physicalDevice, queue, commandPool,
        vertices.data(), vertices.size() * sizeof(Vertex),
        globalVertexBuffer, globalVertexMemory,
        vk::BufferUsageFlagBits::eVertexBuffer);

    uploadViaStaging(device, physicalDevice, queue, commandPool,
        indices.data(), indices.size() * sizeof(uint32_t),
        globalIndexBuffer, globalIndexMemory,
        vk::BufferUsageFlagBits::eIndexBuffer);

    uploadViaStaging(device, physicalDevice, queue, commandPool,
        drawCommands.data(),
        drawCommands.size() * sizeof(vk::DrawIndexedIndirectCommand),
        drawCommandBuffer, drawCommandMemory,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc);

    // 5. Upload material data via staging (materialDataBuffer is device-local)
    {
        const vk::DeviceSize matSize = materialDataCPU.size() * sizeof(MaterialData);
        vk::raii::Buffer       staging    = nullptr;
        vk::raii::DeviceMemory stagingMem = nullptr;
        {
            vk::BufferCreateInfo bi{ .size = matSize, .usage = vk::BufferUsageFlagBits::eTransferSrc };
            staging = vk::raii::Buffer(device, bi);
            auto mr = staging.getMemoryRequirements();
            vk::MemoryAllocateInfo mai{
                .allocationSize  = mr.size,
                .memoryTypeIndex = findMemoryType(physicalDevice, mr.memoryTypeBits,
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
            };
            stagingMem = vk::raii::DeviceMemory(device, mai);
            staging.bindMemory(*stagingMem, 0);
        }
        void* m = stagingMem.mapMemory(0, matSize);
        std::memcpy(m, materialDataCPU.data(), static_cast<size_t>(matSize));
        stagingMem.unmapMemory();
        auto cb = beginSingleShot(device, commandPool);
        cb.copyBuffer(*staging, *materialDataBuffer, vk::BufferCopy{ .size = matSize });
        endSingleShot(cb, queue);
    }

    // 6. Initial object data upload
    flushObjectData();

    valid = true;
}

// ============================================================================
// updateObjectTransforms — refresh model matrices from ECS each frame
// ============================================================================
void GPUScene::updateObjectTransforms(entt::registry& registry)
{
    for (auto& range : entityRanges)
    {
        if (!registry.valid(range.entity)) continue;

        auto* transform = registry.try_get<TransformComponent>(range.entity);
        if (!transform) continue;

        const glm::mat4 model  = transform->GetTransformMatrix();
        const glm::vec3 lCtr   = glm::vec3(range.localBoundingSphere);
        const float     lRad   = range.localBoundingSphere.w;
        const glm::vec3 wCtr   = glm::vec3(model * glm::vec4(lCtr, 1.0f));
        const float sx = glm::length(glm::vec3(model[0]));
        const float sy = glm::length(glm::vec3(model[1]));
        const float sz = glm::length(glm::vec3(model[2]));
        const float wRad = lRad * glm::max(glm::max(sx, sy), sz);

        for (uint32_t d = 0; d < range.drawCount; ++d)
        {
            const uint32_t objIdx = drawCommands[range.firstDraw + d].firstInstance;
            if (objIdx < static_cast<uint32_t>(objectDataCPU.size()))
            {
                objectDataCPU[objIdx].modelMatrix    = model;
                objectDataCPU[objIdx].boundingSphere = glm::vec4(wCtr, wRad);
            }
        }
    }
}

// ============================================================================
// flushObjectData — copy to GPU
// ============================================================================
void GPUScene::flushObjectData()
{
    if (objectDataMapped && !objectDataCPU.empty())
        std::memcpy(objectDataMapped, objectDataCPU.data(), objectDataCPU.size() * sizeof(ObjectData));
}

// ============================================================================
// updateFrameData
// ============================================================================
void GPUScene::updateFrameData(uint32_t frameIndex, const FrameData& data)
{
    std::memcpy(frameUBOMapped[frameIndex], &data, sizeof(FrameData));
}

// ============================================================================
// destroy
// ============================================================================
void GPUScene::destroy()
{
    valid          = false;
    objectDataMapped = nullptr;
    for (auto& m : frameUBOMapped) m = nullptr;
    for (auto& m : drawCountReadbackMapped) m = nullptr;
    lastVisibleDrawCount = 0;
    nextBindlessIdx = 0;
}
