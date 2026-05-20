// GPU-driven indirect rendering integration.
// COMPILE: add frustum_cull.comp.spv and pbr_indirect.spv to shaders/
// Implements:
//   - buildGPUScene()              : calls GPUScene::build, creates descriptor infrastructure
//   - buildIndirectPBRPipeline()   : indirect PBR graphics pipeline
//   - buildCullPipeline()          : compute frustum culling pipeline
//   - recordCullPass()             : dispatch culling compute
//   - recordIndirectScenePass()    : one vkCmdDrawIndexedIndirectCount for all static meshes
//   - buildIndirectFrameData()     : assemble FrameData UBO from camera/lights/shadows

#include "renderer/renderer.h"
#include "renderer/gpu_scene.h"
#include "renderer/passes/pass_common.h"
#include "vulkan/uniform_buffer.h"
#include "vulkan/depth_target.h"
#include "ECS/components/renderable_component.h"
#include "ECS/components/transform_component.h"
#include "ECS/components/light_component.h"

#include <fstream>
#include <iostream>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

// ---------------------------------------------------------------------------
// Shader loading helper
// ---------------------------------------------------------------------------
static std::vector<uint32_t> loadSPV(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open shader: " + path);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}

// ---------------------------------------------------------------------------
// buildGPUScene
// Merges all RenderableComponent entities into GPUScene and creates the
// indirect-pass descriptor infrastructure.
// ---------------------------------------------------------------------------
void Renderer::buildGPUScene()
{
    auto& registry = mEnttScene.getRegistry();

    // Skip if there are no static renderables
    bool hasRenderables = false;
    registry.view<RenderableComponent>().each([&](auto, auto&) { hasRenderables = true; });
    if (!hasRenderables) return;

    mGPUScene.build(registry, device, physicalDevice, queue, commandPool,
                    textureSampler,
                    *defaultTextureView,
                    *defaultNormalView);

    if (!mGPUScene.valid) return;

    // Descriptor sets (must precede pipeline layout creation)
    buildIndirectGlobalDescriptorSets();
    buildCullDescriptorSets();

    // Pipelines — require descriptor set layouts built above
    try {
        buildIndirectPBRPipeline();
        buildCullPipeline();
        mIndirectRenderingEnabled = true;
    } catch (const std::exception& e) {
        std::cerr << "[GPUScene] Indirect pipeline build failed: " << e.what()
                  << " — falling back to per-draw path.\n";
        mIndirectRenderingEnabled = false;
    }
}

// ---------------------------------------------------------------------------
// buildIndirectGlobalDescriptorSets
// Set 0 for pbr_indirect: FrameData UBO + ObjectData SSBO + MaterialData SSBO
//                          + 5 shadow cascade maps + irradiance + prefiltered + BRDF LUT
// ---------------------------------------------------------------------------
void Renderer::buildIndirectGlobalDescriptorSets()
{
    // Descriptor set layout
    const std::array<vk::DescriptorSetLayoutBinding, 11> bindings = {{
        // 0: FrameData UBO
        { .binding = 0,  .descriptorType = vk::DescriptorType::eUniformBuffer,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        // 1: ObjectData SSBO
        { .binding = 1,  .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
        // 2: MaterialData SSBO
        { .binding = 2,  .descriptorType = vk::DescriptorType::eStorageBuffer,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        // 3-7: Shadow cascade maps
        { .binding = 3,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 4,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 5,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 6,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        { .binding = 7,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        // 8: Irradiance cube
        { .binding = 8,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        // 9: Prefiltered cube
        { .binding = 9,  .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
        // 10: BRDF LUT
        { .binding = 10, .descriptorType = vk::DescriptorType::eCombinedImageSampler,
          .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment },
    }};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()
    };
    mIndirectGlobalSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    // Descriptor pool
    std::array<vk::DescriptorPoolSize, 3> poolSizes = {{
        { vk::DescriptorType::eUniformBuffer,         MAX_FRAMES_IN_FLIGHT },
        { vk::DescriptorType::eStorageBuffer,         MAX_FRAMES_IN_FLIGHT * 2 },
        { vk::DescriptorType::eCombinedImageSampler,  MAX_FRAMES_IN_FLIGHT * 8 },
    }};
    vk::DescriptorPoolCreateInfo poolInfo{
        .maxSets       = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes    = poolSizes.data()
    };
    mIndirectGlobalPool = vk::raii::DescriptorPool(device, poolInfo);

    // Allocate one set per frame-in-flight
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *mIndirectGlobalSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool     = *mIndirectGlobalPool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts        = layouts.data()
    };
    auto rawSets = device.allocateDescriptorSets(allocInfo);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        mIndirectGlobalSets.emplace_back(std::move(rawSets[i]));

    // Write static bindings (same for both frames)
    updateIndirectGlobalDescriptors();
}

void Renderer::updateIndirectGlobalDescriptors()
{
    if (mIndirectGlobalSets.empty()) return;

    for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi)
    {
        vk::DescriptorSet dstSet = *mIndirectGlobalSets[fi];

        // 0: FrameData UBO
        vk::DescriptorBufferInfo uboInfo{
            .buffer = *mGPUScene.frameUBOBuffers[fi],
            .offset = 0, .range = sizeof(FrameData) };

        // 1: ObjectData SSBO
        vk::DescriptorBufferInfo objInfo{
            .buffer = *mGPUScene.objectDataBuffer,
            .offset = 0, .range = vk::WholeSize };

        // 2: MaterialData SSBO
        vk::DescriptorBufferInfo matInfo{
            .buffer = *mGPUScene.materialDataBuffer,
            .offset = 0, .range = vk::WholeSize };

        // 3-7: Shadow maps
        auto shadowLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        std::array<vk::DescriptorImageInfo, 5> shadowInfos;
        for (int c = 0; c < 5; ++c)
            shadowInfos[c] = { .sampler = *shadowSampler, .imageView = *shadowImageViews[c],
                               .imageLayout = shadowLayout };

        // 8-10: IBL
        vk::DescriptorImageInfo iblIrr { .sampler = *mIBLSampler, .imageView = *mIBLIrradianceView,
                                          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo iblPre { .sampler = *mIBLSampler, .imageView = *mIBLPrefilteredView,
                                          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
        vk::DescriptorImageInfo iblLut { .sampler = *mIBLSampler, .imageView = *mIBLBrdfLutView,
                                          .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };

        std::vector<vk::WriteDescriptorSet> writes = {
            { .dstSet = dstSet, .dstBinding = 0,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eUniformBuffer,        .pBufferInfo = &uboInfo },
            { .dstSet = dstSet, .dstBinding = 1,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,        .pBufferInfo = &objInfo },
            { .dstSet = dstSet, .dstBinding = 2,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eStorageBuffer,        .pBufferInfo = &matInfo },
            { .dstSet = dstSet, .dstBinding = 3,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfos[0] },
            { .dstSet = dstSet, .dstBinding = 4,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfos[1] },
            { .dstSet = dstSet, .dstBinding = 5,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfos[2] },
            { .dstSet = dstSet, .dstBinding = 6,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfos[3] },
            { .dstSet = dstSet, .dstBinding = 7,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &shadowInfos[4] },
            { .dstSet = dstSet, .dstBinding = 8,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblIrr },
            { .dstSet = dstSet, .dstBinding = 9,  .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblPre },
            { .dstSet = dstSet, .dstBinding = 10, .descriptorCount = 1,
              .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &iblLut },
        };
        device.updateDescriptorSets(writes, nullptr);
    }
}

// ---------------------------------------------------------------------------
// buildCullDescriptorSets
// Set 0 for frustum_cull.comp: drawCmdIn, drawCmdOut, drawCount, objectData
// ---------------------------------------------------------------------------
void Renderer::buildCullDescriptorSets()
{
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {{
        { .binding = 0, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 2, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute },
        { .binding = 3, .descriptorType = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1,
          .stageFlags = vk::ShaderStageFlagBits::eCompute },
    }};

    vk::DescriptorSetLayoutCreateInfo layoutInfo{
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings    = bindings.data()
    };
    mCullSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

    vk::DescriptorPoolSize poolSize{ vk::DescriptorType::eStorageBuffer, 4 };
    vk::DescriptorPoolCreateInfo poolInfo{
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize
    };
    mCullPool = vk::raii::DescriptorPool(device, poolInfo);

    vk::DescriptorSetAllocateInfo allocInfo{
        .descriptorPool = *mCullPool, .descriptorSetCount = 1, .pSetLayouts = &*mCullSetLayout
    };
    auto sets = device.allocateDescriptorSets(allocInfo);
    mCullSet = std::move(sets[0]);

    // Write bindings
    vk::DescriptorBufferInfo inInfo  { .buffer = *mGPUScene.drawCommandBuffer, .offset = 0, .range = vk::WholeSize };
    vk::DescriptorBufferInfo outInfo { .buffer = *mGPUScene.cullOutputBuffer,  .offset = 0, .range = vk::WholeSize };
    vk::DescriptorBufferInfo cntInfo { .buffer = *mGPUScene.drawCountBuffer,   .offset = 0, .range = sizeof(uint32_t) };
    vk::DescriptorBufferInfo objInfo { .buffer = *mGPUScene.objectDataBuffer,  .offset = 0, .range = vk::WholeSize };

    std::array<vk::WriteDescriptorSet, 4> writes = {{
        { .dstSet = *mCullSet, .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &inInfo  },
        { .dstSet = *mCullSet, .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &outInfo },
        { .dstSet = *mCullSet, .dstBinding = 2, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &cntInfo },
        { .dstSet = *mCullSet, .dstBinding = 3, .descriptorCount = 1,
          .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &objInfo },
    }};
    device.updateDescriptorSets(writes, nullptr);
}

// ---------------------------------------------------------------------------
// buildIndirectPBRPipeline
// ---------------------------------------------------------------------------
void Renderer::buildIndirectPBRPipeline()
{
    auto vsCode = loadSPV("shaders/pbr_indirect.spv");
    auto fsCode = loadSPV("shaders/pbr_indirect.spv"); // Slang emits vertex+fragment in one SPV

    vk::ShaderModuleCreateInfo smInfo;
    smInfo.codeSize = vsCode.size() * 4;
    smInfo.pCode    = vsCode.data();
    vk::raii::ShaderModule shaderMod(device, smInfo);

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {{
        { .stage = vk::ShaderStageFlagBits::eVertex,
          .module = *shaderMod, .pName = "vertMain" },
        { .stage = vk::ShaderStageFlagBits::eFragment,
          .module = *shaderMod, .pName = "fragMain" }
    }};

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDescs   = Vertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInput{
        .vertexBindingDescriptionCount   = 1,   .pVertexBindingDescriptions   = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size()),
        .pVertexAttributeDescriptions    = attrDescs.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
        .topology = vk::PrimitiveTopology::eTriangleList
    };

    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1, .scissorCount = 1 };

    vk::PipelineRasterizationStateCreateInfo rasterizer{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode    = vk::CullModeFlagBits::eBack,
        .frontFace   = vk::FrontFace::eCounterClockwise,
        .lineWidth   = 1.0f
    };

    vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1
    };

    vk::PipelineDepthStencilStateCreateInfo depthStencil{
        .depthTestEnable  = true, .depthWriteEnable = true,
        .depthCompareOp   = vk::CompareOp::eLess
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{
        .blendEnable    = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };
    vk::PipelineColorBlendStateCreateInfo colorBlend{
        .attachmentCount = 1, .pAttachments = &colorBlendAttachment
    };

    std::array<vk::DynamicState, 2> dynamicStates = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState{
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data()
    };

    // Pipeline layout: set 0 = global, set 1 = bindless textures (from GPUScene)
    std::array<vk::DescriptorSetLayout, 2> setLayouts = {
        *mIndirectGlobalSetLayout,
        *mGPUScene.bindlessLayout
    };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
        .setLayoutCount = static_cast<uint32_t>(setLayouts.size()),
        .pSetLayouts    = setLayouts.data()
    };
    mIndirectPBRPipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // Dynamic rendering attachment info
    vk::Format colorFmt = vk::Format::eR16G16B16A16Sfloat; // matches existing scene pass
    vk::Format depthFmt = DepthTarget::findDepthFormat(physicalDevice);
    vk::PipelineRenderingCreateInfo renderingInfo{
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFmt,
        .depthAttachmentFormat   = depthFmt
    };

    vk::GraphicsPipelineCreateInfo pipelineInfo{
        .pNext               = &renderingInfo,
        .stageCount          = static_cast<uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState   = &multisampling,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlend,
        .pDynamicState       = &dynamicState,
        .layout              = *mIndirectPBRPipelineLayout
    };

    mIndirectPBRPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo);
}

// ---------------------------------------------------------------------------
// buildCullPipeline
// ---------------------------------------------------------------------------
void Renderer::buildCullPipeline()
{
    auto code = loadSPV("shaders/frustum_cull.comp.spv");
    vk::ShaderModuleCreateInfo smInfo{ .codeSize = code.size() * 4, .pCode = code.data() };
    vk::raii::ShaderModule cullShader(device, smInfo);

    // Push constants: 6 vec4 frustum planes + 1 uint drawCount = 100 bytes
    vk::PushConstantRange pcRange{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0, .size = sizeof(glm::vec4) * 6 + sizeof(uint32_t)
    };
    vk::PipelineLayoutCreateInfo layoutInfo{
        .setLayoutCount   = 1,
        .pSetLayouts      = &*mCullSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcRange
    };
    mCullPipelineLayout = vk::raii::PipelineLayout(device, layoutInfo);

    vk::ComputePipelineCreateInfo pipelineInfo{
        .stage  = { .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = *cullShader, .pName = "main" },
        .layout = *mCullPipelineLayout
    };
    mCullPipeline = device.createComputePipeline(nullptr, pipelineInfo);
}

// ---------------------------------------------------------------------------
// buildIndirectFrameData
// Assembles the FrameData UBO from the current camera, lights, and shadow state.
// Mirrors the cascade computation in uniform_buffer.cpp.
// ---------------------------------------------------------------------------
FrameData Renderer::buildIndirectFrameData(
    const std::array<glm::vec4, MAX_POINT_LIGHTS>& lightPos,
    const std::array<glm::vec4, MAX_POINT_LIGHTS>& lightCol) const
{
    FrameData fd{};

    const float aspect  = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
    fd.view = camera.getViewMatrix();
    fd.proj = camera.getProjectionMatrix(aspect, 0.1f, 3000.0f);

    // Cascade computation (identical to uniform_buffer.cpp)
    const float camNear = 0.1f, camFar = 3000.0f;
    const float shadowMaxDist = glm::clamp(shadowSettings.shadowMaxDistance, camNear, camFar);
    const float cascadeFar    = glm::min(camFar, shadowMaxDist);
    const float lambda        = glm::clamp(shadowSettings.lambda, 0.0f, 1.0f);

    float splits[SHADOW_CASCADE_COUNT];
    for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
    {
        float p    = static_cast<float>(i + 1) / static_cast<float>(SHADOW_CASCADE_COUNT);
        float logS = camNear * std::pow(cascadeFar / camNear, p);
        float uniS = camNear + (cascadeFar - camNear) * p;
        splits[i]  = lambda * logS + (1.0f - lambda) * uniS;
    }
    fd.cascadeSplits = glm::vec4(splits[0], splits[1], splits[2], splits[3]);

    glm::vec3 lightDir = glm::length(shadowSettings.lightDirection) > 1e-6f
        ? glm::normalize(shadowSettings.lightDirection)
        : glm::normalize(glm::vec3(23.0f, 90.0f, -8.0f));

    fd.directionalLightDirection = glm::vec4(lightDir, 0.0f);
    fd.directionalLightColor     = glm::vec4(shadowSettings.lightColor, 0.0f);

    const float tanHalfFov = std::tan(glm::radians(camera.getZoom()) * 0.5f);
    const glm::mat4 invView = glm::inverse(fd.view);
    const glm::vec3 camPos  = glm::vec3(invView[3]);
    const glm::vec3 camRight   = glm::normalize(glm::vec3(invView[0]));
    const glm::vec3 camUp      = glm::normalize(glm::vec3(invView[1]));
    const glm::vec3 camForward = glm::normalize(-glm::vec3(invView[2]));

    float prevSplit = camNear;
    for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade)
    {
        float nearDist = prevSplit, farDist = splits[cascade];
        prevSplit = farDist;
        float cascadeT     = SHADOW_CASCADE_COUNT > 1
            ? static_cast<float>(cascade) / static_cast<float>(SHADOW_CASCADE_COUNT - 1) : 0.0f;
        float expansion    = glm::mix(1.0f, glm::max(1.0f, shadowSettings.farCascadeExpansion), cascadeT * cascadeT);
        float nHH = tanHalfFov * nearDist, nHW = nHH * aspect;
        float fHH = tanHalfFov * farDist,  fHW = fHH * aspect;

        glm::vec3 nC = camPos + camForward * nearDist;
        glm::vec3 fC = camPos + camForward * farDist;
        glm::vec3 corners[8] = {
            nC + camUp*nHH - camRight*nHW, nC + camUp*nHH + camRight*nHW,
            nC - camUp*nHH + camRight*nHW, nC - camUp*nHH - camRight*nHW,
            fC + camUp*fHH - camRight*fHW, fC + camUp*fHH + camRight*fHW,
            fC - camUp*fHH + camRight*fHW, fC - camUp*fHH - camRight*fHW,
        };

        glm::vec3 centre(0.0f);
        for (auto& c : corners) centre += c;
        centre /= 8.0f;

        float radius = 0.0f;
        for (auto& c : corners) radius = glm::max(radius, glm::length(c - centre));
        radius = std::ceil(radius * 32.0f) / 32.0f;

        glm::mat4 lightView = glm::lookAtRH(centre - lightDir * (radius * 2.0f), centre, glm::vec3(0, 0, 1));

        glm::vec3 mn(1e9f), mx(-1e9f);
        glm::vec3 castOffset = -lightDir * (shadowSettings.casterPadding * expansion);
        for (auto& c : corners)
        {
            glm::vec3 lc  = glm::vec3(lightView * glm::vec4(c, 1.0f));
            glm::vec3 lcc = glm::vec3(lightView * glm::vec4(c + castOffset, 1.0f));
            mn = glm::min(mn, glm::min(lc, lcc));
            mx = glm::max(mx, glm::max(lc, lcc));
        }

        glm::vec3 centreLS = glm::vec3(lightView * glm::vec4(centre, 1.0f));
        float texel = (radius * 2.0f) / 2048.0f;
        centreLS.x  = std::floor(centreLS.x / texel) * texel;
        centreLS.y  = std::floor(centreLS.y / texel) * texel;

        float pad = shadowSettings.shadowPadding * expansion;
        float covPad = glm::max(0.0f, radius * shadowSettings.coveragePaddingFactor * expansion);
        float depPad = glm::max(0.0f, radius * shadowSettings.depthPaddingFactor * expansion);
        glm::mat4 lightProj = glm::orthoRH_ZO(
            centreLS.x - radius - covPad, centreLS.x + radius + covPad,
            centreLS.y - radius - covPad, centreLS.y + radius + covPad,
            -mx.z - pad - depPad, -mn.z + pad + depPad);

        fd.lightSpaceMatrices[cascade] = lightProj * lightView;
    }

    for (size_t i = 0; i < MAX_POINT_LIGHTS; ++i)
    {
        fd.lightPositions[i] = lightPos[i];
        fd.lightColors[i]    = lightCol[i];
    }

    fd.camPos                  = glm::vec4(camPos, 1.0f);
    fd.exposure                = 2.0f;
    fd.gamma                   = 2.2f;
    fd.prefilteredCubeMipLevels = static_cast<float>(mIBLPrefilteredMips);
    fd.scaleIBLAmbient          = mIBLAmbientScale;
    fd.shadowTuning = glm::vec4(shadowSettings.biasScale, shadowSettings.biasMin,
                                 shadowSettings.cascadeBlendFactor, shadowSettings.enabled);
    fd.shadowDebug  = glm::vec4(shadowSettings.cascadeDebugView, 0, 0, 0);

    return fd;
}

// ---------------------------------------------------------------------------
// extractFrustumPlanes — Gribb-Hartmann method from view-projection matrix
// ---------------------------------------------------------------------------
static std::array<glm::vec4, 6> extractFrustumPlanes(const glm::mat4& vp)
{
    std::array<glm::vec4, 6> planes;
    // Each plane: ax + by + cz + d (inward-pointing normals)
    // Left
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0],
                           vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    // Right
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0],
                           vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    // Bottom
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1],
                           vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    // Top
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1],
                           vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    // Near
    planes[4] = glm::vec4(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    // Far
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2],
                           vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    // Normalise so the dot product gives signed distance in world units
    for (auto& p : planes)
    {
        float len = glm::length(glm::vec3(p));
        if (len > 1e-6f) p /= len;
    }
    return planes;
}

// ---------------------------------------------------------------------------
// recordCullPass
// Resets drawCountBuffer, dispatches frustum cull compute, inserts barrier.
// Called BEFORE recordIndirectScenePass in the same command buffer.
// ---------------------------------------------------------------------------
void Renderer::recordCullPass(vk::raii::CommandBuffer& cmd)
{
    if (!mGPUScene.valid) return;

    const uint32_t totalDraws = mGPUScene.drawCount();
    if (totalDraws == 0) return;
    const vk::DeviceSize cmdBufSize = vk::DeviceSize(totalDraws) * sizeof(vk::DrawIndexedIndirectCommand);

    if (!mCullingEnabled)
    {
        // Pass all draws through: copy source commands → cull output, set full count.
        cmd.copyBuffer(*mGPUScene.drawCommandBuffer, *mGPUScene.cullOutputBuffer,
            vk::BufferCopy{ .size = cmdBufSize });
        cmd.fillBuffer(*mGPUScene.drawCountBuffer, 0, sizeof(uint32_t), totalDraws);

        // Barrier: transfer writes → indirect draw reads + readback copy
        std::array<vk::BufferMemoryBarrier2, 2> barriers = {{
            {   .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eTransferRead,
                .buffer = *mGPUScene.cullOutputBuffer, .offset = 0, .size = vk::WholeSize },
            {   .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eTransferRead,
                .buffer = *mGPUScene.drawCountBuffer, .offset = 0, .size = sizeof(uint32_t) },
        }};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pBufferMemoryBarriers    = barriers.data()
        });
    }
    else
    {
        // --- 1. Reset draw count to 0 ---
        cmd.fillBuffer(*mGPUScene.drawCountBuffer, 0, sizeof(uint32_t), 0u);

        // Barrier: fillBuffer write → shader read/write
        vk::BufferMemoryBarrier2 fillBarrier{
            .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .buffer = *mGPUScene.drawCountBuffer, .offset = 0, .size = sizeof(uint32_t)
        };
        cmd.pipelineBarrier2(vk::DependencyInfo{ .bufferMemoryBarrierCount = 1,
                                                  .pBufferMemoryBarriers    = &fillBarrier });

        // --- 2. Dispatch cull compute ---
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *mCullPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *mCullPipelineLayout, 0, *mCullSet, nullptr);

        const float aspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
        glm::mat4 vp = camera.getProjectionMatrix(aspect, 0.1f, 3000.0f) * camera.getViewMatrix();
        auto planes  = extractFrustumPlanes(vp);

        struct CullPushConstants {
            glm::vec4 frustumPlanes[6];
            uint32_t  drawCount;
        };
        CullPushConstants pc;
        for (int i = 0; i < 6; ++i) pc.frustumPlanes[i] = planes[i];
        pc.drawCount = totalDraws;
        cmd.pushConstants<CullPushConstants>(*mCullPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pc);

        const uint32_t groups = (totalDraws + 63) / 64;
        cmd.dispatch(groups, 1, 1);

        // --- 3. Barrier: compute writes → indirect draw reads + readback copy ---
        std::array<vk::BufferMemoryBarrier2, 2> barriers = {{
            {   .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead,
                .buffer = *mGPUScene.cullOutputBuffer, .offset = 0, .size = vk::WholeSize },
            {   .srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eTransferRead,
                .buffer = *mGPUScene.drawCountBuffer, .offset = 0, .size = sizeof(uint32_t) },
        }};
        cmd.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
            .pBufferMemoryBarriers    = barriers.data()
        });
    }

    // --- Readback: copy draw count to host-visible buffer for this frame slot ---
    cmd.copyBuffer(*mGPUScene.drawCountBuffer, *mGPUScene.drawCountReadbackBuffers[frameIndex],
        vk::BufferCopy{ .size = sizeof(uint32_t) });
}

// ---------------------------------------------------------------------------
// recordIndirectScenePass
// Single vkCmdDrawIndexedIndirectCount for all static mesh draws.
// Call after recordCullPass in the same render pass.
// ---------------------------------------------------------------------------
void Renderer::recordIndirectScenePass(vk::raii::CommandBuffer& cmd)
{
    if (!mGPUScene.valid) return;

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *mIndirectPBRPipeline);

    // Bind global vertex / index buffers
    vk::Buffer vb = *mGPUScene.globalVertexBuffer;
    vk::DeviceSize off = 0;
    cmd.bindVertexBuffers(0, vb, off);
    cmd.bindIndexBuffer(*mGPUScene.globalIndexBuffer, 0, vk::IndexType::eUint32);

    // Set 0: global frame data (per frame-in-flight)
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           *mIndirectPBRPipelineLayout, 0,
                           *mIndirectGlobalSets[frameIndex], nullptr);

    // Set 1: bindless textures
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                           *mIndirectPBRPipelineLayout, 1,
                           *mGPUScene.bindlessSet, nullptr);

    const vk::DeviceSize stride = sizeof(vk::DrawIndexedIndirectCommand);
    cmd.drawIndexedIndirectCount(
        *mGPUScene.cullOutputBuffer,  // buffer of surviving draw commands
        0,                            // offset
        *mGPUScene.drawCountBuffer,   // buffer holding the count
        0,                            // count offset
        mGPUScene.drawCount(),        // maxDrawCount
        static_cast<uint32_t>(stride));
}

// ---------------------------------------------------------------------------
// updateIndirectFrameData
// Called every frame to refresh FrameData UBO from current camera/lights state.
// ---------------------------------------------------------------------------
void Renderer::updateIndirectFrameData()
{
    if (!mGPUScene.valid) return;

    // Safe to read: the fence for this frameIndex was already waited in drawFrame().
    if (mGPUScene.drawCountReadbackMapped[frameIndex])
        mGPUScene.lastVisibleDrawCount =
            *static_cast<const uint32_t*>(mGPUScene.drawCountReadbackMapped[frameIndex]);

    auto& registry = mEnttScene.getRegistry();
    std::array<glm::vec4, MAX_POINT_LIGHTS> lightPos{}, lightCol{};
    collectPointLights(registry, lightPos, lightCol);

    FrameData fd = buildIndirectFrameData(lightPos, lightCol);
    mGPUScene.updateFrameData(frameIndex, fd);

    mGPUScene.updateObjectTransforms(registry);
    mGPUScene.flushObjectData();
}
