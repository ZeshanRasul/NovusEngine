#pragma once

#include <vector>
#include <array>
#include <vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>
#include <entt/entt.hpp>

#include "renderer_types.h"
#include "../vulkan/uniform_buffer.h" // SHADOW_CASCADE_COUNT, ShadowSettings

// Forward declarations
struct RenderableComponent;
struct TransformComponent;
class Camera;

constexpr uint32_t GPU_MAX_DRAWS             = 4096;
constexpr uint32_t GPU_MAX_BINDLESS_TEXTURES = 1024;
constexpr uint32_t GPU_FRAMES_IN_FLIGHT      = 2;    // mirrors MAX_FRAMES_IN_FLIGHT

// Per-frame global data uploaded to a single shared UBO.
// Model matrices live in ObjectData SSBO — this contains only frame-level data.
struct FrameData
{
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightSpaceMatrices[SHADOW_CASCADE_COUNT];
    glm::vec4 directionalLightDirection;
    glm::vec4 directionalLightColor;
    glm::vec4 lightPositions[MAX_POINT_LIGHTS];
    glm::vec4 lightColors[MAX_POINT_LIGHTS];
    glm::vec4 camPos;
    glm::vec4 cascadeSplits;
    float     exposure;
    float     gamma;
    float     prefilteredCubeMipLevels;
    float     scaleIBLAmbient;
    glm::vec4 shadowTuning;
    glm::vec4 shadowDebug;
};

// ============================================================================
// GPUScene — owns all GPU resources for indirect draw + frustum culling.
//
// Lifecycle:
//   build()                — on scene load / entity change
//   updateObjectTransforms() + flushObjectData() — every frame (CPU→GPU)
//   updateFrameData()      — every frame (camera, lights, etc.)
//   drawCount()            — passed to cull compute dispatch
// ============================================================================
class GPUScene
{
public:
    // -----------------------------------------------------------------------
    // GPU-side structs — must match shader declarations exactly.
    // -----------------------------------------------------------------------
    struct alignas(16) ObjectData
    {
        glm::mat4 modelMatrix;    // 64 bytes
        glm::vec4 boundingSphere; // xyz = world-space centre, w = radius — 16 bytes
        uint32_t  materialIndex;  //  4 bytes
        uint32_t  _pad[3];        // 12 bytes pad to next 16-byte boundary
    };  // 96 bytes total
    static_assert(sizeof(ObjectData) % 16 == 0, "ObjectData must be 16-byte aligned");

    struct alignas(16) MaterialData
    {
        glm::vec4 baseColorFactor;         // 16
        float     metallicFactor;          //  4
        float     roughnessFactor;         //  4
        float     alphaMask;               //  4
        float     alphaMaskCutoff;         //  4
        int32_t   baseColorTexIdx;         //  4  (-1 = use default)
        int32_t   metallicRoughnessTexIdx; //  4
        int32_t   normalTexIdx;            //  4
        int32_t   occlusionTexIdx;         //  4
        int32_t   emissiveTexIdx;          //  4
        uint32_t  _pad[3];                 // 12
    };  // 64 bytes total
    static_assert(sizeof(MaterialData) % 16 == 0, "MaterialData must be 16-byte aligned");

    // -----------------------------------------------------------------------
    // GPU buffers
    // -----------------------------------------------------------------------

    // Merged scene geometry
    vk::raii::Buffer       globalVertexBuffer = nullptr;
    vk::raii::DeviceMemory globalVertexMemory = nullptr;
    vk::raii::Buffer       globalIndexBuffer  = nullptr;
    vk::raii::DeviceMemory globalIndexMemory  = nullptr;

    // Source draw commands (built once, read each frame by cull compute)
    vk::raii::Buffer       drawCommandBuffer  = nullptr;
    vk::raii::DeviceMemory drawCommandMemory  = nullptr;

    // Cull outputs (written by compute, consumed by vkCmdDrawIndexedIndirectCount)
    vk::raii::Buffer       cullOutputBuffer   = nullptr;
    vk::raii::DeviceMemory cullOutputMemory   = nullptr;
    vk::raii::Buffer       drawCountBuffer    = nullptr; // single uint32
    vk::raii::DeviceMemory drawCountMemory    = nullptr;

    // Object data SSBO (transforms updated every frame for moving objects)
    vk::raii::Buffer       objectDataBuffer   = nullptr;
    vk::raii::DeviceMemory objectDataMemory   = nullptr;
    void*                  objectDataMapped   = nullptr;

    // Material data SSBO (immutable after build)
    vk::raii::Buffer       materialDataBuffer = nullptr;
    vk::raii::DeviceMemory materialDataMemory = nullptr;

    // Per-frame global UBO (one per frame-in-flight)
    std::array<vk::raii::Buffer,       GPU_FRAMES_IN_FLIGHT> frameUBOBuffers   = { nullptr, nullptr };
    std::array<vk::raii::DeviceMemory, GPU_FRAMES_IN_FLIGHT> frameUBOMemories  = { nullptr, nullptr };
    std::array<void*,                  GPU_FRAMES_IN_FLIGHT> frameUBOMapped     = {};

    // Host-visible readback for the post-cull draw count (one per frame-in-flight)
    std::array<vk::raii::Buffer,       GPU_FRAMES_IN_FLIGHT> drawCountReadbackBuffers  = { nullptr, nullptr };
    std::array<vk::raii::DeviceMemory, GPU_FRAMES_IN_FLIGHT> drawCountReadbackMemories = { nullptr, nullptr };
    std::array<void*,                  GPU_FRAMES_IN_FLIGHT> drawCountReadbackMapped    = {};
    uint32_t lastVisibleDrawCount = 0;

    // Bindless PBR texture descriptor set (set 1 in the indirect pipeline)
    vk::raii::DescriptorPool      bindlessPool   = nullptr;
    vk::raii::DescriptorSetLayout bindlessLayout = nullptr;
    vk::raii::DescriptorSet       bindlessSet    = nullptr;
    uint32_t                      nextBindlessIdx = 0;

    // -----------------------------------------------------------------------
    // CPU mirrors
    // -----------------------------------------------------------------------
    std::vector<Vertex>                          vertices;
    std::vector<uint32_t>                        indices;
    std::vector<vk::DrawIndexedIndirectCommand>  drawCommands;
    std::vector<ObjectData>                      objectDataCPU;
    std::vector<MaterialData>                    materialDataCPU;

    // Entity → draw range mapping — used to patch transforms without full rebuild.
    struct EntityDrawRange
    {
        entt::entity entity;
        uint32_t     firstDraw;
        uint32_t     drawCount;
        glm::vec4    localBoundingSphere; // xyz = mesh-space centre, w = local radius
    };
    std::vector<EntityDrawRange> entityRanges;

    bool valid = false;

    // -----------------------------------------------------------------------
    // Interface
    // -----------------------------------------------------------------------

    // Register a texture image view into the bindless descriptor set.
    // Returns the global index to store in MaterialData.
    uint32_t registerTexture(vk::raii::Device& device,
                             vk::ImageView     imageView,
                             vk::Sampler       sampler);

    // Full scene rebuild — call when the set of entities changes.
    void build(entt::registry&           registry,
               vk::raii::Device&         device,
               vk::raii::PhysicalDevice& physicalDevice,
               vk::raii::Queue&          queue,
               vk::raii::CommandPool&    commandPool,
               vk::raii::Sampler&        defaultSampler,
               vk::ImageView             defaultTextureView,
               vk::ImageView             defaultNormalView);

    // Refresh model matrices and bounding spheres from current ECS transforms.
    void updateObjectTransforms(entt::registry& registry);

    // Upload the refreshed object data to the GPU (call after updateObjectTransforms).
    void flushObjectData();

    // Upload the per-frame global UBO (view, proj, lights, shadows…).
    void updateFrameData(uint32_t frameIndex, const FrameData& data);

    uint32_t drawCount() const { return static_cast<uint32_t>(drawCommands.size()); }

    void destroy();

private:
    void createBindlessDescriptors(vk::raii::Device& device);
    void createSSBOs(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice);
    void createCullBuffers(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice);
    void createFrameUBOs(vk::raii::Device& device, vk::raii::PhysicalDevice& physicalDevice);

    static void uploadViaStaging(vk::raii::Device&         device,
                                 vk::raii::PhysicalDevice& physicalDevice,
                                 vk::raii::Queue&          queue,
                                 vk::raii::CommandPool&    commandPool,
                                 const void*               data,
                                 vk::DeviceSize            size,
                                 vk::raii::Buffer&         dstBuffer,
                                 vk::raii::DeviceMemory&   dstMemory,
                                 vk::BufferUsageFlags      extraUsage);

    static glm::vec4 computeLocalBoundingSphere(const std::vector<Vertex>& verts,
                                                uint32_t firstVertex, uint32_t vertexCount);
};
