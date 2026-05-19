#include "renderer/renderer.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <ktx.h>

#include "../vulkan/buffer.h"
#include "../vulkan/image.h"
#include "../vulkan/image_view.h"
#include "../ECS/components/renderable_component.h"
#include "../../lib/ImGuiFileDialog.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

uint32_t readU32LE(const uint8_t* p)
{
    return static_cast<uint32_t>(p[0]) |
        (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) |
        (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t readU64LE(const uint8_t* p)
{
    return static_cast<uint64_t>(p[0]) |
        (static_cast<uint64_t>(p[1]) << 8) |
        (static_cast<uint64_t>(p[2]) << 16) |
        (static_cast<uint64_t>(p[3]) << 24) |
        (static_cast<uint64_t>(p[4]) << 32) |
        (static_cast<uint64_t>(p[5]) << 40) |
        (static_cast<uint64_t>(p[6]) << 48) |
        (static_cast<uint64_t>(p[7]) << 56);
}

std::string getKtx2HeaderSummary(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return "Header read failed: cannot open file";

    std::array<uint8_t, 104> h{};
    f.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(h.size()));
    if (f.gcount() < static_cast<std::streamsize>(h.size()))
        return "Header read failed: file too small for KTX2 header + level index";

    constexpr uint8_t ktx2Magic[12] = { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };
    const bool magicOk = std::equal(std::begin(ktx2Magic), std::end(ktx2Magic), h.begin());

    std::ostringstream ss;
    ss << "KTX2 header: magic=" << (magicOk ? "ok" : "bad")
       << ", vkFormat=" << readU32LE(&h[12])
       << ", pixelWidth=" << readU32LE(&h[20])
       << ", pixelHeight=" << readU32LE(&h[24])
       << ", layerCount=" << readU32LE(&h[32])
       << ", faceCount=" << readU32LE(&h[36])
       << ", levelCount=" << readU32LE(&h[40])
       << ", supercompression=" << readU32LE(&h[44])
       << ", level0Offset=" << readU64LE(&h[80])
       << ", level0Length=" << readU64LE(&h[88])
       << ", level0UncompressedLength=" << readU64LE(&h[96]);
    return ss.str();
}

// Upload raw pixel data to a device-local VkImage via a staging buffer.
// Handles both 2D (layerCount=1) and cubemap (layerCount=6) images.
// copyRegions maps each staging-buffer offset to its image subresource.
void uploadImageData(
    vk::raii::Device&        device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::CommandPool&   commandPool,
    vk::raii::Queue&         queue,
    vk::Image                dstImage,
    const void*              data,
    vk::DeviceSize           dataSize,
    const std::vector<vk::BufferImageCopy>& copyRegions)
{
    // Staging buffer
    vk::raii::Buffer       stagingBuf  = nullptr;
    vk::raii::DeviceMemory stagingMem  = nullptr;
    Buffer::createBuffer(device, physicalDevice, dataSize,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBuf, stagingMem);

    void* mapped = stagingMem.mapMemory(0, dataSize);
    std::memcpy(mapped, data, static_cast<size_t>(dataSize));
    stagingMem.unmapMemory();

    // One-time command buffer
    vk::CommandBufferAllocateInfo allocInfo{
        .commandPool        = *commandPool,
        .level              = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    };
    auto cmds = device.allocateCommandBuffers(allocInfo);
    cmds[0].begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // Undefined → TransferDst
    uint32_t layerCount = 0;
    uint32_t levelCount = 0;
    for (auto& r : copyRegions) {
        layerCount = std::max(layerCount, r.imageSubresource.baseArrayLayer + r.imageSubresource.layerCount);
        levelCount = std::max(levelCount, r.imageSubresource.mipLevel + 1);
    }

    vk::ImageMemoryBarrier2 toTransfer{
        .srcStageMask  = vk::PipelineStageFlagBits2::eTopOfPipe,
        .srcAccessMask = {},
        .dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout     = vk::ImageLayout::eUndefined,
        .newLayout     = vk::ImageLayout::eTransferDstOptimal,
        .image         = dstImage,
        .subresourceRange = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = levelCount,
            .baseArrayLayer = 0,
            .layerCount     = layerCount
        }
    };
    cmds[0].pipelineBarrier2({ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toTransfer });

    cmds[0].copyBufferToImage(*stagingBuf, dstImage,
        vk::ImageLayout::eTransferDstOptimal, copyRegions);

    // TransferDst → ShaderReadOnly
    vk::ImageMemoryBarrier2 toRead{
        .srcStageMask  = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask  = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
        .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
        .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image         = dstImage,
        .subresourceRange = toTransfer.subresourceRange
    };
    cmds[0].pipelineBarrier2({ .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &toRead });

    cmds[0].end();
    queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*cmds[0] });
    queue.waitIdle();
}

// Find a memory type index satisfying the given properties.
uint32_t findMemoryType(vk::raii::PhysicalDevice& physicalDevice,
    uint32_t typeBits, vk::MemoryPropertyFlags props)
{
    auto memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("IBL: no suitable memory type");
}

// Create a device-local image + DeviceMemory (bypassing VMA for IBL resources).
void createDeviceImage(
    vk::raii::Device&         device,
    vk::raii::PhysicalDevice& physicalDevice,
    const vk::ImageCreateInfo& info,
    vk::raii::Image&          outImage,
    vk::raii::DeviceMemory&   outMemory)
{
    outImage  = vk::raii::Image(device, info);
    auto req  = outImage.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{
        .allocationSize  = req.size,
        .memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits,
                                          vk::MemoryPropertyFlagBits::eDeviceLocal)
    };
    outMemory = vk::raii::DeviceMemory(device, allocInfo);
    outImage.bindMemory(*outMemory, 0);
}

// Map a KTX1 GL internal format to the nearest Vulkan format for common IBL types.
vk::Format glInternalFormatToVk(uint32_t glFmt)
{
    switch (glFmt)
    {
    case 0x881A: return vk::Format::eR16G16B16A16Sfloat;  // GL_RGBA16F
    case 0x8814: return vk::Format::eR32G32B32A32Sfloat;  // GL_RGBA32F
    case 0x8058: return vk::Format::eR8G8B8A8Unorm;       // GL_RGBA8
    case 0x1908: return vk::Format::eR8G8B8A8Unorm;       // GL_RGBA
    case 0x822F: return vk::Format::eR16G16Sfloat;        // GL_RG16F  (BRDF LUT)
    case 0x822B: return vk::Format::eR8G8Unorm;           // GL_RG8
    case 0x8F94: return vk::Format::eR16G16B16A16Sfloat;  // GL_RGBA16F_EXT (ES ext)
    default:     return vk::Format::eUndefined;
    }
}

// Direct KTX2 reader: bypasses KTX-Software entirely.
// Used as a fallback when KTX-Software rejects a structurally valid file due to
// strict spec validation (e.g. non-zero uncompressedByteLength for supercompression=0,
// non-standard DFD blocks). Only supports raw uncompressed KTX2.
std::string loadKTX2Direct(
    vk::raii::Device&         device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::CommandPool&    commandPool,
    vk::raii::Queue&          queue,
    const std::string&        path,
    bool                      isCubemap,
    vk::raii::Image&          outImage,
    vk::raii::DeviceMemory&   outMemory,
    vk::raii::ImageView&      outView,
    uint32_t&                 outMipLevels)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return "Cannot open file: " + path;
    auto fileSize = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(fileSize);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(fileSize));
    f.close();

    if (fileSize < 80)
        return "File too small for KTX2 header: " + path;

    constexpr uint8_t ktx2Magic[12] = {0xAB,0x4B,0x54,0x58,0x20,0x32,0x30,0xBB,0x0D,0x0A,0x1A,0x0A};
    if (!std::equal(std::begin(ktx2Magic), std::end(ktx2Magic), buf.begin()))
        return "Not a KTX2 file (bad magic): " + path;

    const uint32_t vkFmt     = readU32LE(&buf[12]);
    const uint32_t width     = readU32LE(&buf[20]);
    const uint32_t height    = readU32LE(&buf[24]);
    const uint32_t faceCount = readU32LE(&buf[36]);
    uint32_t       levelCount = readU32LE(&buf[40]);
    const uint32_t supercomp = readU32LE(&buf[44]);

    if (supercomp != 0)
        return "Direct KTX2 reader only supports supercompression=0, got " +
               std::to_string(supercomp) + ": " + path;
    if (levelCount == 0) levelCount = 1;

    if (isCubemap && faceCount != 6)
        return "Expected cubemap (6 faces), file has " + std::to_string(faceCount) + ": " + path;
    if (!isCubemap && faceCount != 1)
        return "Expected 2D texture (1 face), file has " + std::to_string(faceCount) + ": " + path;

    if (fileSize < 80 + static_cast<std::size_t>(levelCount) * 24)
        return "File too small for level index: " + path;

    // Level index at byte 80: each entry is { byteOffset(8), byteLength(8), uncompressedByteLength(8) }.
    // Entry i corresponds to mip level i (0 = largest).
    struct LevelEntry { uint64_t byteOffset; uint64_t byteLength; };
    std::vector<LevelEntry> levels(levelCount);
    for (uint32_t lv = 0; lv < levelCount; ++lv) {
        levels[lv].byteOffset = readU64LE(&buf[80 + lv * 24]);
        levels[lv].byteLength = readU64LE(&buf[80 + lv * 24 + 8]);
    }

    const vk::Format format = static_cast<vk::Format>(vkFmt);
    if (format == vk::Format::eUndefined)
        return "vkFormat is VK_FORMAT_UNDEFINED: " + path;

    const uint32_t layers = isCubemap ? 6u : 1u;
    outMipLevels = levelCount;

    vk::ImageCreateInfo imageInfo{
        .flags       = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{},
        .imageType   = vk::ImageType::e2D,
        .format      = format,
        .extent      = { width, height, 1 },
        .mipLevels   = levelCount,
        .arrayLayers = layers,
        .samples     = vk::SampleCountFlagBits::e1,
        .tiling      = vk::ImageTiling::eOptimal,
        .usage       = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode    = vk::SharingMode::eExclusive,
        .initialLayout  = vk::ImageLayout::eUndefined
    };
    createDeviceImage(device, physicalDevice, imageInfo, outImage, outMemory);

    // Pack all mip levels into a single staging buffer and build copy regions.
    vk::DeviceSize totalSize = 0;
    for (uint32_t lv = 0; lv < levelCount; ++lv) totalSize += levels[lv].byteLength;

    std::vector<uint8_t> staging(totalSize);
    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(levelCount * faceCount);

    vk::DeviceSize dst = 0;
    for (uint32_t lv = 0; lv < levelCount; ++lv) {
        const uint64_t srcOff = levels[lv].byteOffset;
        const uint64_t len    = levels[lv].byteLength;
        if (srcOff + len > fileSize)
            return "Level " + std::to_string(lv) + " data extends past end of file: " + path;

        std::memcpy(staging.data() + dst, buf.data() + srcOff, len);

        // Faces are stored sequentially within each level's data block.
        const vk::DeviceSize faceBytes = len / faceCount;
        for (uint32_t fc = 0; fc < faceCount; ++fc) {
            regions.push_back({
                .bufferOffset      = dst + fc * faceBytes,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = lv,
                    .baseArrayLayer = fc,
                    .layerCount     = 1
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { std::max(1u, width >> lv), std::max(1u, height >> lv), 1 }
            });
        }
        dst += len;
    }

    uploadImageData(device, physicalDevice, commandPool, queue,
        *outImage, staging.data(), totalSize, regions);

    vk::ImageViewCreateInfo viewInfo{
        .image    = *outImage,
        .viewType = isCubemap ? vk::ImageViewType::eCube : vk::ImageViewType::e2D,
        .format   = format,
        .subresourceRange = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = levelCount,
            .baseArrayLayer = 0,
            .layerCount     = layers
        }
    };
    outView = vk::raii::ImageView(device, viewInfo);
    return {};
}

// Load a KTX (v1 or v2) file and return a fully-uploaded VkImage + VkImageView.
// Returns empty string on success, or a human-readable error message on failure.
// isCubemap=true  → expects 6 array faces.
// isCubemap=false → expects a regular 2D texture.
std::string loadKTX2(
    vk::raii::Device&         device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::CommandPool&    commandPool,
    vk::raii::Queue&          queue,
    const std::string&        path,
    bool                      isCubemap,
    vk::raii::Image&          outImage,
    vk::raii::DeviceMemory&   outMemory,
    vk::raii::ImageView&      outView,
    uint32_t&                 outMipLevels)
{
    ktxTexture* ktxTex = nullptr;
    KTX_error_code ktxResult = ktxTexture_CreateFromNamedFile(
        path.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &ktxTex);

    if (ktxResult != KTX_SUCCESS) {
        // KTX-Software rejected the file (commonly KTX_FILE_DATA_ERROR for IBL files
        // from older bakers that write non-compliant level index fields).
        // Fall back to the direct reader which bypasses library validation.
        return loadKTX2Direct(device, physicalDevice, commandPool, queue,
            path, isCubemap, outImage, outMemory, outView, outMipLevels);
    }

    vk::Format format = vk::Format::eUndefined;

    if (ktxTex->classId == ktxTexture2_c) {
        auto* tex2 = reinterpret_cast<ktxTexture2*>(ktxTex);
        if (ktxTexture2_NeedsTranscoding(tex2)) {
            ktxResult = ktxTexture2_TranscodeBasis(tex2, KTX_TTF_BC7_RGBA, 0);
            if (ktxResult != KTX_SUCCESS)
                ktxResult = ktxTexture2_TranscodeBasis(tex2, KTX_TTF_RGBA32, 0);
            if (ktxResult != KTX_SUCCESS) {
                ktxTexture_Destroy(ktxTex);
                return "ktxTexture2_TranscodeBasis failed (code " + std::to_string(static_cast<int>(ktxResult)) + "): "
                    + std::string(ktxErrorString(ktxResult)) + "\n" + path;
            }
        }
        format = static_cast<vk::Format>(tex2->vkFormat);
    }
    else {
        auto* tex1 = reinterpret_cast<ktxTexture1*>(ktxTex);
        format = glInternalFormatToVk(tex1->glInternalformat);
        if (format == vk::Format::eUndefined) {
            std::ostringstream s;
            s << std::hex << tex1->glInternalformat;
            ktxTexture_Destroy(ktxTex);
            return "Unrecognised GL internal format 0x" + s.str() + " in " + path;
        }
    }

    if (format == vk::Format::eUndefined) {
        ktxTexture_Destroy(ktxTex);
        return "vkFormat is VK_FORMAT_UNDEFINED after loading: " + path;
    }

    const uint32_t width = ktxTex->baseWidth;
    const uint32_t height = ktxTex->baseHeight;
    const uint32_t mipLevels = ktxTex->numLevels;
    const uint32_t faces = ktxTex->numFaces;
    const uint32_t layers = isCubemap ? 6u : 1u;

    if (isCubemap && faces != 6u) {
        ktxTexture_Destroy(ktxTex);
        return "Expected cubemap (6 faces) for this slot, file has " + std::to_string(faces) + " face(s):\n" + path;
    }
    if (!isCubemap && faces != 1u) {
        ktxTexture_Destroy(ktxTex);
        return "Expected 2D texture (1 face) for this slot, file has " + std::to_string(faces) + " face(s):\n" + path;
    }

    outMipLevels = mipLevels;

    vk::ImageCreateInfo imageInfo{
        .flags       = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible
                                 : vk::ImageCreateFlags{},
        .imageType   = vk::ImageType::e2D,
        .format      = format,
        .extent      = { width, height, 1 },
        .mipLevels   = mipLevels,
        .arrayLayers = layers,
        .samples     = vk::SampleCountFlagBits::e1,
        .tiling      = vk::ImageTiling::eOptimal,
        .usage       = vk::ImageUsageFlagBits::eSampled |
                       vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode    = vk::SharingMode::eExclusive,
        .initialLayout  = vk::ImageLayout::eUndefined
    };
    createDeviceImage(device, physicalDevice, imageInfo, outImage, outMemory);

    // Build copy regions: one per (mip, face) pair.
    ktx_uint8_t* src      = ktxTexture_GetData(ktxTex);
    ktx_size_t   dataSize = ktxTexture_GetDataSize(ktxTex);

    std::vector<vk::BufferImageCopy> regions;
    regions.reserve(mipLevels * faces);
    for (uint32_t m = 0; m < mipLevels; ++m) {
        for (uint32_t f = 0; f < faces; ++f) {
            ktx_size_t offset = 0;
            ktxTexture_GetImageOffset(ktxTex, m, 0, f, &offset);
            regions.push_back({
                .bufferOffset      = offset,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = m,
                    .baseArrayLayer = f,
                    .layerCount     = 1
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { std::max(1u, width >> m),
                                 std::max(1u, height >> m), 1 }
            });
        }
    }

    uploadImageData(device, physicalDevice, commandPool, queue,
        *outImage, src, dataSize, regions);

    vk::ImageViewCreateInfo viewInfo{
        .image    = *outImage,
        .viewType = isCubemap ? vk::ImageViewType::eCube
                              : vk::ImageViewType::e2D,
        .format   = format,
        .subresourceRange = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = mipLevels,
            .baseArrayLayer = 0,
            .layerCount     = layers
        }
    };
    outView = vk::raii::ImageView(device, viewInfo);

    ktxTexture_Destroy(ktxTex);
    return {};
}

// Create a 1×1 solid-colour 2D or cube image used as a neutral default.
void create1x1Image(
    vk::raii::Device&         device,
    vk::raii::PhysicalDevice& physicalDevice,
    vk::raii::CommandPool&    commandPool,
    vk::raii::Queue&          queue,
    std::array<uint8_t, 4>    rgba,
    bool                      isCubemap,
    vk::raii::Image&          outImage,
    vk::raii::DeviceMemory&   outMemory,
    vk::raii::ImageView&      outView)
{
    const uint32_t layers = isCubemap ? 6u : 1u;
    vk::ImageCreateInfo imageInfo{
        .flags       = isCubemap ? vk::ImageCreateFlagBits::eCubeCompatible
                                 : vk::ImageCreateFlags{},
        .imageType   = vk::ImageType::e2D,
        .format      = vk::Format::eR8G8B8A8Unorm,
        .extent      = { 1, 1, 1 },
        .mipLevels   = 1,
        .arrayLayers = layers,
        .samples     = vk::SampleCountFlagBits::e1,
        .tiling      = vk::ImageTiling::eOptimal,
        .usage       = vk::ImageUsageFlagBits::eSampled |
                       vk::ImageUsageFlagBits::eTransferDst,
        .sharingMode   = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout::eUndefined
    };
    createDeviceImage(device, physicalDevice, imageInfo, outImage, outMemory);

    // Upload the same 1×1 pixel for every face
    std::vector<uint8_t> pixelData(4 * layers);
    for (uint32_t f = 0; f < layers; ++f)
        std::memcpy(pixelData.data() + f * 4, rgba.data(), 4);

    std::vector<vk::BufferImageCopy> regions;
    for (uint32_t f = 0; f < layers; ++f) {
        regions.push_back({
            .bufferOffset     = f * 4,
            .imageSubresource = {
                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                .mipLevel       = 0,
                .baseArrayLayer = f,
                .layerCount     = 1
            },
            .imageExtent = { 1, 1, 1 }
        });
    }
    uploadImageData(device, physicalDevice, commandPool, queue,
        *outImage, pixelData.data(), pixelData.size(), regions);

    vk::ImageViewCreateInfo viewInfo{
        .image    = *outImage,
        .viewType = isCubemap ? vk::ImageViewType::eCube : vk::ImageViewType::e2D,
        .format   = vk::Format::eR8G8B8A8Unorm,
        .subresourceRange = {
            .aspectMask     = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = layers
        }
    };
    outView = vk::raii::ImageView(device, viewInfo);
}

} // namespace

// ---------------------------------------------------------------------------
// Renderer methods
// ---------------------------------------------------------------------------

void Renderer::createIBLSampler()
{
    vk::SamplerCreateInfo info{
        .magFilter        = vk::Filter::eLinear,
        .minFilter        = vk::Filter::eLinear,
        .mipmapMode       = vk::SamplerMipmapMode::eLinear,
        .addressModeU     = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV     = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW     = vk::SamplerAddressMode::eClampToEdge,
        .mipLodBias       = 0.0f,
        .anisotropyEnable = vk::True,
        .maxAnisotropy    = physicalDevice.getProperties().limits.maxSamplerAnisotropy,
        .compareEnable    = vk::False,
        .minLod           = 0.0f,
        .maxLod           = static_cast<float>(mIBLPrefilteredMips),
        .borderColor      = vk::BorderColor::eFloatOpaqueWhite
    };
    mIBLSampler = vk::raii::Sampler(device, info);
}

void Renderer::createDefaultIBLResources()
{
    // Neutral defaults: mid-grey cubemaps + white BRDF LUT
    // Mid-grey irradiance gives a flat ~18% ambient; white BRDF LUT is a valid fallback.
    std::array<uint8_t,4> grey  = {128, 128, 128, 255};
    std::array<uint8_t,4> white = {255, 255, 255, 255};

    create1x1Image(device, physicalDevice, commandPool, queue, grey,  true,
        mIBLIrradianceImage, mIBLIrradianceMemory, mIBLIrradianceView);
    create1x1Image(device, physicalDevice, commandPool, queue, grey,  true,
        mIBLPrefilteredImage, mIBLPrefilteredMemory, mIBLPrefilteredView);
    create1x1Image(device, physicalDevice, commandPool, queue, white, false,
        mIBLBrdfLutImage, mIBLBrdfLutMemory, mIBLBrdfLutView);

    mIBLPrefilteredMips = 1;
    createIBLSampler();
}

std::string Renderer::loadIBL(const std::string& directory)
{
    // Strip any trailing path separator so directory + "/" + stem is always clean.
    std::string dir = directory;
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\'))
        dir.pop_back();

    if (dir.empty())
        return "Empty directory path.";

    // Resolve each asset, preferring KTX2 over KTX1.
    auto resolve = [&](const char* stem) -> std::string {
        for (auto& ext : { ".ktx2", ".ktx" }) {
            std::string p = dir + "/" + stem + ext;
            if (std::filesystem::exists(p)) return p;
        }
        return {};
    };

    std::string irradiancePath  = resolve("irradiance");
    std::string prefilteredPath = resolve("prefiltered");
    std::string brdfLutPath     = resolve("brdf_lut");

    if (irradiancePath.empty())
        return "irradiance.ktx2 not found in:\n" + dir;
    if (prefilteredPath.empty())
        return "prefiltered.ktx2 not found in:\n" + dir;
    if (brdfLutPath.empty())
        return "brdf_lut.ktx2 not found in:\n" + dir;

    return loadIBLFiles(irradiancePath, prefilteredPath, brdfLutPath);
}

std::string Renderer::loadIBLFiles(const std::string& irradiancePath,
                                   const std::string& prefilteredPath,
                                   const std::string& brdfLutPath)
{
    if (irradiancePath.empty())
        return "Irradiance file path is empty.";
    if (prefilteredPath.empty())
        return "Prefiltered file path is empty.";
    if (brdfLutPath.empty())
        return "BRDF LUT file path is empty.";

    // Temporary storage — only commit to members if all three succeed.
    vk::raii::Image        newIrrImage  = nullptr, newPreImage  = nullptr, newLutImage  = nullptr;
    vk::raii::DeviceMemory newIrrMem    = nullptr, newPreMem    = nullptr, newLutMem    = nullptr;
    vk::raii::ImageView    newIrrView   = nullptr, newPreView   = nullptr, newLutView   = nullptr;
    uint32_t               newMips      = 1;

    uint32_t dummyMips = 1;
    std::string err;
    err = loadKTX2(device, physicalDevice, commandPool, queue, irradiancePath, true,
                   newIrrImage, newIrrMem, newIrrView, dummyMips);
    if (!err.empty()) return err;
    err = loadKTX2(device, physicalDevice, commandPool, queue, prefilteredPath, true,
                   newPreImage, newPreMem, newPreView, newMips);
    if (!err.empty()) return err;
    err = loadKTX2(device, physicalDevice, commandPool, queue, brdfLutPath, false,
                   newLutImage, newLutMem, newLutView, dummyMips);
    if (!err.empty()) return err;

    // Commit
    mIBLIrradianceImage  = std::move(newIrrImage);
    mIBLIrradianceMemory = std::move(newIrrMem);
    mIBLIrradianceView   = std::move(newIrrView);
    mIBLPrefilteredImage  = std::move(newPreImage);
    mIBLPrefilteredMemory = std::move(newPreMem);
    mIBLPrefilteredView   = std::move(newPreView);
    mIBLBrdfLutImage  = std::move(newLutImage);
    mIBLBrdfLutMemory = std::move(newLutMem);
    mIBLBrdfLutView   = std::move(newLutView);
    mIBLPrefilteredMips = newMips;

    mIBLSampler = nullptr;
    createIBLSampler();
    mIBLLoaded = true;
    updateIBLDescriptors();
    return {};
}

void Renderer::updateIBLDescriptors()
{
    // Push updated IBL image infos into every existing per-material descriptor set,
    // without reallocating — only bindings 11, 12, 13 are touched.
    auto& registry = mEnttScene.getRegistry();
    for (auto [entity, renderable] : registry.view<RenderableComponent>().each()) {
        (void)entity;
        for (auto& setsPerMaterial : renderable.materialDescriptorSets) {
            for (auto& descSet : setsPerMaterial) {
                vk::DescriptorImageInfo irrInfo{
                    .sampler     = *mIBLSampler,
                    .imageView   = *mIBLIrradianceView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
                vk::DescriptorImageInfo preInfo{
                    .sampler     = *mIBLSampler,
                    .imageView   = *mIBLPrefilteredView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
                vk::DescriptorImageInfo lutInfo{
                    .sampler     = *mIBLSampler,
                    .imageView   = *mIBLBrdfLutView,
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
                };
                std::array writes{
                    vk::WriteDescriptorSet{.dstSet = *descSet, .dstBinding = 11,
                        .descriptorCount = 1,
                        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                        .pImageInfo      = &irrInfo },
                    vk::WriteDescriptorSet{.dstSet = *descSet, .dstBinding = 12,
                        .descriptorCount = 1,
                        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                        .pImageInfo      = &preInfo },
                    vk::WriteDescriptorSet{.dstSet = *descSet, .dstBinding = 13,
                        .descriptorCount = 1,
                        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                        .pImageInfo      = &lutInfo }
                };
                device.updateDescriptorSets(writes, {});
            }
        }
    }
}

void Renderer::renderIBLPanel(bool isEditMode)
{
    if (!((isEditMode || playShowDebugUI) && uiShowIBLWindow))
        return;

    ImGui::Begin("IBL");

    ImGui::Text("Status: %s", mIBLLoaded ? "Loaded" : "Default (grey cubemap)");
    ImGui::SliderFloat("IBL Scale", &mIBLAmbientScale, 0.0f, 2.0f, "%.3f");

    if (mIBLLoaded)
        ImGui::Text("Prefiltered mip levels: %u", mIBLPrefilteredMips);

    ImGui::Separator();
    ImGui::TextUnformatted("IBL files:");
    ImGui::BulletText("irradiance.(ktx2 or ktx)");
    ImGui::BulletText("prefiltered.(ktx2 or ktx)");
    ImGui::BulletText("brdf_lut.(ktx2 or ktx)");
    ImGui::TextUnformatted("Generate with: cmgen (Filament) or IBLBaker.");
    ImGui::Separator();

    auto openCenteredDialog = [](const char* key, const char* title, const char* filters) {
        IGFD::FileDialogConfig config;
        config.path = ".";
        config.countSelectionMax = 1;
        config.flags = ImGuiFileDialogFlags_Modal;
        ImGui::SetNextWindowPos(
            ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f),
            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGuiFileDialog::Instance()->OpenDialog(key, title, filters, config);
    };

    if (ImGui::Button("Browse for IBL Directory...")) {
        openCenteredDialog("ChooseIBLDir", "Choose IBL Directory", nullptr);
    }

    ImGui::SameLine();
    if (ImGui::Button("Select IBL Files Manually...")) {
        openCenteredDialog("ChooseIBLIrradiance", "Choose irradiance file", "KTX files{.ktx2,.ktx}");
    }

    static std::string sIBLError;
    static std::string sIBLInfo;
    static std::string sIBLIrradiancePath;
    static std::string sIBLPrefilteredPath;
    static std::string sIBLBrdfLutPath;
    bool openPrefilteredDialog = false;
    bool openBrdfDialog = false;

    if (!sIBLInfo.empty())
        ImGui::TextColored(ImVec4(0.45f, 0.95f, 0.45f, 1.0f), "%s", sIBLInfo.c_str());

    if (ImGuiFileDialog::Instance()->Display("ChooseIBLDir")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            std::replace(dir.begin(), dir.end(), '\\', '/');

            // Prefer a relative path so the directory is portable across machines.
            std::filesystem::path currentPath = std::filesystem::current_path();
            auto rel = std::filesystem::relative(dir, currentPath);
            if (!rel.empty() && rel.native().find(L"..") == std::wstring::npos)
                dir = rel.generic_string();

            sIBLError = loadIBL(dir);
            if (sIBLError.empty())
                sIBLInfo = "IBL loaded successfully from directory.";
            else
                sIBLInfo.clear();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("ChooseIBLIrradiance")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            sIBLIrradiancePath = ImGuiFileDialog::Instance()->GetFilePathName();
            std::replace(sIBLIrradiancePath.begin(), sIBLIrradiancePath.end(), '\\', '/');
            openPrefilteredDialog = true;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (openPrefilteredDialog)
        openCenteredDialog("ChooseIBLPrefiltered", "Choose prefiltered file", "KTX files{.ktx2,.ktx}");

    if (ImGuiFileDialog::Instance()->Display("ChooseIBLPrefiltered")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            sIBLPrefilteredPath = ImGuiFileDialog::Instance()->GetFilePathName();
            std::replace(sIBLPrefilteredPath.begin(), sIBLPrefilteredPath.end(), '\\', '/');
            openBrdfDialog = true;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    if (openBrdfDialog)
        openCenteredDialog("ChooseIBLBrdfLut", "Choose BRDF LUT file", "KTX files{.ktx2,.ktx}");

    if (ImGuiFileDialog::Instance()->Display("ChooseIBLBrdfLut")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            sIBLBrdfLutPath = ImGuiFileDialog::Instance()->GetFilePathName();
            std::replace(sIBLBrdfLutPath.begin(), sIBLBrdfLutPath.end(), '\\', '/');

            sIBLError = loadIBLFiles(sIBLIrradiancePath, sIBLPrefilteredPath, sIBLBrdfLutPath);
            if (sIBLError.empty())
                sIBLInfo = "IBL loaded successfully from manual files.";
            else
                sIBLInfo.clear();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // Defer OpenPopup to the next iteration so it fires inside the IBL window, not the dialog.
    if (!sIBLError.empty())
        ImGui::OpenPopup("IBL Load Failed");

    if (ImGui::BeginPopupModal("IBL Load Failed", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted(sIBLError.c_str());
        if (ImGui::Button("OK")) { sIBLError.clear(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    ImGui::End();
}
