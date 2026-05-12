#include "shader_storage_buffer.h"

bool ShaderStorageBuffer::init(VkRenderData& renderData, VkShaderStorageBufferData& SSBOData, size_t bufferSize)
{
    if (bufferSize == 0)
        bufferSize = 1024;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    VkResult result = vmaCreateBuffer(renderData.rdAllocator, &bufferInfo, &allocInfo,
        &SSBOData.buffer, &SSBOData.bufferAlloc, nullptr);
    if (result != VK_SUCCESS)
    {
        Logger::log(1, "%s error: could not allocate SSBO via VMA (error: %i)\n", __FUNCTION__, result);
        return false;
    }

    SSBOData.bufferSize = bufferSize;
    Logger::log(1, "%s: created SSBO of size %zu\n", __FUNCTION__, bufferSize);
    return true;
}

bool ShaderStorageBuffer::checkForResize(VkRenderData& renderData, VkShaderStorageBufferData& SSBOData, size_t bufferSize)
{
    if (bufferSize > SSBOData.bufferSize)
    {
        Logger::log(1, "%s: resize SSBO %p from %zu to %zu bytes\n", __FUNCTION__, SSBOData.buffer, SSBOData.bufferSize, bufferSize);
        cleanup(renderData, SSBOData);
        return init(renderData, SSBOData, bufferSize);
    }
    return false;
}

void ShaderStorageBuffer::cleanup(VkRenderData& renderData, VkShaderStorageBufferData& SSBOData)
{
    if (renderData.rdGraphicsQueue != VK_NULL_HANDLE)
    {
        VkResult result = vkQueueWaitIdle(renderData.rdGraphicsQueue);
        if (result != VK_SUCCESS)
        {
            Logger::log(1, "%s warning: could not wait for queue idle (error: %i)\n", __FUNCTION__, result);
        }
    }

    if (SSBOData.buffer != VK_NULL_HANDLE || SSBOData.bufferAlloc != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(renderData.rdAllocator, SSBOData.buffer, SSBOData.bufferAlloc);
    }

    SSBOData.buffer = VK_NULL_HANDLE;
    SSBOData.bufferAlloc = VK_NULL_HANDLE;
    SSBOData.bufferSize = 0;
    SSBOData.descriptorSet = VK_NULL_HANDLE;
}
