#include "sync.h"

#include <cassert>

void Sync::createSyncObjects(vk::raii::Device& device,
    size_t swapChainImageCount,
    uint32_t framesInFlight,
    std::vector<vk::raii::Semaphore>& presentCompleteSemaphores,
    std::vector<vk::raii::Semaphore>& renderFinishedSemaphores,
    std::vector<vk::raii::Fence>& inFlightFences)
{
    assert(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty());

    for (size_t i = 0; i < swapChainImageCount; i++)
        renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});

    for (uint32_t i = 0; i < framesInFlight; i++)
    {
        presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo{});
        inFlightFences.emplace_back(device, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled });
    }
}