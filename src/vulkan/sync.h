#pragma once

#include <vector>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

class Sync
{
public:
    static void createSyncObjects(vk::raii::Device& device,
        size_t swapChainImageCount,
        uint32_t framesInFlight,
        std::vector<vk::raii::Semaphore>& presentCompleteSemaphores,
        std::vector<vk::raii::Semaphore>& renderFinishedSemaphores,
        std::vector<vk::raii::Fence>& inFlightFences);
};