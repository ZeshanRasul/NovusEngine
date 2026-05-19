#include "renderer/renderer.h"

#include <algorithm>
#include <cmath>

void Renderer::createTimestampQueryPool()
{
    auto queueProps = physicalDevice.getQueueFamilyProperties();
    if (queueIndex >= queueProps.size() || queueProps[queueIndex].timestampValidBits == 0)
        return;

    mTimestampPeriod     = physicalDevice.getProperties().limits.timestampPeriod;
    mTimestampsSupported = true;

    const uint32_t totalQueries = MAX_FRAMES_IN_FLIGHT * GPU_PASS_SLOT_COUNT * 2;

    vk::QueryPoolCreateInfo poolInfo{
        .queryType  = vk::QueryType::eTimestamp,
        .queryCount = totalQueries
    };
    mTimestampQueryPool = vk::raii::QueryPool(device, poolInfo);

    // vkResetQueryPool (Vulkan 1.2 core) is not exposed on vk::raii::Device in all SDK
    // versions, so go through the raw C API handle.
    vkResetQueryPool(*device, *mTimestampQueryPool, 0, totalQueries);
}

void Renderer::readTimestamps()
{
    if (!mTimestampsSupported)
        return;

    const uint32_t firstQuery  = frameIndex * GPU_PASS_SLOT_COUNT * 2;
    constexpr uint32_t queryCount = GPU_PASS_SLOT_COUNT * 2;

    auto [res, data] = mTimestampQueryPool.getResults<uint64_t>(
        firstQuery, queryCount,
        sizeof(uint64_t) * queryCount,
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64);

    if (res != vk::Result::eSuccess)
        return;

    // timestampPeriod is in nanoseconds per tick; convert delta to milliseconds
    auto toMs = [&](uint64_t start, uint64_t end) -> float {
        return static_cast<float>((end - start) * mTimestampPeriod) * 1e-6f;
    };

    mGpuTimings.shadowMs = toMs(data[0], data[1]);
    mGpuTimings.sceneMs  = toMs(data[2], data[3]);
    mGpuTimings.bloomMs  = toMs(data[4], data[5]);
    mGpuTimings.fxaaMs   = toMs(data[6], data[7]);
    mGpuTimings.imguiMs  = toMs(data[8], data[9]);
    mGpuTimings.totalMs  = mGpuTimings.shadowMs + mGpuTimings.sceneMs
                         + mGpuTimings.bloomMs  + mGpuTimings.fxaaMs
                         + mGpuTimings.imguiMs;
}

void Renderer::renderGpuTimingsPanel(bool isEditMode)
{
    if (!((isEditMode || playShowDebugUI) && uiShowGpuTimingsWindow))
        return;

    ImGui::Begin("GPU Timings");

    if (!mTimestampsSupported)
    {
        ImGui::TextUnformatted("Timestamps not supported on this queue family.");
        ImGui::End();
        return;
    }

    const float available = ImGui::GetContentRegionAvail().x;
    const float labelWidth = 90.0f;
    const float valueWidth = 70.0f;
    const float barWidth   = available - labelWidth - valueWidth - 12.0f;
    const float maxMs      = std::max(mGpuTimings.totalMs, 0.001f);

    auto passRow = [&](const char* label, float ms, ImVec4 colour) {
        ImGui::TextColored(colour, "%s", label);
        ImGui::SameLine(labelWidth);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, colour);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.15f, 0.15f, 0.15f, 1.0f));
        ImGui::ProgressBar(ms / maxMs, ImVec2(barWidth, ImGui::GetTextLineHeight()), "");
        ImGui::PopStyleColor(2);
        ImGui::SameLine();
        ImGui::Text("%.3f ms", ms);
    };

    passRow("Shadow",  mGpuTimings.shadowMs, ImVec4(0.90f, 0.60f, 0.20f, 1.0f));
    passRow("Scene",   mGpuTimings.sceneMs,  ImVec4(0.30f, 0.70f, 0.90f, 1.0f));
    passRow("Bloom",   mGpuTimings.bloomMs,  ImVec4(0.90f, 0.40f, 0.70f, 1.0f));
    passRow("FXAA",    mGpuTimings.fxaaMs,   ImVec4(0.50f, 0.90f, 0.40f, 1.0f));
    passRow("ImGui",   mGpuTimings.imguiMs,  ImVec4(0.70f, 0.70f, 0.70f, 1.0f));

    ImGui::Separator();

    ImGui::Text("GPU Total   %.3f ms", mGpuTimings.totalMs);

    const ImGuiIO& io = ImGui::GetIO();
    const float cpuMs = io.Framerate > 0.0f ? 1000.0f / io.Framerate : 0.0f;
    ImGui::Text("CPU Frame   %.3f ms  (%.0f FPS)", cpuMs, io.Framerate);

    ImGui::End();
}
