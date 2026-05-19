#pragma once

#include <string>

#include "../vulkan/uniform_buffer.h"

struct RenderPresetSettings
{
	bool renderEnableShadows = true;
	bool renderEnablePostProcessing = true;
	bool renderEnableFxaa = true;
	bool renderEnableBloom = true;

	float fxaaExposure = 1.0f;
	float fxaaGamma = 2.2f;

	bool bloomEnabled = true;
	float bloomThreshold = 0.13f;
	float bloomSoftKnee = 0.5f;
	float bloomPrefilterScale = 2.0f;
	float bloomIntensity = 0.7f;
	float bloomBlurScale = 1.0f;
	int bloomBlurPasses = 2;

	int postProcessDebugMode = 0;
};

std::string saveRenderSettings(const ShadowSettings& shadowSettings, const RenderPresetSettings* renderSettings = nullptr);

bool loadRenderSettings(const std::string& jsonContent, ShadowSettings& shadowSettings, RenderPresetSettings* renderSettings = nullptr);
