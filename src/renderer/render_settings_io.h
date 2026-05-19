#pragma once
#include <string>
#include "../vulkan/uniform_buffer.h"
#include <json.hpp>

namespace {
	using json = nlohmann::json;

	json saveRenderSettings(const ShadowSettings& shadowSettings)
    {
        json root;

        root["shadowSettings"] = {
            { "shadowMaxDistance", shadowSettings.shadowMaxDistance },
            { "lambda", shadowSettings.lambda },
            { "biasScale", shadowSettings.biasScale },
            { "biasMin", shadowSettings.biasMin },
            { "enabled", shadowSettings.enabled },
            { "lightDirection", { shadowSettings.lightDirection.x, shadowSettings.lightDirection.y, shadowSettings.lightDirection.z } },
            { "lightColor", { shadowSettings.lightColor.x, shadowSettings.lightColor.y, shadowSettings.lightColor.z } },
            { "cascadeBlendFactor", shadowSettings.cascadeBlendFactor },
            { "cascadeDebugView", shadowSettings.cascadeDebugView },
            { "shadowPadding", shadowSettings.shadowPadding },
            { "coveragePaddingFactor", shadowSettings.coveragePaddingFactor },
            { "depthPaddingFactor", shadowSettings.depthPaddingFactor },
            { "casterPadding", shadowSettings.casterPadding },
            { "farCascadeExpansion", shadowSettings.farCascadeExpansion }
        };

        return root.dump(2);
    };

    bool loadRenderSettings(const std::string& jsonContent, ShadowSettings& shadowSettings)
    {
        return true;
    };
}