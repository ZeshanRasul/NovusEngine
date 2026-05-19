#include "render_settings_io.h"

#include <nlohmann/json.hpp>

std::string saveRenderSettings(const ShadowSettings& shadowSettings, const RenderPresetSettings* renderSettings)
{
    using json = nlohmann::json;
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

    if (renderSettings)
    {
        root["renderSettings"] = {
            { "renderEnableShadows", renderSettings->renderEnableShadows },
            { "renderEnablePostProcessing", renderSettings->renderEnablePostProcessing },
            { "renderEnableFxaa", renderSettings->renderEnableFxaa },
            { "renderEnableBloom", renderSettings->renderEnableBloom },
            { "fxaaExposure", renderSettings->fxaaExposure },
            { "fxaaGamma", renderSettings->fxaaGamma },
            { "bloomEnabled", renderSettings->bloomEnabled },
            { "bloomThreshold", renderSettings->bloomThreshold },
            { "bloomSoftKnee", renderSettings->bloomSoftKnee },
            { "bloomPrefilterScale", renderSettings->bloomPrefilterScale },
            { "bloomIntensity", renderSettings->bloomIntensity },
            { "bloomBlurScale", renderSettings->bloomBlurScale },
            { "bloomBlurPasses", renderSettings->bloomBlurPasses },
            { "postProcessDebugMode", renderSettings->postProcessDebugMode }
        };
    }

    return root.dump(2);
}

bool loadRenderSettings(const std::string& jsonContent, ShadowSettings& shadowSettings, RenderPresetSettings* renderSettings)
{
    using json = nlohmann::json;
    json root = json::parse(jsonContent, nullptr, false);
    if (root.is_discarded() || !root.contains("shadowSettings") || !root["shadowSettings"].is_object())
        return false;

    const json& shadow = root["shadowSettings"];
    shadowSettings.shadowMaxDistance = shadow.value("shadowMaxDistance", shadowSettings.shadowMaxDistance);
    shadowSettings.lambda = shadow.value("lambda", shadowSettings.lambda);
    shadowSettings.biasScale = shadow.value("biasScale", shadowSettings.biasScale);
    shadowSettings.biasMin = shadow.value("biasMin", shadowSettings.biasMin);
    shadowSettings.enabled = shadow.value("enabled", shadowSettings.enabled);
    if (shadow.contains("lightDirection") && shadow["lightDirection"].is_array() && shadow["lightDirection"].size() == 3)
    {
        shadowSettings.lightDirection = glm::vec3(
            shadow["lightDirection"][0].get<float>(),
            shadow["lightDirection"][1].get<float>(),
            shadow["lightDirection"][2].get<float>());
    }
    if (shadow.contains("lightColor") && shadow["lightColor"].is_array() && shadow["lightColor"].size() == 3)
    {
        shadowSettings.lightColor = glm::vec3(
            shadow["lightColor"][0].get<float>(),
            shadow["lightColor"][1].get<float>(),
            shadow["lightColor"][2].get<float>());
    }
    shadowSettings.cascadeBlendFactor = shadow.value("cascadeBlendFactor", shadowSettings.cascadeBlendFactor);
    shadowSettings.cascadeDebugView = shadow.value("cascadeDebugView", shadowSettings.cascadeDebugView);
    shadowSettings.shadowPadding = shadow.value("shadowPadding", shadowSettings.shadowPadding);
    shadowSettings.coveragePaddingFactor = shadow.value("coveragePaddingFactor", shadowSettings.coveragePaddingFactor);
    shadowSettings.depthPaddingFactor = shadow.value("depthPaddingFactor", shadowSettings.depthPaddingFactor);
    shadowSettings.casterPadding = shadow.value("casterPadding", shadowSettings.casterPadding);
    shadowSettings.farCascadeExpansion = shadow.value("farCascadeExpansion", shadowSettings.farCascadeExpansion);

    if (renderSettings && root.contains("renderSettings") && root["renderSettings"].is_object())
    {
        const json& render = root["renderSettings"];
        renderSettings->renderEnableShadows = render.value("renderEnableShadows", renderSettings->renderEnableShadows);
        renderSettings->renderEnablePostProcessing = render.value("renderEnablePostProcessing", renderSettings->renderEnablePostProcessing);
        renderSettings->renderEnableFxaa = render.value("renderEnableFxaa", renderSettings->renderEnableFxaa);
        renderSettings->renderEnableBloom = render.value("renderEnableBloom", renderSettings->renderEnableBloom);
        renderSettings->fxaaExposure = render.value("fxaaExposure", renderSettings->fxaaExposure);
        renderSettings->fxaaGamma = render.value("fxaaGamma", renderSettings->fxaaGamma);
        renderSettings->bloomEnabled = render.value("bloomEnabled", renderSettings->bloomEnabled);
        renderSettings->bloomThreshold = render.value("bloomThreshold", renderSettings->bloomThreshold);
        renderSettings->bloomSoftKnee = render.value("bloomSoftKnee", renderSettings->bloomSoftKnee);
        renderSettings->bloomPrefilterScale = render.value("bloomPrefilterScale", renderSettings->bloomPrefilterScale);
        renderSettings->bloomIntensity = render.value("bloomIntensity", renderSettings->bloomIntensity);
        renderSettings->bloomBlurScale = render.value("bloomBlurScale", renderSettings->bloomBlurScale);
        renderSettings->bloomBlurPasses = render.value("bloomBlurPasses", renderSettings->bloomBlurPasses);
        renderSettings->postProcessDebugMode = render.value("postProcessDebugMode", renderSettings->postProcessDebugMode);
    }

    return true;
}
