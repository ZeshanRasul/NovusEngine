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
		json root = json::parse(jsonContent, nullptr, false);
		if (root.is_discarded() || !root.contains("shadowSettings") || !root["shadowSettings"].is_object()) {
			return false;
		}


		shadowSettings.shadowMaxDistance = root["shadowSettings"]["shadowMaxDistance"];
		shadowSettings.lambda = root["shadowSettings"]["lambda"];
		shadowSettings.biasScale = root["shadowSettings"]["biasScale"];
		shadowSettings.biasMin = root["shadowSettings"]["biasMin"];
		shadowSettings.enabled = root["shadowSettings"]["enabled"];
		shadowSettings.lightDirection = {
			root["shadowSettings"]["lightDirection"][0],
			root["shadowSettings"]["lightDirection"][1],
			root["shadowSettings"]["lightDirection"][2]
		};
		shadowSettings.lightColor = {
			root["shadowSettings"]["lightColor"][0],
			root["shadowSettings"]["lightColor"][1],
			root["shadowSettings"]["lightColor"][2]
		};
		shadowSettings.cascadeBlendFactor = root["shadowSettings"]["cascadeBlendFactor"];
		shadowSettings.cascadeDebugView = root["shadowSettings"]["cascadeDebugView"];
		shadowSettings.shadowPadding = root["shadowSettings"]["shadowPadding"];
		shadowSettings.coveragePaddingFactor = root["shadowSettings"]["coveragePaddingFactor"];
		shadowSettings.depthPaddingFactor = root["shadowSettings"]["depthPaddingFactor"];
		shadowSettings.casterPadding = root["shadowSettings"]["casterPadding"];
		shadowSettings.farCascadeExpansion = root["shadowSettings"]["farCascadeExpansion"];

		return true;
	};
}