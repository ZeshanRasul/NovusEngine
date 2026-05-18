#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

#include "ECS/components/animation_component.h"
#include "ECS/components/hierarchy_component.h"
#include "ECS/components/light_component.h"
#include "ECS/components/player_controller_component.h"
#include "ECS/components/transform_component.h"
#include "ECS/components/physics_component.h"

namespace Core::SceneSerialization
{
    using json = nlohmann::json;

    struct PendingHierarchy
    {
        entt::entity child = entt::null;
        int64_t parentId = -1;
        glm::vec3 localPosition = glm::vec3(0.0f);
        glm::quat localRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        glm::vec3 localScale = glm::vec3(1.0f);
        bool hasLocal = false;
    };

    void serializeCommonComponents(const entt::registry& registry, entt::entity entity, json& node);
    void deserializeCommonComponents(entt::registry& registry, entt::entity entity, const json& node);

    PendingHierarchy buildPendingHierarchy(const entt::registry& registry, entt::entity entity, const json& node, bool defaultToRootWhenMissing = false);
    void applyPendingHierarchy(entt::registry& registry, const std::vector<PendingHierarchy>& pendingHierarchy, const std::unordered_map<uint32_t, entt::entity>& entityMap);
}
