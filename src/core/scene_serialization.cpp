#include "core/scene_serialization.h"

#include <algorithm>

#include "ECS/entt/transform_systems.h"
#include "ECS/entt/scene.h"

namespace Core::SceneSerialization
{
    void serializeCommonComponents(const entt::registry& registry, entt::entity entity, json& node)
    {
        if (const auto* transform = registry.try_get<TransformComponent>(entity)) {
            const auto& pos = transform->GetPosition();
            const auto& rot = transform->GetRotation();
            const auto& scale = transform->GetScale();
            node["transform"] = {
                { "position", { pos.x, pos.y, pos.z } },
                { "rotation", { rot.x, rot.y, rot.z, rot.w } },
                { "scale", { scale.x, scale.y, scale.z } }
            };
        }

        if (const auto* animation = registry.try_get<AnimationComponent>(entity)) {
            node["animation"] = {
                { "clipIndex", animation->clipIndex },
                { "speed", animation->speed }
            };
        }

        if (const auto* pointLight = registry.try_get<PointLightComponent>(entity)) {
            node["pointLight"] = {
                { "color", { pointLight->color.x, pointLight->color.y, pointLight->color.z } },
                { "intensity", pointLight->intensity },
                { "range", pointLight->range },
                { "enabled", pointLight->enabled }
            };
        }

        if (const auto* rigidBody = registry.try_get<RigidBodyComponent>(entity)) {
            node["rigidBody"] = {
                { "bodyType", static_cast<int>(rigidBody->bodyType) },
                { "mass", rigidBody->mass },
                { "friction", rigidBody->friction },
                { "restitution", rigidBody->restitution },
                { "useGravity", rigidBody->useGravity },
                { "linearVelocity", { rigidBody->linearVelocity.x, rigidBody->linearVelocity.y, rigidBody->linearVelocity.z } }
            };
        }

        if (const auto* collider = registry.try_get<ColliderComponent>(entity)) {
            node["collider"] = {
                { "shapeType", static_cast<int>(collider->shapeType) },
                { "halfExtents", { collider->halfExtents.x, collider->halfExtents.y, collider->halfExtents.z } },
                { "radius", collider->radius },
                { "halfHeight", collider->halfHeight },
                { "centerOffset", { collider->centerOffset.x, collider->centerOffset.y, collider->centerOffset.z } },
                { "alignBottomToEntity", collider->alignBottomToEntity }
            };
        }

        if (const auto* playerController = registry.try_get<PlayerControllerComponent>(entity)) {
            node["playerController"] = {
                { "enabled", playerController->enabled },
                { "moveSpeed", playerController->moveSpeed },
                { "sprintSpeed", playerController->sprintSpeed },
                { "jumpImpulse", playerController->jumpImpulse }
            };
        }
    }

    void deserializeCommonComponents(entt::registry& registry, entt::entity entity, const json& node)
    {
        if (node.contains("transform")) {
            auto& transform = registry.emplace_or_replace<TransformComponent>(entity);
            const auto& t = node["transform"];
            if (t.contains("position") && t["position"].is_array() && t["position"].size() == 3) {
                transform.SetPosition(glm::vec3(t["position"][0].get<float>(), t["position"][1].get<float>(), t["position"][2].get<float>()));
            }
            if (t.contains("rotation") && t["rotation"].is_array() && t["rotation"].size() == 4) {
                transform.SetRotation(glm::quat(t["rotation"][3].get<float>(), t["rotation"][0].get<float>(), t["rotation"][1].get<float>(), t["rotation"][2].get<float>()));
            }
            if (t.contains("scale") && t["scale"].is_array() && t["scale"].size() == 3) {
                transform.SetScale(glm::vec3(t["scale"][0].get<float>(), t["scale"][1].get<float>(), t["scale"][2].get<float>()));
            }
        }

        if (node.contains("animation")) {
            auto& anim = registry.emplace_or_replace<AnimationComponent>(entity);
            anim.clipIndex = node["animation"].value("clipIndex", 0u);
            anim.speed = node["animation"].value("speed", 1.0f);
        }

        if (node.contains("pointLight") && node["pointLight"].is_object()) {
            auto& light = registry.emplace_or_replace<PointLightComponent>(entity);
            const auto& pl = node["pointLight"];
            if (pl.contains("color") && pl["color"].is_array() && pl["color"].size() == 3)
                light.color = glm::vec3(pl["color"][0].get<float>(), pl["color"][1].get<float>(), pl["color"][2].get<float>());
            light.intensity = pl.value("intensity", light.intensity);
            light.range = pl.value("range", light.range);
            light.enabled = pl.value("enabled", light.enabled);
        }

        if (node.contains("rigidBody") && node["rigidBody"].is_object()) {
            auto& rb = registry.emplace_or_replace<RigidBodyComponent>(entity);
            const auto& rbNode = node["rigidBody"];
            rb.bodyType = static_cast<RigidBodyType>(rbNode.value("bodyType", static_cast<int>(rb.bodyType)));
            rb.mass = rbNode.value("mass", rb.mass);
            rb.friction = rbNode.value("friction", rb.friction);
            rb.restitution = rbNode.value("restitution", rb.restitution);
            rb.useGravity = rbNode.value("useGravity", rb.useGravity);
            if (rbNode.contains("linearVelocity") && rbNode["linearVelocity"].is_array() && rbNode["linearVelocity"].size() == 3) {
                rb.linearVelocity = glm::vec3(rbNode["linearVelocity"][0].get<float>(), rbNode["linearVelocity"][1].get<float>(), rbNode["linearVelocity"][2].get<float>());
            }
            rb.registeredInWorld = false;
            rb.bodyId = -1;
        }

        if (node.contains("collider") && node["collider"].is_object()) {
            auto& col = registry.emplace_or_replace<ColliderComponent>(entity);
            const auto& colNode = node["collider"];
            col.shapeType = static_cast<ColliderShapeType>(colNode.value("shapeType", static_cast<int>(col.shapeType)));
            if (colNode.contains("halfExtents") && colNode["halfExtents"].is_array() && colNode["halfExtents"].size() == 3)
                col.halfExtents = glm::vec3(colNode["halfExtents"][0].get<float>(), colNode["halfExtents"][1].get<float>(), colNode["halfExtents"][2].get<float>());
            col.radius = colNode.value("radius", col.radius);
            col.halfHeight = colNode.value("halfHeight", col.halfHeight);
            if (colNode.contains("centerOffset") && colNode["centerOffset"].is_array() && colNode["centerOffset"].size() == 3)
                col.centerOffset = glm::vec3(colNode["centerOffset"][0].get<float>(), colNode["centerOffset"][1].get<float>(), colNode["centerOffset"][2].get<float>());
            col.alignBottomToEntity = colNode.value("alignBottomToEntity", col.alignBottomToEntity);
        }

        if (node.contains("playerController") && node["playerController"].is_object()) {
            auto& playerController = registry.emplace_or_replace<PlayerControllerComponent>(entity);
            const auto& pcNode = node["playerController"];
            playerController.enabled = pcNode.value("enabled", playerController.enabled);
            playerController.moveSpeed = pcNode.value("moveSpeed", playerController.moveSpeed);
            playerController.sprintSpeed = pcNode.value("sprintSpeed", playerController.sprintSpeed);
            playerController.jumpImpulse = pcNode.value("jumpImpulse", playerController.jumpImpulse);
        }
    }

    PendingHierarchy buildPendingHierarchy(const entt::registry& registry, entt::entity entity, const json& node, bool defaultToRootWhenMissing)
    {
        PendingHierarchy pending{};
        pending.child = entity;

        if (node.contains("hierarchy") && node["hierarchy"].is_object()) {
            const auto& hNode = node["hierarchy"];
            pending.parentId = hNode.value("parent", -1ll);
            if (hNode.contains("localPosition") && hNode["localPosition"].is_array() && hNode["localPosition"].size() == 3 &&
                hNode.contains("localRotation") && hNode["localRotation"].is_array() && hNode["localRotation"].size() == 4 &&
                hNode.contains("localScale") && hNode["localScale"].is_array() && hNode["localScale"].size() == 3) {
                pending.localPosition = glm::vec3(hNode["localPosition"][0].get<float>(), hNode["localPosition"][1].get<float>(), hNode["localPosition"][2].get<float>());
                pending.localRotation = glm::quat(hNode["localRotation"][3].get<float>(), hNode["localRotation"][0].get<float>(), hNode["localRotation"][1].get<float>(), hNode["localRotation"][2].get<float>());
                pending.localScale = glm::vec3(hNode["localScale"][0].get<float>(), hNode["localScale"][1].get<float>(), hNode["localScale"][2].get<float>());
                pending.hasLocal = true;
            }
        }
        else if (defaultToRootWhenMissing) {
            pending.parentId = -1;
        }
        else if (const auto* t = registry.try_get<TransformComponent>(entity)) {
            pending.localPosition = t->GetPosition();
            pending.localRotation = t->GetRotation();
            pending.localScale = t->GetScale();
        }

        return pending;
    }

    void applyPendingHierarchy(entt::registry& registry, const std::vector<PendingHierarchy>& pendingHierarchy, const std::unordered_map<uint32_t, entt::entity>& entityMap)
    {
        for (const auto& entry : pendingHierarchy) {
            auto& childHierarchy = registry.emplace_or_replace<HierarchyComponent>(entry.child);
            childHierarchy.parent = entt::null;
            childHierarchy.children.clear();
            if (entry.hasLocal) {
                childHierarchy.localPosition = entry.localPosition;
                childHierarchy.localRotation = entry.localRotation;
                childHierarchy.localScale = entry.localScale;
            }
        }

        for (const auto& entry : pendingHierarchy) {
            if (!registry.valid(entry.child))
                continue;
            if (entry.parentId < 0)
                continue;

            auto it = entityMap.find(static_cast<uint32_t>(entry.parentId));
            if (it == entityMap.end())
                continue;

            entt::entity parent = it->second;
            if (!registry.valid(parent) || parent == entry.child)
                continue;

            auto& childHierarchy = registry.emplace_or_replace<HierarchyComponent>(entry.child);
            childHierarchy.parent = parent;

            auto& parentHierarchy = registry.emplace_or_replace<HierarchyComponent>(parent);
            if (std::find(parentHierarchy.children.begin(), parentHierarchy.children.end(), entry.child) == parentHierarchy.children.end())
                parentHierarchy.children.push_back(entry.child);
        }

        for (const auto& entry : pendingHierarchy) {
            if (!registry.valid(entry.child))
                continue;
            if (!entry.hasLocal)
                TransformSystems::RecalculateLocalFromParent(registry, entry.child);
        }

        for (const auto& [id, entity] : entityMap) {
            (void)id;
            auto* hierarchy = registry.try_get<HierarchyComponent>(entity);
            if (!hierarchy)
                continue;
            if (hierarchy->parent != entt::null && registry.valid(hierarchy->parent))
                continue;
            TransformSystems::ApplyWorldFromParent(registry, entity);
        }
    }
}
