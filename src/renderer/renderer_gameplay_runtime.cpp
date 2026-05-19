#include "renderer/renderer.h"

Gameplay::RuntimeContext Renderer::buildGameplayRuntimeContext()
{
    auto& registry = mEnttScene.getRegistry();

    Gameplay::RuntimeContext ctx{};
    ctx.registry = &registry;
    ctx.physicsSystem = &physicsSystem;

    ctx.destroyEntity = [this](entt::entity entity)
        {
            auto& reg = mEnttScene.getRegistry();
            if (!reg.valid(entity))
                return;

            if (reg.any_of<RigidBodyComponent>(entity))
                physicsSystem.unregisterEntity(entity, reg);

            if (auto* hc = reg.try_get<HierarchyComponent>(entity)) {
                if (hc->parent != entt::null && reg.valid(hc->parent)) {
                    if (auto* parent = reg.try_get<HierarchyComponent>(hc->parent)) {
                        auto it = std::remove(parent->children.begin(), parent->children.end(), entity);
                        parent->children.erase(it, parent->children.end());
                    }
                }
                for (entt::entity child : hc->children) {
                    if (reg.valid(child)) {
                        auto& ch = reg.emplace_or_replace<HierarchyComponent>(child);
                        ch.parent = entt::null;
                    }
                }
            }

            if (reg.any_of<AssimpInstanceComponent>(entity))
                removeInstanceFromEntity(entity);

            if (mEnttScene.isValid(entity))
                mEnttScene.destroyEntity(entity);
        };

    ctx.spawnPrefab = [this](const std::string& prefabPath, const glm::vec3& spawnOffset) -> entt::entity
        {
            if (!instantiatePrefab(prefabPath))
                return entt::null;

            entt::entity spawned = mEnttSelectedEntity;
            auto& reg = mEnttScene.getRegistry();
            if (reg.valid(spawned)) {
                if (auto* tr = reg.try_get<TransformComponent>(spawned))
                    tr->SetPosition(tr->GetPosition() + spawnOffset);
            }
            return spawned;
        };

    return ctx;
}
