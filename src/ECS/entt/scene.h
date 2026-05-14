#pragma once

#include <entt/entt.hpp>
#include <string>
#include <functional>
#include "../components/transform_component.h"
#include "assimp_instance_component.h"

struct EnttTagComponent
{
    std::string name = "Entity";
};

class EnttScene
{
public:
    entt::entity createEntity(const std::string& name = "Entity")
    {
        entt::entity entity = registry.create();
        registry.emplace<EnttTagComponent>(entity, name);
        // Ensure transform and tag are present for new entities
        registry.emplace_or_replace<TransformComponent>(entity);
        return entity;
    }

    void destroyEntity(entt::entity entity)
    {
        if (!registry.valid(entity))
            return;

        // If this entity holds an AssimpInstanceComponent, invoke the registered callback
        // to allow the owner to clean up GPU resources tied to the instance.
        if (assimpInstanceDestroyCallback && registry.any_of<AssimpInstanceComponent>(entity)) {
            auto &comp = registry.get<AssimpInstanceComponent>(entity);
            if (comp.instance) {
                assimpInstanceDestroyCallback(comp.instance);
            }
        }

        registry.destroy(entity);
    }

    bool isValid(entt::entity entity) const
    {
        return registry.valid(entity);
    }

    entt::registry& getRegistry() { return registry; }
    const entt::registry& getRegistry() const { return registry; }

    void setAssimpInstanceDestroyCallback(std::function<void(std::shared_ptr<AssimpInstance>)> cb) {
        assimpInstanceDestroyCallback = std::move(cb);
    }

private:
    entt::registry registry;
    std::function<void(std::shared_ptr<AssimpInstance>)> assimpInstanceDestroyCallback{};
};
