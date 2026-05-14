#pragma once

#include <entt/entt.hpp>
#include <string>

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
        return entity;
    }

    void destroyEntity(entt::entity entity)
    {
        if (registry.valid(entity))
            registry.destroy(entity);
    }

    bool isValid(entt::entity entity) const
    {
        return registry.valid(entity);
    }

    entt::registry& getRegistry() { return registry; }
    const entt::registry& getRegistry() const { return registry; }

private:
    entt::registry registry;
};
