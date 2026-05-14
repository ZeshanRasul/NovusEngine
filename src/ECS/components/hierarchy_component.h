#pragma once

#include <entt/entt.hpp>
#include <vector>

struct HierarchyComponent
{
    entt::entity parent = entt::null;
    std::vector<entt::entity> children{};
};
