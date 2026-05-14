#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

struct HierarchyComponent
{
    entt::entity parent = entt::null;
    std::vector<entt::entity> children{};
    glm::vec3 localPosition = glm::vec3(0.0f);
    glm::quat localRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 localScale = glm::vec3(1.0f);
};
