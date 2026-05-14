#pragma once

#include <glm/glm.hpp>

struct PointLightComponent
{
    glm::vec3 color = glm::vec3(1.0f);
    float intensity = 800.0f;
    float range = 100.0f;
    bool enabled = true;
};
