#pragma once
#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <array>

void collectPointLights(entt::registry&, std::array<glm::vec4, MAX_POINT_LIGHTS>& positions, std::array<glm::vec4, MAX_POINT_LIGHTS>& colors);