#include "pass_common.h"
#include "../../ECS/components/light_component.h"
#include "../../ECS/components/transform_component.h"

void collectPointLights(entt::registry& registry, std::array<glm::vec4, MAX_POINT_LIGHTS>& positions, std::array<glm::vec4, MAX_POINT_LIGHTS>& colors)
{
    for (size_t i = 0; i < MAX_POINT_LIGHTS; ++i)
    {
        registry.view<PointLightComponent>().each([&](auto entity, const PointLightComponent& plc)
        {
            (void)entity;
            if (i < MAX_POINT_LIGHTS)
            {
                auto& pos = registry.get<TransformComponent>(entity).GetPosition();
                positions[i] = glm::vec4(pos, 1.0f);
                colors[i] = glm::vec4(plc.color * plc.intensity, 1.0f);
                ++i;
            }
			});

	}
}
