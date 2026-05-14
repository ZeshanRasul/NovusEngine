#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "../components/transform_component.h"
#include "assimp_instance_component.h"
#include "../../model/AssimpInstance.h"
#include "../../model/AssimpModel.h"

namespace AssimpSystems
{
    inline void SyncTransformsFromEntt(
        entt::registry& registry,
        const std::unordered_map<AssimpInstance*, entt::entity>& entityMap)
    {
        for (auto [entity, component, transform] : registry.view<AssimpInstanceComponent, TransformComponent>().each())
        {
            (void)entity;
            if (!component.instance)
                continue;

            auto mapIt = entityMap.find(component.instance.get());
            if (mapIt == entityMap.end() || mapIt->second != entity)
                continue;

            InstanceSettings settings = component.instance->getInstanceSettings();
            settings.isWorldPosition = transform.GetPosition();
            settings.isWorldRotation = glm::degrees(glm::eulerAngles(transform.GetRotation()));
            settings.isScale = transform.GetScale().x;
            component.instance->setInstanceSettings(settings);
        }
    }

    struct ModelInstanceBatch
    {
        std::shared_ptr<AssimpModel> model;
        const std::vector<std::shared_ptr<AssimpInstance>>* instances = nullptr;
        const std::string* modelName = nullptr;
    };

    inline std::vector<ModelInstanceBatch> CollectValidModelInstanceBatches(
        const std::unordered_map<std::string, std::vector<std::shared_ptr<AssimpInstance>>>& instancesPerModel)
    {
        std::vector<ModelInstanceBatch> batches;
        batches.reserve(instancesPerModel.size());

        for (const auto& [modelName, instances] : instancesPerModel)
        {
            if (instances.empty())
                continue;

            const auto& firstInstance = instances.front();
            if (!firstInstance)
                continue;

            auto model = firstInstance->getModel();
            if (!model)
                continue;

            batches.push_back(ModelInstanceBatch{
                .model = model,
                .instances = &instances,
                .modelName = &modelName
            });
        }

        return batches;
    }
}
