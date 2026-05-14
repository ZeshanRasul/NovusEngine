#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include <entt/entt.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>

#include "../components/transform_component.h"
#include "assimp_instance_component.h"
#include "../../model/ModelAndInstanceData.h"
#include "../../model/AssimpInstance.h"
#include "../../model/AssimpModel.h"

namespace AssimpSystems
{
 inline void AttachInstanceComponent(
        entt::registry& registry,
        entt::entity entity,
        const std::shared_ptr<AssimpInstance>& instance)
    {
        auto& assimpComponent = registry.emplace_or_replace<AssimpInstanceComponent>(entity);
        assimpComponent.instance = instance;
    }

    inline void RegisterInstance(
        ModelAndInstanceData& modelData,
        entt::registry& registry,
        entt::entity entity,
        const std::shared_ptr<AssimpInstance>& instance,
        const std::function<void(const std::shared_ptr<AssimpInstance>&)>& onRegistered = {})
    {
        if (!instance || !registry.valid(entity))
            return;

        AttachInstanceComponent(registry, entity, instance);

        if (std::find(modelData.miAssimpInstances.begin(), modelData.miAssimpInstances.end(), instance) == modelData.miAssimpInstances.end())
        {
            modelData.miAssimpInstances.emplace_back(instance);
        }

        auto model = instance->getModel();
        if (model)
        {
            auto& perModelInstances = modelData.miAssimpInstancesPerModel[model->getModelFileName()];
            if (std::find(perModelInstances.begin(), perModelInstances.end(), instance) == perModelInstances.end())
            {
                perModelInstances.emplace_back(instance);
            }
        }

        if (onRegistered)
        {
            onRegistered(instance);
        }
    }

    inline entt::entity FindEntityForInstance(
        entt::registry& registry,
        AssimpInstance* rawInstance)
    {
        if (!rawInstance)
            return entt::null;

        for (auto [entity, component] : registry.view<AssimpInstanceComponent>().each())
        {
            if (component.instance && component.instance.get() == rawInstance)
                return entity;
        }

        return entt::null;
    }

    inline std::shared_ptr<AssimpInstance> FindInstance(
        const ModelAndInstanceData& modelData,
        AssimpInstance* rawInstance)
    {
        auto it = std::find_if(modelData.miAssimpInstances.begin(), modelData.miAssimpInstances.end(),
            [rawInstance](const std::shared_ptr<AssimpInstance>& instance) {
                return instance.get() == rawInstance;
            });

        if (it == modelData.miAssimpInstances.end())
            return nullptr;

        return *it;
    }

    inline void UnregisterInstance(
        ModelAndInstanceData& modelData,
        const std::shared_ptr<AssimpInstance>& instance,
        const std::function<void(const std::shared_ptr<AssimpInstance>&)>& onUnregistering = {})
    {
        if (!instance)
            return;

        if (onUnregistering)
        {
            onUnregistering(instance);
        }

        modelData.miAssimpInstances.erase(
            std::remove(modelData.miAssimpInstances.begin(), modelData.miAssimpInstances.end(), instance),
            modelData.miAssimpInstances.end());

        auto model = instance->getModel();
        if (model)
        {
            auto perModelIt = modelData.miAssimpInstancesPerModel.find(model->getModelFileName());
            if (perModelIt != modelData.miAssimpInstancesPerModel.end())
            {
                auto& perModel = perModelIt->second;
                perModel.erase(std::remove(perModel.begin(), perModel.end(), instance), perModel.end());
                if (perModel.empty())
                {
                    modelData.miAssimpInstancesPerModel.erase(perModelIt);
                }
            }
        }

    }

    inline void SyncTransformsFromEntt(
       entt::registry& registry)
    {
        for (auto [entity, component, transform] : registry.view<AssimpInstanceComponent, TransformComponent>().each())
        {
            (void)entity;
            if (!component.instance)
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
