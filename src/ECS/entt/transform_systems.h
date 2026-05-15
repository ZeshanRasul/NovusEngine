#pragma once

#include <cmath>

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "../components/hierarchy_component.h"
#include "../components/transform_component.h"

namespace TransformSystems
{
	inline glm::vec3 MakeSafeScale(const glm::vec3& scale)
	{
		return glm::vec3(
			std::abs(scale.x) < 1e-5f ? 1.0f : scale.x,
			std::abs(scale.y) < 1e-5f ? 1.0f : scale.y,
			std::abs(scale.z) < 1e-5f ? 1.0f : scale.z);
	}

	inline void RecalculateLocalFromParent(entt::registry& registry, entt::entity child)
	{
		auto* childHierarchy = registry.try_get<HierarchyComponent>(child);
		auto* childTransform = registry.try_get<TransformComponent>(child);
		if (!childHierarchy || !childTransform)
			return;

		if (childHierarchy->parent == entt::null || !registry.valid(childHierarchy->parent))
		{
			childHierarchy->localPosition = childTransform->GetPosition();
			childHierarchy->localRotation = childTransform->GetRotation();
			childHierarchy->localScale = childTransform->GetScale();
			return;
		}

		auto* parentTransform = registry.try_get<TransformComponent>(childHierarchy->parent);
		if (!parentTransform)
			return;

		const glm::vec3 safeParentScale = MakeSafeScale(parentTransform->GetScale());
		childHierarchy->localPosition = glm::inverse(parentTransform->GetRotation()) *
			((childTransform->GetPosition() - parentTransform->GetPosition()) / safeParentScale);
		childHierarchy->localRotation = glm::inverse(parentTransform->GetRotation()) * childTransform->GetRotation();
		childHierarchy->localScale = childTransform->GetScale() / safeParentScale;
	}

	inline void ApplyWorldFromParent(entt::registry& registry, entt::entity parentEntity)
	{
		auto* parentHierarchy = registry.try_get<HierarchyComponent>(parentEntity);
		auto* parentTransform = registry.try_get<TransformComponent>(parentEntity);
		if (!parentHierarchy || !parentTransform)
			return;

		for (const entt::entity child : parentHierarchy->children)
		{
			if (!registry.valid(child))
				continue;

			auto* childTransform = registry.try_get<TransformComponent>(child);
			auto* childHierarchy = registry.try_get<HierarchyComponent>(child);
			if (!childTransform || !childHierarchy)
				continue;

			const glm::vec3 parentScale = parentTransform->GetScale();
			const glm::vec3 worldPos = parentTransform->GetPosition() +
				(parentTransform->GetRotation() * (childHierarchy->localPosition * parentScale));
			const glm::quat worldRot = parentTransform->GetRotation() * childHierarchy->localRotation;
			const glm::vec3 worldScale = parentScale * childHierarchy->localScale;

			childTransform->SetPosition(worldPos);
			childTransform->SetRotation(worldRot);
			childTransform->SetScale(worldScale);

			ApplyWorldFromParent(registry, child);
		}
	}

	inline void UpdateHierarchyFromRoots(entt::registry& registry)
	{
		for (auto [entity, hierarchy] : registry.view<HierarchyComponent>().each())
		{
			if (hierarchy.parent != entt::null && registry.valid(hierarchy.parent))
				continue;

			RecalculateLocalFromParent(registry, entity);
			ApplyWorldFromParent(registry, entity);
		}
	}
}
