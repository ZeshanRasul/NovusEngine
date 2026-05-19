#include "renderer/renderer.h"

#include <glm/gtc/type_ptr.hpp>

namespace {
	std::string makeUniqueEntityNameLocal(entt::registry& registry, const std::string& baseName)
	{
		auto nameExists = [&](const std::string& candidate) {
			for (auto [entity, tag] : registry.view<EnttTagComponent>().each())
			{
				(void)entity;
				if (tag.name == candidate)
					return true;
			}
			return false;
			};

		if (!nameExists(baseName))
			return baseName;

		for (int suffix = 2; ; ++suffix)
		{
			std::string candidate = baseName + " (" + std::to_string(suffix) + ")";
			if (!nameExists(candidate))
				return candidate;
		}
	}
}

void Renderer::renderEnttEditor(glm::mat4 view, glm::mat4 projection)
{
	auto& registry = mEnttScene.getRegistry();
	static entt::entity sNameEditEntity = entt::null;
	static char sNameEditBuffer[256]{};
	auto isInMultiSelection = [&](entt::entity entity) {
		return std::find(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), entity) != mEnttMultiSelection.end();
		};
	auto removeFromMultiSelection = [&](entt::entity entity) {
		auto it = std::remove(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), entity);
		if (it != mEnttMultiSelection.end())
			mEnttMultiSelection.erase(it, mEnttMultiSelection.end());
		};
	auto detachHierarchy = [&](entt::entity entity) {
		auto* hc = registry.try_get<HierarchyComponent>(entity);
		if (!hc)
			return;

		if (hc->parent != entt::null && registry.valid(hc->parent)) {
			auto* parentHc = registry.try_get<HierarchyComponent>(hc->parent);
			if (parentHc) {
				auto it = std::remove(parentHc->children.begin(), parentHc->children.end(), entity);
				if (it != parentHc->children.end())
					parentHc->children.erase(it, parentHc->children.end());
			}
		}

		for (auto child : hc->children) {
			if (!registry.valid(child))
				continue;
			auto& childHc = registry.emplace_or_replace<HierarchyComponent>(child);
			childHc.parent = entt::null;
		}
		hc->children.clear();
		hc->parent = entt::null;
		};
	auto isDescendantOf = [&](entt::entity child, entt::entity possibleAncestor) {
		entt::entity cursor = child;
		while (cursor != entt::null && registry.valid(cursor)) {
			auto* hc = registry.try_get<HierarchyComponent>(cursor);
			if (!hc)
				return false;
			cursor = hc->parent;
			if (cursor == possibleAncestor)
				return true;
		}
		return false;
		};
	auto recalcLocalFromParent = [&](entt::entity child) {
		TransformSystems::RecalculateLocalFromParent(registry, child);
		};
	std::function<void(entt::entity)> applyWorldFromParent = [&](entt::entity parentEntity) {
		TransformSystems::ApplyWorldFromParent(registry, parentEntity);
		};
	auto setParent = [&](entt::entity child, entt::entity newParent) {
		if (!registry.valid(child))
			return;
		if (newParent == child)
			return;
		if (newParent != entt::null && isDescendantOf(newParent, child))
			return;

		auto& childHc = registry.emplace_or_replace<HierarchyComponent>(child);
		if (childHc.parent != entt::null && registry.valid(childHc.parent)) {
			auto* oldParentHc = registry.try_get<HierarchyComponent>(childHc.parent);
			if (oldParentHc) {
				auto it = std::remove(oldParentHc->children.begin(), oldParentHc->children.end(), child);
				if (it != oldParentHc->children.end())
					oldParentHc->children.erase(it, oldParentHc->children.end());
			}
		}

		childHc.parent = newParent;
		if (newParent != entt::null && registry.valid(newParent)) {
			auto& parentHc = registry.emplace_or_replace<HierarchyComponent>(newParent);
			if (std::find(parentHc.children.begin(), parentHc.children.end(), child) == parentHc.children.end()) {
				parentHc.children.push_back(child);
			}
		}

		recalcLocalFromParent(child);
		if (newParent != entt::null)
			applyWorldFromParent(newParent);
		};
	auto selectEntityAndSync = [&](entt::entity entity) {
		mEnttSelectedEntity = entity;
		mEnttMultiSelection.clear();
		mEnttMultiSelection.push_back(entity);
		sNameEditEntity = entt::null;

		if (auto* assimp = registry.try_get<AssimpInstanceComponent>(entity); assimp && assimp->instance) {
			auto instanceIt = std::find(mModelInstData.miAssimpInstances.begin(), mModelInstData.miAssimpInstances.end(), assimp->instance);
			if (instanceIt != mModelInstData.miAssimpInstances.end()) {
				mModelInstData.miSelectedInstance = static_cast<int>(std::distance(mModelInstData.miAssimpInstances.begin(), instanceIt));
			}

			auto model = assimp->instance->getModel();
			if (model) {
				auto modelIt = std::find(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(), model);
				if (modelIt != mModelInstData.miModelList.end()) {
					mModelInstData.miSelectedModel = static_cast<int>(std::distance(mModelInstData.miModelList.begin(), modelIt));
				}
			}
		}
		};

	mEnttMultiSelection.erase(
		std::remove_if(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), [&](entt::entity e) {
			return !mEnttScene.isValid(e);
			}),
		mEnttMultiSelection.end());
	std::sort(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), [](entt::entity a, entt::entity b) {
		return entt::to_integral(a) < entt::to_integral(b);
		});
	mEnttMultiSelection.erase(std::unique(mEnttMultiSelection.begin(), mEnttMultiSelection.end()), mEnttMultiSelection.end());

	if (mEnttScene.isValid(mEnttSelectedEntity) && mEnttMultiSelection.empty()) {
		mEnttMultiSelection.push_back(mEnttSelectedEntity);
	}

	TransformSystems::UpdateHierarchyFromRoots(registry);

	auto duplicateEntity = [&](entt::entity sourceEntity) -> entt::entity {
		if (!mEnttScene.isValid(sourceEntity))
			return entt::null;

		auto* sourceTag = registry.try_get<EnttTagComponent>(sourceEntity);
		auto* sourceTransform = registry.try_get<TransformComponent>(sourceEntity);
		auto* sourceAnimation = registry.try_get<AnimationComponent>(sourceEntity);
		auto* sourceAssimp = registry.try_get<AssimpInstanceComponent>(sourceEntity);

		if (sourceAssimp && sourceAssimp->instance) {
			size_t previousInstanceCount = mModelInstData.miAssimpInstances.size();
			cloneInstance(sourceAssimp->instance);
			if (mModelInstData.miAssimpInstances.size() <= previousInstanceCount)
				return entt::null;

			auto duplicatedInstance = mModelInstData.miAssimpInstances.back();
			entt::entity duplicatedEntity = AssimpSystems::FindEntityForInstance(registry, duplicatedInstance.get());
			if (duplicatedEntity == entt::null)
				return entt::null;

			if (auto* duplicatedTag = registry.try_get<EnttTagComponent>(duplicatedEntity); duplicatedTag && sourceTag) {
				duplicatedTag->name = makeUniqueEntityNameLocal(registry, sourceTag->name + " Copy");
			}

			return duplicatedEntity;
		}

		std::string duplicatedName = sourceTag ? makeUniqueEntityNameLocal(registry, sourceTag->name + " Copy") : makeUniqueEntityNameLocal(registry, "Entity Copy");
		entt::entity duplicatedEntity = mEnttScene.createEntity(duplicatedName);

		if (sourceTransform) {
			auto& duplicatedTransform = registry.emplace_or_replace<TransformComponent>(duplicatedEntity);
			duplicatedTransform.SetPosition(sourceTransform->GetPosition() + glm::vec3(1.0f, 0.0f, -1.0f));
			duplicatedTransform.SetRotation(sourceTransform->GetRotation());
			duplicatedTransform.SetScale(sourceTransform->GetScale());
		}

		if (sourceAnimation) {
			registry.emplace_or_replace<AnimationComponent>(duplicatedEntity, *sourceAnimation);
		}

		return duplicatedEntity;
		};

	ImGui::Begin("ECS Scene");
	if (ImGui::Button("Undo"))
		performUndo();
	ImGui::SameLine();
	if (ImGui::Button("Redo"))
		performRedo();
	ImGui::SameLine();
	if (ImGui::Button("Save Scene")) {
		mSceneRuntimeService.saveScene([this]() { return serializeEnttScene(); });
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Scene")) {
		mSceneRuntimeService.loadScene([this](const std::string& jsonContent) {
			pushUndoSnapshot();
			return deserializeEnttScene(jsonContent);
			});
	}

	if (ImGui::Button("Create Entity"))
	{
		pushUndoSnapshot();
		mEnttSelectedEntity = mEnttScene.createEntity("New Entity");
		mEnttMultiSelection.clear();
		mEnttMultiSelection.push_back(mEnttSelectedEntity);
	}	ImGui::Separator();

	registry.view<EnttTagComponent>().each([&](entt::entity entity, EnttTagComponent& tag)
		{
			ImGui::PushID(static_cast<int>(entt::to_integral(entity)));
			const bool isSelected = isInMultiSelection(entity);
			if (ImGui::Selectable(tag.name.c_str(), isSelected)) {
				const bool ctrlDown = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl) || ImGui::GetIO().KeyCtrl;
				if (ctrlDown) {
					if (isInMultiSelection(entity)) {
						removeFromMultiSelection(entity);
						if (mEnttSelectedEntity == entity) {
							mEnttSelectedEntity = mEnttMultiSelection.empty() ? entt::null : mEnttMultiSelection.back();
						}
					}
					else {
						mEnttMultiSelection.push_back(entity);
						mEnttSelectedEntity = entity;
					}
				}
				else {
					selectEntityAndSync(entity);
				}
			}
			ImGui::PopID();
		});
	ImGui::Text("Selected: %d", static_cast<int>(mEnttMultiSelection.size()));
	ImGui::End();

	ImGui::Begin("ECS Lights");
	if (ImGui::Button("Create Point Light")) {
		pushUndoSnapshot();
		auto lightEntity = mEnttScene.createEntity("Point Light");
		auto& lightTransform = registry.emplace_or_replace<TransformComponent>(lightEntity);
		lightTransform.SetPosition(camera.getPosition() + camera.getFront() * 8.0f);
		auto& light = registry.emplace_or_replace<PointLightComponent>(lightEntity);
		light.color = glm::vec3(255.0f);
		light.intensity = 800.0f;
		light.range = 100.0f;
		light.enabled = true;
		selectEntityAndSync(lightEntity);
	}

	entt::entity lightEntityToDelete = entt::null;
	for (auto [entity, tag, light, transform] : registry.view<EnttTagComponent, PointLightComponent, TransformComponent>().each())
	{
		ImGui::PushID(static_cast<int>(entt::to_integral(entity)));
		const bool selected = (mEnttSelectedEntity == entity);
		if (ImGui::Selectable(tag.name.c_str(), selected)) {
			selectEntityAndSync(entity);
		}
		ImGui::SameLine();
		ImGui::Checkbox("##Enabled", &light.enabled);
		ImGui::SameLine();
		ImGui::Text("I: %.0f", light.intensity);
		ImGui::SameLine();
		if (ImGui::SmallButton("Delete")) {
			lightEntityToDelete = entity;
		}
		ImGui::PopID();
	}

	if (lightEntityToDelete != entt::null && mEnttScene.isValid(lightEntityToDelete)) {
		pushUndoSnapshot();
		detachHierarchy(lightEntityToDelete);
		mEnttScene.destroyEntity(lightEntityToDelete);
		if (mEnttSelectedEntity == lightEntityToDelete) {
			mEnttSelectedEntity = entt::null;
			mEnttMultiSelection.clear();
		}
	}
	ImGui::End();

	ImGui::SetNextWindowBgAlpha(1.0f);
	ImGui::Begin("ECS Inspector");
	if (mEnttScene.isValid(mEnttSelectedEntity))
	{
		auto* tag = registry.try_get<EnttTagComponent>(mEnttSelectedEntity);
		auto* transform = registry.try_get<TransformComponent>(mEnttSelectedEntity);
		auto* animation = registry.try_get<AnimationComponent>(mEnttSelectedEntity);

		if (tag)
		{
			if (sNameEditEntity != mEnttSelectedEntity) {
				sNameEditEntity = mEnttSelectedEntity;
				std::snprintf(sNameEditBuffer, sizeof(sNameEditBuffer), "%s", tag->name.c_str());
			}

			const bool nameEdited = ImGui::InputText("Name", sNameEditBuffer, sizeof(sNameEditBuffer));
			if (nameEdited) {
				tag->name = sNameEditBuffer;
			}
			if (ImGui::IsItemDeactivatedAfterEdit()) {
				pushUndoSnapshot();
			}
		}

		if (transform)
		{

			if (ImGui::IsKeyPressed(ImGuiKey_T))
				currentOperation = ImGuizmo::OPERATION::TRANSLATE;
			if (ImGui::IsKeyPressed(ImGuiKey_E))
				currentOperation = ImGuizmo::OPERATION::ROTATE;
			if (ImGui::IsKeyPressed(ImGuiKey_S))
				currentOperation = ImGuizmo::OPERATION::SCALE;
			if (ImGui::RadioButton("Translate", currentOperation == ImGuizmo::OPERATION::TRANSLATE))
			{
				currentOperation = ImGuizmo::OPERATION::TRANSLATE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", currentOperation == ImGuizmo::OPERATION::ROTATE))
			{
				currentOperation = ImGuizmo::OPERATION::ROTATE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", currentOperation == ImGuizmo::OPERATION::SCALE))
			{
				currentOperation = ImGuizmo::OPERATION::SCALE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("None", currentOperation == -1))
				ImGuizmo::Enable(false);

			glm::vec3 position = transform->GetPosition();
			glm::vec3 rotation = glm::eulerAngles(transform->GetRotation());
			glm::vec3 scale = transform->GetScale();

			if (ImGui::DragFloat3("Translation", &position.x, 0.1f))
				transform->SetPosition(position);
			if (ImGui::DragFloat3("Rotation", &rotation.x, 0.1f))
				transform->SetRotation(glm::quat(rotation));
			if (ImGui::DragFloat3("Scale", &scale.x, 0.1f, 0.01f, 100.0f))
				transform->SetScale(scale);

			glm::mat4 transformMatrix = transform->GetTransformMatrix();
			glm::mat4 gizmoProjection = projection;
			gizmoProjection[1][1] *= -1.0f;
			const bool manipulated = ImGuizmo::Manipulate(glm::value_ptr(view),
				glm::value_ptr(gizmoProjection),
				currentOperation,
				currentMode,
				glm::value_ptr(transformMatrix));

			if ((manipulated || ImGuizmo::IsUsing()) && currentOperation != static_cast<ImGuizmo::OPERATION>(-1))
			{
				glm::vec3 pos, rot, scale;
				ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transformMatrix),
					glm::value_ptr(pos),
					glm::value_ptr(rot),
					glm::value_ptr(scale));

				switch (currentOperation)
				{
				case ImGuizmo::OPERATION::TRANSLATE:
					transform->SetPosition(pos);
					break;
				case ImGuizmo::OPERATION::ROTATE:
					transform->SetRotation(glm::quat(glm::radians(rot)));
					break;
				case ImGuizmo::OPERATION::SCALE:
					transform->SetScale(scale);
					break;
				default:
					break;
				}

				if (auto* assimp = registry.try_get<AssimpInstanceComponent>(mEnttSelectedEntity); assimp && assimp->instance)
				{
					InstanceSettings settings = assimp->instance->getInstanceSettings();
					settings.isWorldPosition = transform->GetPosition();
					settings.isWorldRotation = glm::degrees(glm::eulerAngles(transform->GetRotation()));
					settings.isScale = transform->GetScale().x;
					assimp->instance->setInstanceSettings(settings);
				}
			}
		}

		ImGui::Separator();
		ImGui::TextUnformatted("Components");
		const bool hasAnimationComp = registry.any_of<AnimationComponent>(mEnttSelectedEntity);
		if (!hasAnimationComp) {
			if (ImGui::Button("Add AnimationComponent")) {
				pushUndoSnapshot();
				registry.emplace_or_replace<AnimationComponent>(mEnttSelectedEntity);
				animation = registry.try_get<AnimationComponent>(mEnttSelectedEntity);
			}
		}
		else {
			if (ImGui::Button("Remove AnimationComponent")) {
				pushUndoSnapshot();
				registry.remove<AnimationComponent>(mEnttSelectedEntity);
				animation = nullptr;
			}
		}

		if (!registry.any_of<HierarchyComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add HierarchyComponent")) {
				pushUndoSnapshot();
				registry.emplace_or_replace<HierarchyComponent>(mEnttSelectedEntity);
			}
		}
		else {
			if (ImGui::Button("Remove HierarchyComponent")) {
				pushUndoSnapshot();
				detachHierarchy(mEnttSelectedEntity);
				registry.remove<HierarchyComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<PointLightComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add PointLightComponent")) {
				pushUndoSnapshot();
				auto& light = registry.emplace_or_replace<PointLightComponent>(mEnttSelectedEntity);
				light.color = glm::vec3(1.0f);
				light.intensity = 800.0f;
				light.range = 100.0f;
				light.enabled = true;
			}
		}
		else {
			if (ImGui::Button("Remove PointLightComponent")) {
				pushUndoSnapshot();
				registry.remove<PointLightComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<RigidBodyComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add RigidBodyComponent")) {
				pushUndoSnapshot();
				auto& rb = registry.emplace_or_replace<RigidBodyComponent>(mEnttSelectedEntity);
				rb.bodyType = RigidBodyType::Dynamic;
				rb.mass = 1.0f;
				rb.friction = 0.5f;
				rb.restitution = 0.2f;
				rb.useGravity = true;
				rb.linearVelocity = glm::vec3(0.0f);
				rb.registeredInWorld = false;
				rb.bodyId = -1;

				if (!registry.any_of<ColliderComponent>(mEnttSelectedEntity)) {
					registry.emplace_or_replace<ColliderComponent>(mEnttSelectedEntity);
				}

				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.registerEntity(entity, registry);
			}
		}
		else {
			if (ImGui::Button("Remove RigidBodyComponent")) {
				pushUndoSnapshot();
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				registry.remove<RigidBodyComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<ColliderComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add ColliderComponent")) {
				pushUndoSnapshot();
				auto& col = registry.emplace_or_replace<ColliderComponent>(mEnttSelectedEntity);
				col.shapeType = ColliderShapeType::Box;
				col.halfExtents = glm::vec3(0.5f);
				col.radius = 0.5f;
				col.halfHeight = 0.5f;
				col.centerOffset = glm::vec3(0.0f);
				col.alignBottomToEntity = false;

				if (registry.any_of<RigidBodyComponent>(mEnttSelectedEntity)) {
					entt::entity entity = mEnttSelectedEntity;
					physicsSystem.registerEntity(entity, registry);
				}
			}
		}
		else {
			if (ImGui::Button("Remove ColliderComponent")) {
				pushUndoSnapshot();
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				registry.remove<ColliderComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<PlayerControllerComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add PlayerControllerComponent")) {
				pushUndoSnapshot();
				registry.emplace_or_replace<PlayerControllerComponent>(mEnttSelectedEntity);
			}
		}
		else {
			if (ImGui::Button("Remove PlayerControllerComponent")) {
				pushUndoSnapshot();
				registry.remove<PlayerControllerComponent>(mEnttSelectedEntity);
			}
		}

		if (auto* hierarchy = registry.try_get<HierarchyComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Hierarchy");

			const char* currentParentName = "None";
			if (hierarchy->parent != entt::null && registry.valid(hierarchy->parent)) {
				if (auto* parentTag = registry.try_get<EnttTagComponent>(hierarchy->parent)) {
					currentParentName = parentTag->name.c_str();
				}
			}
			ImGui::Text("Parent: %s", currentParentName);

			if (ImGui::Button("Detach Parent")) {
				pushUndoSnapshot();
				setParent(mEnttSelectedEntity, entt::null);
			}

			if (ImGui::BeginCombo("Set Parent", currentParentName)) {
				if (ImGui::Selectable("None", hierarchy->parent == entt::null)) {
					pushUndoSnapshot();
					setParent(mEnttSelectedEntity, entt::null);
				}

				for (auto [candidate, candidateTag] : registry.view<EnttTagComponent>().each()) {
					if (candidate == mEnttSelectedEntity)
						continue;
					const bool isSelectedParent = (candidate == hierarchy->parent);
					if (ImGui::Selectable(candidateTag.name.c_str(), isSelectedParent)) {
						pushUndoSnapshot();
						setParent(mEnttSelectedEntity, candidate);
					}
				}
				ImGui::EndCombo();
			}

			ImGui::Text("Children: %d", static_cast<int>(hierarchy->children.size()));
		}

		if (auto* pointLight = registry.try_get<PointLightComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Point Light");
			ImGui::Checkbox("Light Enabled", &pointLight->enabled);
			ImGui::ColorEdit3("Light Color", &pointLight->color.x);
			ImGui::DragFloat("Light Intensity", &pointLight->intensity, 1.0f, 0.0f, 100000.0f, "%.1f");
			ImGui::DragFloat("Light Range", &pointLight->range, 0.1f, 0.0f, 100000.0f, "%.2f");
		}

		if (auto* rigidBody = registry.try_get<RigidBodyComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Rigid Body");

			static const char* bodyTypeLabels[] = { "Static", "Dynamic", "Kinematic" };
			int bodyType = static_cast<int>(rigidBody->bodyType);
			bool rebuildBody = false;

			if (ImGui::Combo("Body Type", &bodyType, bodyTypeLabels, IM_ARRAYSIZE(bodyTypeLabels))) {
				rigidBody->bodyType = static_cast<RigidBodyType>(std::clamp(bodyType, 0, 2));
				rebuildBody = true;
			}

			if (ImGui::DragFloat("Mass", &rigidBody->mass, 0.05f, 0.001f, 10000.0f, "%.3f")) {
				rigidBody->mass = std::max(0.001f, rigidBody->mass);
				rebuildBody = true;
			}
			if (ImGui::DragFloat("Friction", &rigidBody->friction, 0.01f, 0.0f, 10.0f, "%.3f")) {
				rigidBody->friction = std::clamp(rigidBody->friction, 0.0f, 10.0f);
				rebuildBody = true;
			}
			if (ImGui::DragFloat("Restitution", &rigidBody->restitution, 0.01f, 0.0f, 1.0f, "%.3f")) {
				rigidBody->restitution = std::clamp(rigidBody->restitution, 0.0f, 1.0f);
				rebuildBody = true;
			}
			if (ImGui::Checkbox("Use Gravity", &rigidBody->useGravity)) {
				rebuildBody = true;
			}
			if (ImGui::DragFloat3("Linear Velocity", &rigidBody->linearVelocity.x, 0.05f, -10000.0f, 10000.0f, "%.3f")) {
				rebuildBody = true;
			}

			if (rebuildBody && registry.any_of<ColliderComponent>(mEnttSelectedEntity) && registry.any_of<TransformComponent>(mEnttSelectedEntity)) {
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				physicsSystem.registerEntity(entity, registry);
			}
		}

		if (auto* collider = registry.try_get<ColliderComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Collider");

			static const char* shapeTypeLabels[] = { "Box", "Sphere", "Capsule" };
			int shapeType = static_cast<int>(collider->shapeType);
			bool rebuildBody = false;

			if (ImGui::Combo("Shape", &shapeType, shapeTypeLabels, IM_ARRAYSIZE(shapeTypeLabels))) {
				collider->shapeType = static_cast<ColliderShapeType>(std::clamp(shapeType, 0, 2));
				rebuildBody = true;
			}

			if (collider->shapeType == ColliderShapeType::Box) {
				if (ImGui::DragFloat3("Half Extents", &collider->halfExtents.x, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->halfExtents = glm::max(collider->halfExtents, glm::vec3(0.001f));
					rebuildBody = true;
				}
			}
			else if (collider->shapeType == ColliderShapeType::Sphere) {
				if (ImGui::DragFloat("Radius", &collider->radius, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->radius = std::max(0.001f, collider->radius);
					rebuildBody = true;
				}
			}
			else {
				if (ImGui::DragFloat("Radius", &collider->radius, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->radius = std::max(0.001f, collider->radius);
					rebuildBody = true;
				}
				if (ImGui::DragFloat("Half Height", &collider->halfHeight, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->halfHeight = std::max(0.001f, collider->halfHeight);
					rebuildBody = true;
				}
			}

			if (ImGui::DragFloat3("Center Offset", &collider->centerOffset.x, 0.05f, -10000.0f, 10000.0f, "%.3f")) {
				rebuildBody = true;
			}

			if (ImGui::Checkbox("Align Bottom To Entity", &collider->alignBottomToEntity)) {
				rebuildBody = true;
			}

			if (rebuildBody && registry.any_of<RigidBodyComponent>(mEnttSelectedEntity) && registry.any_of<TransformComponent>(mEnttSelectedEntity)) {
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				physicsSystem.registerEntity(entity, registry);
			}
		}

		ImGui::Separator();
		ImGui::TextUnformatted("Gameplay");
		if (mGameplayRuntime.hasActiveLayer())
			ImGui::TextUnformatted("Layer: DefaultGameLayer");
		else
			ImGui::TextUnformatted("Layer: None");
		bool useDefaultLayer = mGameplayRuntime.useDefaultLayer();
		if (ImGui::Checkbox("Use Default Gameplay Layer", &useDefaultLayer))
			mGameplayRuntime.setUseDefaultLayer(useDefaultLayer);

		if (auto* controller = registry.try_get<PlayerControllerComponent>(mEnttSelectedEntity)) {
			ImGui::Checkbox("Controller Enabled", &controller->enabled);
			ImGui::DragFloat("Move Speed", &controller->moveSpeed, 0.05f, 0.0f, 1000.0f, "%.3f");
			ImGui::DragFloat("Sprint Speed", &controller->sprintSpeed, 0.05f, 0.0f, 1000.0f, "%.3f");
			ImGui::DragFloat("Jump Impulse", &controller->jumpImpulse, 0.05f, 0.0f, 1000.0f, "%.3f");
		}

		if (!registry.any_of<AssimpInstanceComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::Text("Assimp Instance:");
			if (mModelInstData.miModelList.empty()) {
				ImGui::Text("No models loaded. Import a model first.");
			}
			else {
				static int selModelIdx = 0;
				std::vector<const char*> modelNames;
				modelNames.reserve(mModelInstData.miModelList.size());
				for (auto& m : mModelInstData.miModelList) modelNames.push_back(m->getModelFileName().c_str());
				ImGui::PushItemWidth(200);
				if (ImGui::Combo("Model to Add", &selModelIdx, modelNames.data(), static_cast<int>(modelNames.size()))) {}
				ImGui::PopItemWidth();
				ImGui::SameLine();
				if (ImGui::Button("Add Instance")) {
					pushUndoSnapshot();
					auto model = mModelInstData.miModelList.at(selModelIdx);
					createAssimpInstanceForEntity(model, mEnttSelectedEntity);
				}
			}
		}
		else {
			ImGui::Separator();
			ImGui::Text("Assimp Instance Attached");
			auto& comp = registry.get<AssimpInstanceComponent>(mEnttSelectedEntity);
			if (comp.instance) {
				InstanceSettings instSet = comp.instance->getInstanceSettings();
				if (ImGui::DragFloat3("Instance Pos", glm::value_ptr(instSet.isWorldPosition), 0.1f)) {
					comp.instance->setTranslation(instSet.isWorldPosition);
				}
				glm::vec3 rot = instSet.isWorldRotation;
				if (ImGui::DragFloat3("Instance Rot", glm::value_ptr(rot), 1.0f)) {
					comp.instance->setRotation(rot);
				}
				if (ImGui::DragFloat("Instance Scale", &instSet.isScale, 0.01f, 0.001f, 100.0f)) {
					comp.instance->setScale(instSet.isScale);
				}

				auto model = comp.instance->getModel();
				if (animation && model) {
					auto animClips = model->getAnimClips();
					if (!animClips.empty()) {
						if (animation->clipIndex >= animClips.size()) {
							animation->clipIndex = static_cast<unsigned int>(animClips.size() - 1);
						}

						std::vector<std::string> clipNameStorage;
						clipNameStorage.reserve(animClips.size());
						for (const auto& clip : animClips) {
							clipNameStorage.push_back(clip->getClipName());
						}

						std::vector<const char*> clipNames;
						clipNames.reserve(clipNameStorage.size());
						for (const auto& clipName : clipNameStorage) {
							clipNames.push_back(clipName.c_str());
						}

						int clipIndex = static_cast<int>(animation->clipIndex);
						if (ImGui::Combo("Animation Clip", &clipIndex, clipNames.data(), static_cast<int>(clipNames.size()))) {
							animation->clipIndex = static_cast<unsigned int>(clipIndex);
							InstanceSettings updatedSettings = comp.instance->getInstanceSettings();
							updatedSettings.isAnimClipNr = animation->clipIndex;
							comp.instance->setInstanceSettings(updatedSettings);
						}

						float speed = animation->speed;
						if (ImGui::SliderFloat("Animation Speed", &speed, 0.0f, 2.0f, "%.3f")) {
							animation->speed = speed;
							InstanceSettings updatedSettings = comp.instance->getInstanceSettings();
							updatedSettings.isAnimSpeedFactor = animation->speed;
							comp.instance->setInstanceSettings(updatedSettings);
						}
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Remove Instance")) {
					pushUndoSnapshot();
					removeInstanceFromEntity(mEnttSelectedEntity);
				}
			}
		}

		const bool duplicatePressed = ImGui::Button("Duplicate Entity");
		ImGui::SameLine();
		const bool deletePressed = ImGui::Button("Delete Entity");

		if (duplicatePressed) {
			std::vector<entt::entity> sources = mEnttMultiSelection;
			if (sources.empty() && mEnttScene.isValid(mEnttSelectedEntity)) {
				sources.push_back(mEnttSelectedEntity);
			}

			if (!sources.empty()) {
				pushUndoSnapshot();
				std::vector<entt::entity> duplicatedEntities;
				duplicatedEntities.reserve(sources.size());
				for (entt::entity source : sources) {
					entt::entity duplicated = duplicateEntity(source);
					if (duplicated != entt::null && mEnttScene.isValid(duplicated)) {
						duplicatedEntities.push_back(duplicated);
					}
				}

				if (!duplicatedEntities.empty()) {
					mEnttSelectedEntity = duplicatedEntities.back();
					mEnttMultiSelection = std::move(duplicatedEntities);
					sNameEditEntity = entt::null;
				}
			}
		}

		if (deletePressed) {
			std::vector<entt::entity> toDelete = mEnttMultiSelection;
			if (toDelete.empty() && mEnttScene.isValid(mEnttSelectedEntity)) {
				toDelete.push_back(mEnttSelectedEntity);
			}

			if (!toDelete.empty()) {
				pushUndoSnapshot();
				for (entt::entity entity : toDelete) {
					if (!mEnttScene.isValid(entity))
						continue;

					detachHierarchy(entity);

					if (registry.any_of<RigidBodyComponent>(entity)) {
						physicsSystem.unregisterEntity(entity, registry);
					}

					if (registry.any_of<AssimpInstanceComponent>(entity)) {
						removeInstanceFromEntity(entity);
					}

					if (mEnttScene.isValid(entity)) {
						mEnttScene.destroyEntity(entity);
					}
				}

				mEnttMultiSelection.erase(
					std::remove_if(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), [&](entt::entity e) {
						return !mEnttScene.isValid(e);
						}),
					mEnttMultiSelection.end());

				mEnttSelectedEntity = mEnttMultiSelection.empty() ? entt::null : mEnttMultiSelection.back();
				sNameEditEntity = entt::null;
			}
		}
		ImGui::End();
	}
}
