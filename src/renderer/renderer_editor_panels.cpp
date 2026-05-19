#include "renderer/renderer.h"

#include <algorithm>
#include <cctype>

void Renderer::renderCameraControlsPanel()
{
    if (!uiShowCameraControls)
        return;

    ImGui::SetNextWindowBgAlpha(1.0f);
    ImGui::Begin("Camera Controls");

    if (ImGui::Button("Reset Camera")) {
        camera.setPosition(glm::vec3(400.0f, -120.0f, 0.0f));
        camera.setYaw(180.0f);
        camera.setPitch(-5.0f);
        camera.setMovementSpeed(140.0f);
        camera.setZoom(55.0f);
        camera.getViewMatrix();
        camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f);
    }

    float movementSpeed = camera.getMovementSpeed();
    if (ImGui::SliderFloat("Movement Speed", &movementSpeed, 1.0f, 100.0f)) {
        camera.setMovementSpeed(movementSpeed);
    }

    float sensitivity = camera.getMouseSensitivity();
    if (ImGui::SliderFloat("Mouse Sensitivity", &sensitivity, 0.1f, 1.0f)) {
        camera.setMouseSensitivity(sensitivity);
    }

    float zoom = camera.getZoom();
    if (ImGui::SliderFloat("Zoom", &zoom, 1.0f, 90.0f)) {
        camera.setZoom(zoom);
    }

    const glm::vec3 camPos = camera.getPosition();
    ImGui::Text("Position: (%.2f, %.2f, %.2f)", camPos.x, camPos.y, camPos.z);

    ImGuiIO& io = ImGui::GetIO();
    const float fps = io.Framerate;
    const float frameMs = fps > 0.0f ? (1000.0f / fps) : 0.0f;
    ImGui::Text("FPS: %.1f (%.2f ms)", fps, frameMs);

    ImGui::End();
}

void Renderer::renderPrefabPanel(bool isEditMode, entt::registry& registry)
{
    if (!((isEditMode || playShowDebugUI) && uiShowPrefabWindow))
        return;

    auto makePrefabSafeName = [](std::string name) {
        for (char& c : name)
        {
            const unsigned char uc = static_cast<unsigned char>(c);
            if (!(std::isalnum(uc) || c == '_' || c == '-'))
                c = '_';
        }
        if (name.empty())
            name = "default";
        return name;
        };

    std::string suggestedPrefabName = "default";
    if (mEnttScene.isValid(mEnttSelectedEntity))
    {
        if (const auto* selectedTag = registry.try_get<EnttTagComponent>(mEnttSelectedEntity); selectedTag && !selectedTag->name.empty())
            suggestedPrefabName = selectedTag->name;
    }

    mSceneRuntimeService.state().prefabSaveFilePath() = "prefabs/" + makePrefabSafeName(suggestedPrefabName) + ".prefab.json";

    if (mSceneRuntimeService.state().prefabAssetsDirty())
        refreshPrefabAssetList();

    ImGui::Begin("Prefabs");
    if (ImGui::Button("Refresh"))
    {
        mSceneRuntimeService.state().markPrefabAssetsDirty();
        refreshPrefabAssetList();
    }

    ImGui::SameLine();
    if (ImGui::Button("Save Selected As Prefab"))
    {
        if (saveSelectedAsPrefab(mSceneRuntimeService.state().prefabSaveFilePath()))
        {
            mSceneRuntimeService.state().markPrefabAssetsDirty();
            mSceneRuntimeService.state().prefabFilePath() = mSceneRuntimeService.state().prefabSaveFilePath();
            refreshPrefabAssetList();
        }
    }

    ImGui::Text("Save Path: %s", mSceneRuntimeService.state().prefabSaveFilePath().c_str());

    ImGui::Separator();
    ImGui::Text("Available Prefabs");
    for (int i = 0; i < static_cast<int>(mSceneRuntimeService.state().prefabAssets().size()); ++i)
    {
        const bool selected = (mSceneRuntimeService.state().selectedPrefabAsset() == i);
        if (ImGui::Selectable(mSceneRuntimeService.state().prefabAssets()[i].c_str(), selected))
        {
            mSceneRuntimeService.state().setSelectedPrefabAsset(i);
            mSceneRuntimeService.state().prefabFilePath() = mSceneRuntimeService.state().prefabAssets()[i];
        }
    }

    ImGui::Separator();
    ImGui::Text("Instantiate Path: %s", mSceneRuntimeService.state().prefabFilePath().empty() ? "<none>" : mSceneRuntimeService.state().prefabFilePath().c_str());

    ImGui::BeginDisabled(mSceneRuntimeService.state().selectedPrefabAsset() < 0 || mSceneRuntimeService.state().selectedPrefabAsset() >= static_cast<int>(mSceneRuntimeService.state().prefabAssets().size()));
    if (ImGui::Button("Instantiate Selected Prefab") &&
        mSceneRuntimeService.state().selectedPrefabAsset() >= 0 &&
        mSceneRuntimeService.state().selectedPrefabAsset() < static_cast<int>(mSceneRuntimeService.state().prefabAssets().size()))
    {
        instantiatePrefab(mSceneRuntimeService.state().prefabAssets()[mSceneRuntimeService.state().selectedPrefabAsset()]);
    }
    ImGui::EndDisabled();
    ImGui::End();
}
