#include "renderer/renderer.h"

void Renderer::renderPlayModePanel(bool isEditMode)
{
    ImGui::Begin("Play Mode");
    ImGui::Checkbox("Viewport", &uiShowViewport);
    ImGui::SameLine();
    ImGui::Checkbox("Camera", &uiShowCameraControls);
    ImGui::SameLine();
    ImGui::Checkbox("Play HUD", &uiShowPlayHud);
    ImGui::SameLine();
    ImGui::Checkbox("Post UI", &uiShowPostProcessingWindow);
    ImGui::SameLine();
    ImGui::Checkbox("Shadow UI", &uiShowShadowTuningWindow);
    ImGui::SameLine();
    ImGui::Checkbox("Physics UI", &uiShowPhysicsWindow);
    ImGui::SameLine();
    ImGui::Checkbox("Prefab UI", &uiShowPrefabWindow);

    ImGui::Checkbox("Render Shadows", &renderEnableShadows);
    ImGui::SameLine();
    ImGui::Checkbox("Post Processing", &renderEnablePostProcessing);
    ImGui::SameLine();
    ImGui::BeginDisabled(!renderEnablePostProcessing);
    ImGui::Checkbox("FXAA", &renderEnableFxaa);
    ImGui::SameLine();
    ImGui::Checkbox("Bloom", &renderEnableBloom);
    ImGui::EndDisabled();

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f, 8.0f));

    const bool canPlay = isEditMode;
    const bool canStop = !isEditMode;

    if (canPlay)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.58f, 0.28f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.68f, 0.33f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.48f, 0.23f, 1.0f));
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
    }
    if (ImGui::Button(ICON_FA_PLAY "  Play") && canPlay)
        enterPlayMode();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    if (canStop)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.72f, 0.22f, 0.22f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.82f, 0.27f, 0.27f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.62f, 0.16f, 0.16f, 1.0f));
    }
    else
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
    }
    if (ImGui::Button(ICON_FA_STOP "  Stop") && canStop)
        exitPlayMode();
    ImGui::PopStyleColor(3);

    ImGui::SameLine();
    ImGui::TextUnformatted(sceneState == SceneState::PLAY ? "State: PLAY" : "State: EDIT");
    if (sceneState == SceneState::PLAY)
    {
        ImGui::SameLine();
        ImGui::Checkbox("Show Debug UI", &playShowDebugUI);
    }

    ImGui::PopStyleVar(2);
    ImGui::End();

    shadowSettings.enabled = renderEnableShadows ? 1.0f : 0.0f;
}

void Renderer::renderPostProcessingAndPhysicsPanels(bool isEditMode, entt::registry& registry)
{
    if (!((isEditMode || playShowDebugUI) && uiShowPostProcessingWindow))
        return;

    ImGui::Begin("Post Processing");
    ImGui::SliderFloat("FXAA Exposure", &fxaaExposure, 0.1f, 8.0f, "%.2f");
    ImGui::SliderFloat("FXAA Gamma", &fxaaGamma, 1.0f, 3.0f, "%.2f");
    ImGui::Checkbox("Bloom Enabled", &bloomEnabled);
    ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.05f, 4.0f, "%.2f");
    ImGui::SliderFloat("Bloom Soft Knee", &bloomSoftKnee, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Bloom Prefilter", &bloomPrefilterScale, 0.5f, 6.0f, "%.2f");
    ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 4.0f, "%.3f");
    ImGui::SliderFloat("Bloom Blur Scale", &bloomBlurScale, 0.25f, 3.0f, "%.2f");
    ImGui::SliderInt("Bloom Blur Passes", &bloomBlurPasses, 1, 8);
    const char* debugModes[] = { "Final", "Scene HDR", "Bloom Only" };
    ImGui::Combo("Post Debug", &postProcessDebugMode, debugModes, IM_ARRAYSIZE(debugModes));
    ImGui::End();

    if (uiShowPhysicsWindow)
        ImGui::Begin("Physics Demo");
    else
        ImGui::Begin("Physics Demo", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);

    if (uiShowPhysicsWindow)
    {
        ImGui::Checkbox("Pause Physics", &physicsPaused);
        physicsSystem.setPaused(physicsPaused);

        glm::vec3 gravity = physicsSystem.getGravity();
        if (ImGui::SliderFloat3("Gravity", &gravity.x, -30.0f, 30.0f, "%.2f"))
        {
            physicsSystem.setGravity(gravity);
        }

        ImGui::SliderInt("Spawn Count", &physicsSpawnCount, 1, 128);
        ImGui::SliderFloat("Spawn Height", &physicsSpawnHeight, 0.0f, -3120.0f, "%.1f");

        int rigidBodyCount = 0;
        for (auto entity : registry.view<RigidBodyComponent>()) {
            (void)entity;
            ++rigidBodyCount;
        }
        int colliderCount = 0;
        for (auto entity : registry.view<ColliderComponent>()) {
            (void)entity;
            ++colliderCount;
        }
        ImGui::Text("RigidBodies: %d", rigidBodyCount);
        ImGui::Text("Colliders: %d", colliderCount);

        if (ImGui::Button("Rebuild Physics Registration"))
        {
            physicsSystem.clear();
            for (auto [entity, rb, col, tr] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each())
            {
                (void)rb;
                (void)col;
                (void)tr;
                entt::entity e = entity;
                physicsSystem.registerEntity(e, registry);
            }
        }

        if (ImGui::Button("Reset Physics"))
        {
            physicsSystem.reset();
        }
    }
    ImGui::End();
}
