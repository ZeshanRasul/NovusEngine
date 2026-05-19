#include "renderer/renderer.h"

#include <algorithm>
#include <filesystem>

#include "../../lib/ImGuiFileDialog.h"

void Renderer::renderShadowTuningPanel(bool isEditMode)
{
    if (!isEditMode || !uiShowShadowTuningWindow)
        return;

    ImGui::Begin("Shadow Tuning");
    ImGui::SliderFloat("Shadow Distance", &shadowSettings.shadowMaxDistance, 50.0f, 600.0f);
    ImGui::SliderFloat("Lambda", &shadowSettings.lambda, 0.0f, 1.0f);
    ImGui::SliderFloat("Bias Scale", &shadowSettings.biasScale, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Bias Min", &shadowSettings.biasMin, 0.00001f, 0.005f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Cascade Blend", &shadowSettings.cascadeBlendFactor, 0.0f, 0.5f);
    ImGui::SliderFloat("Coverage Padding", &shadowSettings.coveragePaddingFactor, 0.0f, 0.5f);
    ImGui::SliderFloat("Depth Padding", &shadowSettings.depthPaddingFactor, 0.0f, 1.0f);
    ImGui::SliderFloat("Caster Padding", &shadowSettings.casterPadding, 0.0f, 250.0f);
    ImGui::SliderFloat("Far Cascade Expansion", &shadowSettings.farCascadeExpansion, 1.0f, 4.0f);
    ImGui::SliderFloat("Base Padding", &shadowSettings.shadowPadding, 0.0f, 100.0f);
    ImGui::SliderFloat3("Light Direction", &shadowSettings.lightDirection.x, -1.0f, 1.0f);
    ImGui::ColorEdit4("Light Color", &shadowSettings.lightColor.x);
    ImGui::Checkbox("Cascade Debug View", reinterpret_cast<bool*>(&shadowSettings.cascadeDebugView));
    
    if (ImGui::Button("Reset Shadows")) {
        shadowSettings = ShadowSettings{};
    }
    ImGui::End();
}

void Renderer::renderAnimationControlsPanel(bool isEditMode)
{
    if (!isEditMode)
        return;

    ImGui::Begin("Animation Controls");

    if (ImGui::CollapsingHeader("Models")) {
        bool modelListEmtpy = mModelInstData.miModelList.empty();
        std::string selectedModelName;

        if (!modelListEmtpy) {
            selectedModelName = mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str();
        }

        if (modelListEmtpy) {
            ImGui::BeginDisabled();
        }

        ImGui::AlignTextToFramePadding();
        ImGui::Text("Models :");
        ImGui::SameLine();
        ImGui::PushItemWidth(200);
        if (ImGui::BeginCombo("##ModelCombo", selectedModelName.c_str())) {
            for (int i = 0; i < mModelInstData.miModelList.size(); ++i) {
                const bool isSelected = (mModelInstData.miSelectedModel == i);
                if (ImGui::Selectable(mModelInstData.miModelList.at(i)->getModelFileName().c_str(), isSelected)) {
                    mModelInstData.miSelectedModel = i;
                    selectedModelName = mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str();
                }

                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        if (modelListEmtpy) {
            ImGui::EndDisabled();
        }

        if (ImGui::Button("Import Model")) {
            IGFD::FileDialogConfig config;
            config.path = ".";
            config.countSelectionMax = 1;
            config.flags = ImGuiFileDialogFlags_Modal;
            ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
            ImGuiFileDialog::Instance()->OpenDialog("ChooseModelFile", "Choose Model File",
                "Supported Model Files{.gltf,.glb,.obj,.fbx,.dae,.mdl,.md3,.pk3}", config);
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseModelFile")) {
            if (ImGuiFileDialog::Instance()->IsOk()) {
                std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
                std::filesystem::path currentPath = std::filesystem::current_path();
                std::string relativePath = std::filesystem::relative(filePathName, currentPath).generic_string();
                if (!relativePath.empty()) {
                    filePathName = relativePath;
                }
                std::replace(filePathName.begin(), filePathName.end(), '\\', '/');

                if (!mModelInstData.miModelAddCallbackFunction(filePathName)) {
                    Logger::log(1, "%s error: unable to load model file '%s', unknown error \n", __FUNCTION__, filePathName.c_str());
                }
                else {
                    mModelInstData.miSelectedModel = mModelInstData.miModelList.size() - 1;
                    mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
                }
            }
            ImGuiFileDialog::Instance()->Close();
        }

        if (modelListEmtpy) {
            ImGui::BeginDisabled();
        }

        ImGui::SameLine();
        if (ImGui::Button("Delete Model")) {
            ImGui::OpenPopup("Delete Model?");
        }

        if (ImGui::BeginPopupModal("Delete Model?", nullptr, ImGuiChildFlags_AlwaysAutoResize)) {
            ImGui::Text("Delete Model '%s'?", mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str());
            ImGui::Indent();
            ImGui::Indent();
            if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
                mModelInstData.miModelDeleteCallbackFunction(mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str());
                if (mModelInstData.miSelectedModel > 0) {
                    mModelInstData.miSelectedModel -= 1;
                }
                if (!mModelInstData.miAssimpInstances.empty()) {
                    mModelInstData.miSelectedInstance = 0;
                }
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::SameLine();
        if (ImGui::Button("Create Instance")) {
            std::shared_ptr<AssimpModel> currentModel = mModelInstData.miModelList[mModelInstData.miSelectedModel];
            mModelInstData.miInstanceAddCallbackFunction(currentModel);
            mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
        }

        if (ImGui::Button("Create Multiple Instances")) {
            std::shared_ptr<AssimpModel> currentModel = mModelInstData.miModelList[mModelInstData.miSelectedModel];
            mModelInstData.miInstanceAddManyCallbackFunction(currentModel, mManyInstanceCreateNum);
            mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
        }
        ImGui::SameLine();
        ImGui::SliderInt("##MassInstanceCreation", &mManyInstanceCreateNum, 1, 100, "%d");

        if (modelListEmtpy) {
            ImGui::EndDisabled();
        }
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Instance and animation editing moved to ECS Inspector.");
    ImGui::End();
}

void Renderer::renderPlayHudPanel(bool isEditMode, entt::registry& registry)
{
    if (isEditMode)
        return;

    if (uiShowPlayHud)
        ImGui::Begin("Play HUD");
    else
        ImGui::Begin("Play HUD", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);

    if (uiShowPlayHud)
    {
        ImGui::TextUnformatted("Runtime UI");
        ImGui::Separator();
        ImGui::TextUnformatted("W/A/S/D + Mouse: Move camera");
        ImGui::TextUnformatted("Space/Ctrl: Up/Down");
        ImGui::TextUnformatted("Esc: Toggle mouse capture");

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
    }
    ImGui::End();
}
