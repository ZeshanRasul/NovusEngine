#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

class SceneRuntimeState
{
public:
    void clearHistory();
    bool isHistoryMuted() const;
    void setHistoryMuted(bool muted);
    void pushUndoSnapshot(const std::string& snapshot, size_t maxSnapshots = 100);
    std::optional<std::string> popUndoTarget(const std::string& currentSnapshot);
    std::optional<std::string> popRedoTarget(const std::string& currentSnapshot);

    const std::string& sceneFilePath() const;
    const std::string& editorSceneFilePath() const;
	const std::string& prefabFilePath() const;
	const std::string& renderPresetsFilePath() const;

    std::string& prefabFilePath();
    std::string& prefabSaveFilePath();

    const std::vector<std::string>& prefabAssets() const;
    std::vector<std::string>& prefabAssets();

    int selectedPrefabAsset() const;
    void setSelectedPrefabAsset(int selectedPrefabAsset);

    bool prefabAssetsDirty() const;
    void setPrefabAssetsDirty(bool dirty);
    void markPrefabAssetsDirty();

    void refreshPrefabAssetList(const std::filesystem::path& prefabRoot = "prefabs");

private:
    std::vector<std::string> mUndoSnapshots{};
    std::vector<std::string> mRedoSnapshots{};
    bool mHistoryMuted = false;

    std::string mSceneFilePath = "scene.json";
    std::string mEditorSceneFilePath = "editor_scene.json";
    std::string mPrefabFilePath = "prefabs/default.prefab.json";
    std::string mPrefabSaveFilePath = "prefabs/default.prefab.json";
	std::string mRenderPresetsFilePath = "render_presets.json";
    std::vector<std::string> mPrefabAssets{};
    int mSelectedPrefabAsset = -1;
    bool mPrefabAssetsDirty = true;
};
