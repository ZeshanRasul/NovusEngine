#include "core/scene_runtime_state.h"

#include <algorithm>

void SceneRuntimeState::clearHistory()
{
    mUndoSnapshots.clear();
    mRedoSnapshots.clear();
}

bool SceneRuntimeState::isHistoryMuted() const
{
    return mHistoryMuted;
}

void SceneRuntimeState::setHistoryMuted(bool muted)
{
    mHistoryMuted = muted;
}

void SceneRuntimeState::pushUndoSnapshot(const std::string& snapshot, size_t maxSnapshots)
{
    if (mHistoryMuted)
        return;

    mUndoSnapshots.push_back(snapshot);
    if (mUndoSnapshots.size() > maxSnapshots)
        mUndoSnapshots.erase(mUndoSnapshots.begin());

    mRedoSnapshots.clear();
}

std::optional<std::string> SceneRuntimeState::popUndoTarget(const std::string& currentSnapshot)
{
    if (mUndoSnapshots.empty())
        return std::nullopt;

    mRedoSnapshots.push_back(currentSnapshot);
    std::string targetSnapshot = mUndoSnapshots.back();
    mUndoSnapshots.pop_back();
    return targetSnapshot;
}

std::optional<std::string> SceneRuntimeState::popRedoTarget(const std::string& currentSnapshot)
{
    if (mRedoSnapshots.empty())
        return std::nullopt;

    mUndoSnapshots.push_back(currentSnapshot);
    std::string targetSnapshot = mRedoSnapshots.back();
    mRedoSnapshots.pop_back();
    return targetSnapshot;
}

const std::string& SceneRuntimeState::sceneFilePath() const
{
    return mSceneFilePath;
}

const std::string& SceneRuntimeState::editorSceneFilePath() const
{
    return mEditorSceneFilePath;
}

std::string& SceneRuntimeState::prefabFilePath()
{
    return mPrefabFilePath;
}

std::string& SceneRuntimeState::prefabSaveFilePath()
{
    return mPrefabSaveFilePath;
}

const std::vector<std::string>& SceneRuntimeState::prefabAssets() const
{
    return mPrefabAssets;
}

std::vector<std::string>& SceneRuntimeState::prefabAssets()
{
    return mPrefabAssets;
}

int SceneRuntimeState::selectedPrefabAsset() const
{
    return mSelectedPrefabAsset;
}

void SceneRuntimeState::setSelectedPrefabAsset(int selectedPrefabAsset)
{
    mSelectedPrefabAsset = selectedPrefabAsset;
}

bool SceneRuntimeState::prefabAssetsDirty() const
{
    return mPrefabAssetsDirty;
}

void SceneRuntimeState::setPrefabAssetsDirty(bool dirty)
{
    mPrefabAssetsDirty = dirty;
}

void SceneRuntimeState::markPrefabAssetsDirty()
{
    mPrefabAssetsDirty = true;
}

void SceneRuntimeState::refreshPrefabAssetList(const std::filesystem::path& prefabRoot)
{
    mPrefabAssets.clear();
    if (!std::filesystem::exists(prefabRoot) || !std::filesystem::is_directory(prefabRoot))
    {
        mSelectedPrefabAsset = -1;
        mPrefabAssetsDirty = false;
        return;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(prefabRoot))
    {
        if (!entry.is_regular_file())
            continue;

        const auto path = entry.path();
        const std::string ext = path.extension().string();
        if (ext != ".json")
            continue;

        const std::string pathStr = path.lexically_normal().generic_string();
        if (pathStr.find(".prefab") == std::string::npos)
            continue;

        mPrefabAssets.push_back(pathStr);
    }

    std::sort(mPrefabAssets.begin(), mPrefabAssets.end());
    if (mPrefabAssets.empty())
        mSelectedPrefabAsset = -1;
    else if (mSelectedPrefabAsset < 0 || mSelectedPrefabAsset >= static_cast<int>(mPrefabAssets.size()))
        mSelectedPrefabAsset = 0;

    mPrefabAssetsDirty = false;
}
