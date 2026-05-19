#include "core/scene_runtime_service.h"

#include <fstream>
#include <iterator>

SceneRuntimeState& SceneRuntimeService::state()
{
    return mState;
}

const SceneRuntimeState& SceneRuntimeService::state() const
{
    return mState;
}

void SceneRuntimeService::clearHistory()
{
    mState.clearHistory();
}

void SceneRuntimeService::pushUndoSnapshot(const SerializeFn& serialize)
{
    if (!serialize)
        return;
    mState.pushUndoSnapshot(serialize());
}

void SceneRuntimeService::undo(const SerializeFn& serialize, const DeserializeFn& deserialize)
{
    if (!serialize || !deserialize)
        return;

    auto targetSnapshot = mState.popUndoTarget(serialize());
    if (!targetSnapshot.has_value())
        return;

    const bool prevMute = mState.isHistoryMuted();
    mState.setHistoryMuted(true);
    deserialize(*targetSnapshot);
    mState.setHistoryMuted(prevMute);
}

void SceneRuntimeService::redo(const SerializeFn& serialize, const DeserializeFn& deserialize)
{
    if (!serialize || !deserialize)
        return;

    auto targetSnapshot = mState.popRedoTarget(serialize());
    if (!targetSnapshot.has_value())
        return;

    const bool prevMute = mState.isHistoryMuted();
    mState.setHistoryMuted(true);
    deserialize(*targetSnapshot);
    mState.setHistoryMuted(prevMute);
}

bool SceneRuntimeService::saveScene(const SerializeFn& serialize)
{
    return saveToFile(mState.sceneFilePath(), serialize);
}

bool SceneRuntimeService::loadScene(const DeserializeFn& deserialize)
{
    return loadFromFile(mState.sceneFilePath(), deserialize);
}

bool SceneRuntimeService::saveEditorScene(const SerializeFn& serialize)
{
    return saveToFile(mState.editorSceneFilePath(), serialize);
}

bool SceneRuntimeService::loadEditorScene(const DeserializeFn& deserialize)
{
    return loadFromFile(mState.editorSceneFilePath(), deserialize);
}

bool SceneRuntimeService::saveRenderPresets(const SerializeFn& serialize)
{
    return saveToFile(mState.renderPresetsFilePath(), serialize);
}

bool SceneRuntimeService::loadRenderPresets(const DeserializeFn& deserialize)
{
    return loadFromFile(mState.renderPresetsFilePath(), deserialize);
}

bool SceneRuntimeService::saveToFile(const std::string& filePath, const SerializeFn& serialize)
{
    if (!serialize)
        return false;

    std::ofstream outFile(filePath, std::ios::out | std::ios::trunc);
    if (!outFile.is_open())
        return false;

    outFile << serialize();
    return true;
}

bool SceneRuntimeService::loadFromFile(const std::string& filePath, const DeserializeFn& deserialize)
{
    if (!deserialize)
        return false;

    std::ifstream inFile(filePath);
    if (!inFile.is_open())
        return false;

    std::string jsonContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    return deserialize(jsonContent);
}
