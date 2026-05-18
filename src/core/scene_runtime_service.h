#pragma once

#include <functional>
#include <string>

#include "core/scene_runtime_state.h"

class SceneRuntimeService
{
public:
    using SerializeFn = std::function<std::string()>;
    using DeserializeFn = std::function<bool(const std::string&)>;

    SceneRuntimeState& state();
    const SceneRuntimeState& state() const;

    void clearHistory();
    void pushUndoSnapshot(const SerializeFn& serialize);
    void undo(const SerializeFn& serialize, const DeserializeFn& deserialize);
    void redo(const SerializeFn& serialize, const DeserializeFn& deserialize);

    bool saveScene(const SerializeFn& serialize);
    bool loadScene(const DeserializeFn& deserialize);
    bool saveEditorScene(const SerializeFn& serialize);
    bool loadEditorScene(const DeserializeFn& deserialize);

private:
    bool saveToFile(const std::string& filePath, const SerializeFn& serialize);
    bool loadFromFile(const std::string& filePath, const DeserializeFn& deserialize);

    SceneRuntimeState mState{};
};
