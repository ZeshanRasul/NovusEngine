#pragma once

#include <memory>

#include <GLFW/glfw3.h>

#include "gameplay_layer.h"

namespace Gameplay
{
    class GameplayRuntime
    {
    public:
        void setUseDefaultLayer(bool enabled);
        bool useDefaultLayer() const;
        bool hasActiveLayer() const;

        void enterPlay(RuntimeContext& context);
        void exitPlay(RuntimeContext& context);
        void fixedUpdate(float fixedDeltaTime, GLFWwindow* window, RuntimeContext& context);

    private:
        GameplayInputState sampleInput(GLFWwindow* window) const;

        std::unique_ptr<IGameLayer> mGameLayer = nullptr;
        bool mUseDefaultGameLayer = true;
    };
}
