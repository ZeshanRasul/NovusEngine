#include "gameplay/gameplay_runtime.h"

namespace Gameplay
{
    void GameplayRuntime::setUseDefaultLayer(bool enabled)
    {
        mUseDefaultGameLayer = enabled;
    }

    bool GameplayRuntime::useDefaultLayer() const
    {
        return mUseDefaultGameLayer;
    }

    bool GameplayRuntime::hasActiveLayer() const
    {
        return mGameLayer != nullptr;
    }

    void GameplayRuntime::enterPlay(RuntimeContext& context)
    {
        if (mUseDefaultGameLayer)
            mGameLayer = std::make_unique<DefaultGameLayer>();

        if (mGameLayer)
            mGameLayer->onEnterPlay(context);
    }

    void GameplayRuntime::exitPlay(RuntimeContext& context)
    {
        if (mGameLayer)
            mGameLayer->onExitPlay(context);
    }

    void GameplayRuntime::fixedUpdate(float fixedDeltaTime, GLFWwindow* window, RuntimeContext& context)
    {
        if (!mGameLayer)
            return;

        GameplayInputState input = sampleInput(window);
        mGameLayer->onFixedUpdate(fixedDeltaTime, input, context);
    }

    GameplayInputState GameplayRuntime::sampleInput(GLFWwindow* window) const
    {
        GameplayInputState input{};
        if (!window)
            return input;

        input.moveForward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
        input.moveBackward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
        input.moveLeft = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
        input.moveRight = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
        input.jump = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        input.sprint = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
        return input;
    }
}
