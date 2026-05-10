#pragma once

#include <functional>
#include <unordered_map>
#include <vector>
#include <glm/glm.hpp>
#include "../ECS/components/camera_component.h"

// Input actions that our application can respond to
enum class InputAction {
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    MOVE_UP,
    MOVE_DOWN,
    LOOK_UP,
    LOOK_DOWN,
    LOOK_LEFT,
    LOOK_RIGHT,
    ZOOM_IN,
    ZOOM_OUT,
    TOGGLE_UI_MODE,
    // Add more actions as needed
};

// Input state that tracks the current state of inputs
struct InputState {
    glm::vec2 cursorPosition = { 0.0f, 0.0f };
    glm::vec2 cursorDelta = { 0.0f, 0.0f };
    bool mouseButtons[3] = { false, false, false };
    float scrollDelta = 0.0f;

    // For touch input
    struct TouchPoint {
        int id;
        glm::vec2 position;
        glm::vec2 delta;
    };
    std::vector<TouchPoint> touchPoints;

    // Reset delta values after each frame
    void resetDeltas() {
        cursorDelta = { 0.0f, 0.0f };
        scrollDelta = 0.0f;
        for (auto& touch : touchPoints) {
            touch.delta = { 0.0f, 0.0f };
        }
    }
};

class InputSystem {
public:
    static void Initialize(GLFWwindow* window, Camera* camera);
    static void Shutdown();

    // Update input state (called once per frame)
    static void Update(float deltaTime);

    // Register a callback for an input action
    static void RegisterActionCallback(InputAction action, std::function<void(float)> callback);

    // Process a platform-specific input event
    static bool ProcessInputEvent(void* event);

    // Get the current input state
    static InputState& GetInputState() { return inputState; };

    // Check if ImGui is capturing input
    static bool IsImGuiCapturingKeyboard();
    static bool IsImGuiCapturingMouse();

	static Camera* gCamera; // Global pointer to the camera for input callbacks
    static float dt;
private:
    static InputState inputState;
    static std::unordered_map<InputAction, std::function<void(float)>> actionCallbacks;
};