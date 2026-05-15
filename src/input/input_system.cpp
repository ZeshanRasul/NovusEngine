#include <GLFW/glfw3.h>
#include "../../include/imgui.h"
#include "input_system.h"

// Define static member variables
InputState InputSystem::inputState;
std::unordered_map<InputAction, std::function<void(float)>> InputSystem::actionCallbacks
{
	{ InputAction::MOVE_FORWARD , [](float dt) { gCamera->processKeyboard(CameraMovement::FORWARD, dt); } },
	{InputAction::MOVE_BACKWARD, [](float dt) { gCamera->processKeyboard(CameraMovement::BACKWARD, dt); } },
	{ InputAction::MOVE_LEFT    , [](float dt) { gCamera->processKeyboard(CameraMovement::LEFT, dt); } },
	{ InputAction::MOVE_RIGHT   , [](float dt) { gCamera->processKeyboard(CameraMovement::RIGHT, dt); } },
	{ InputAction::MOVE_UP      , [](float dt) { gCamera->processKeyboard(CameraMovement::UP, dt); } },
	{ InputAction::MOVE_DOWN    , [](float dt) { gCamera->processKeyboard(CameraMovement::DOWN, dt); } },
};
Camera globalCamera; // Global camera instance for input callbacks
Camera* InputSystem::gCamera = &globalCamera; // Initialize camera pointer
float InputSystem::dt = 0.0f;

// Define input modes
enum class InputMode {
	CAMERA_CONTROL,
	UI_INTERACTION,
	OBJECT_MANIPULATION
};

// Store the GLFW window pointer
static GLFWwindow* gWindow = nullptr;
static bool mouseCaptureMode = false;

// Current input mode
static InputMode currentInputMode = InputMode::CAMERA_CONTROL;

// Set the input mode
void setInputMode(InputMode mode) {
	currentInputMode = mode;

	// Update platform-specific settings based on the mode
	// This example shows how to implement this with GLFW
	if (mode == InputMode::CAMERA_CONTROL) {
		// In GLFW, we can disable the cursor for camera control
		glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}
	else {
		// For UI interaction, we want the cursor to be visible
		glfwSetInputMode(gWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}

	// With other windowing libraries, you would use their equivalent APIs
}

// Toggle between camera control and UI interaction modes
void toggleInputMode() {
	if (currentInputMode == InputMode::CAMERA_CONTROL) {
		setInputMode(InputMode::UI_INTERACTION);
	}
	else {
		setInputMode(InputMode::CAMERA_CONTROL);
	}
}


// GLFW callback functions
static void glfwMouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	ImGuiIO& io = ImGui::GetIO();
	if (button >= 0 && button < ImGuiMouseButton_COUNT)
		io.AddMouseButtonEvent(button, action == GLFW_PRESS);

	if (button >= 0 && button < 3) {
		InputState& state = InputSystem::GetInputState();
		state.mouseButtons[button] = action == GLFW_PRESS;
	}
}

static void glfwCursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
	InputState& state = InputSystem::GetInputState();

	// Calculate delta from last position
	glm::vec2 newPos(static_cast<float>(xpos), static_cast<float>(ypos));
	state.cursorDelta = newPos - state.cursorPosition;
	state.cursorPosition = newPos;
	ImGuiIO& io = ImGui::GetIO();
	io.AddMousePosEvent(static_cast<float>(xpos), static_cast<float>(ypos));

	if (mouseCaptureMode && InputSystem::gCamera) {
		const float xoffset = state.cursorDelta.x;
		const float yoffset = -state.cursorDelta.y;
		InputSystem::gCamera->processMouseMovement(xoffset, yoffset);
	}
}

static void glfwScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
	InputState& state = InputSystem::GetInputState();
	state.scrollDelta = static_cast<float>(yoffset);
	ImGuiIO& io = ImGui::GetIO();
	io.AddMouseWheelEvent(static_cast<float>(xoffset), static_cast<float>(yoffset));

	if (mouseCaptureMode && InputSystem::gCamera) {
		InputSystem::gCamera->processMouseScroll(static_cast<float>(yoffset));
	}
}

static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Map GLFW keys to our input actions
	if (action == GLFW_PRESS || action == GLFW_RELEASE) {
		bool pressed = (action == GLFW_PRESS);

		// Toggle mouse capture mode with Escape key
		if (key == GLFW_KEY_ESCAPE && pressed) {
			mouseCaptureMode = !mouseCaptureMode;

			if (mouseCaptureMode) {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}
			else {
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}

	}
}

void InputSystem::Initialize(GLFWwindow* window, Camera* camera) {
	gWindow = window;
	gCamera = camera;

	// Set up GLFW callbacks
	glfwSetMouseButtonCallback(window, glfwMouseButtonCallback);
	glfwSetCursorPosCallback(window, glfwCursorPosCallback);
	glfwSetScrollCallback(window, glfwScrollCallback);
	glfwSetKeyCallback(window, glfwKeyCallback);

	// Initially capture the cursor for camera control
	mouseCaptureMode = true;
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	setInputMode(InputMode::CAMERA_CONTROL); // Start in camera control mode
}

void InputSystem::Update(float deltaTime) {
	// Poll for input events
	glfwPollEvents();

	dt = deltaTime;

	if (ImGui::GetIO().WantCaptureKeyboard) {
		inputState.resetDeltas();
		return;
	}

	// Update key states for continuous actions (like movement)
	if (glfwGetKey(gWindow, GLFW_KEY_W) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_FORWARD); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}

	if (glfwGetKey(gWindow, GLFW_KEY_S) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_BACKWARD); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}
	if (glfwGetKey(gWindow, GLFW_KEY_A) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_LEFT); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}
	if (glfwGetKey(gWindow, GLFW_KEY_D) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_RIGHT); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}
	if (glfwGetKey(gWindow, GLFW_KEY_SPACE) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_UP); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}
	if (glfwGetKey(gWindow, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
		if (auto it = actionCallbacks.find(InputAction::MOVE_DOWN); it != actionCallbacks.end()) {
			it->second(deltaTime);
		}
	}

	// Reset delta values after processing
	inputState.resetDeltas();
}

bool InputSystem::IsImGuiCapturingKeyboard() {
	return ImGui::GetIO().WantCaptureKeyboard;
}

bool InputSystem::IsImGuiCapturingMouse() {
	return ImGui::GetIO().WantCaptureMouse;
}

