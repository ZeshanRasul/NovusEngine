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

static ImGuiKey glfwKeyToImGui(int key)
{
	switch (key)
	{
	case GLFW_KEY_TAB:           return ImGuiKey_Tab;
	case GLFW_KEY_LEFT:          return ImGuiKey_LeftArrow;
	case GLFW_KEY_RIGHT:         return ImGuiKey_RightArrow;
	case GLFW_KEY_UP:            return ImGuiKey_UpArrow;
	case GLFW_KEY_DOWN:          return ImGuiKey_DownArrow;
	case GLFW_KEY_PAGE_UP:       return ImGuiKey_PageUp;
	case GLFW_KEY_PAGE_DOWN:     return ImGuiKey_PageDown;
	case GLFW_KEY_HOME:          return ImGuiKey_Home;
	case GLFW_KEY_END:           return ImGuiKey_End;
	case GLFW_KEY_INSERT:        return ImGuiKey_Insert;
	case GLFW_KEY_DELETE:        return ImGuiKey_Delete;
	case GLFW_KEY_BACKSPACE:     return ImGuiKey_Backspace;
	case GLFW_KEY_SPACE:         return ImGuiKey_Space;
	case GLFW_KEY_ENTER:         return ImGuiKey_Enter;
	case GLFW_KEY_ESCAPE:        return ImGuiKey_Escape;
	case GLFW_KEY_APOSTROPHE:    return ImGuiKey_Apostrophe;
	case GLFW_KEY_COMMA:         return ImGuiKey_Comma;
	case GLFW_KEY_MINUS:         return ImGuiKey_Minus;
	case GLFW_KEY_PERIOD:        return ImGuiKey_Period;
	case GLFW_KEY_SLASH:         return ImGuiKey_Slash;
	case GLFW_KEY_SEMICOLON:     return ImGuiKey_Semicolon;
	case GLFW_KEY_EQUAL:         return ImGuiKey_Equal;
	case GLFW_KEY_LEFT_BRACKET:  return ImGuiKey_LeftBracket;
	case GLFW_KEY_BACKSLASH:     return ImGuiKey_Backslash;
	case GLFW_KEY_RIGHT_BRACKET: return ImGuiKey_RightBracket;
	case GLFW_KEY_GRAVE_ACCENT:  return ImGuiKey_GraveAccent;
	case GLFW_KEY_CAPS_LOCK:     return ImGuiKey_CapsLock;
	case GLFW_KEY_LEFT_SHIFT:    return ImGuiKey_LeftShift;
	case GLFW_KEY_LEFT_CONTROL:  return ImGuiKey_LeftCtrl;
	case GLFW_KEY_LEFT_ALT:      return ImGuiKey_LeftAlt;
	case GLFW_KEY_LEFT_SUPER:    return ImGuiKey_LeftSuper;
	case GLFW_KEY_RIGHT_SHIFT:   return ImGuiKey_RightShift;
	case GLFW_KEY_RIGHT_CONTROL: return ImGuiKey_RightCtrl;
	case GLFW_KEY_RIGHT_ALT:     return ImGuiKey_RightAlt;
	case GLFW_KEY_RIGHT_SUPER:   return ImGuiKey_RightSuper;
	case GLFW_KEY_KP_ENTER:      return ImGuiKey_KeypadEnter;
	case GLFW_KEY_F1:  return ImGuiKey_F1;  case GLFW_KEY_F2:  return ImGuiKey_F2;
	case GLFW_KEY_F3:  return ImGuiKey_F3;  case GLFW_KEY_F4:  return ImGuiKey_F4;
	case GLFW_KEY_F5:  return ImGuiKey_F5;  case GLFW_KEY_F6:  return ImGuiKey_F6;
	case GLFW_KEY_F7:  return ImGuiKey_F7;  case GLFW_KEY_F8:  return ImGuiKey_F8;
	case GLFW_KEY_F9:  return ImGuiKey_F9;  case GLFW_KEY_F10: return ImGuiKey_F10;
	case GLFW_KEY_F11: return ImGuiKey_F11; case GLFW_KEY_F12: return ImGuiKey_F12;
	case GLFW_KEY_0: return ImGuiKey_0; case GLFW_KEY_1: return ImGuiKey_1;
	case GLFW_KEY_2: return ImGuiKey_2; case GLFW_KEY_3: return ImGuiKey_3;
	case GLFW_KEY_4: return ImGuiKey_4; case GLFW_KEY_5: return ImGuiKey_5;
	case GLFW_KEY_6: return ImGuiKey_6; case GLFW_KEY_7: return ImGuiKey_7;
	case GLFW_KEY_8: return ImGuiKey_8; case GLFW_KEY_9: return ImGuiKey_9;
	case GLFW_KEY_A: return ImGuiKey_A; case GLFW_KEY_B: return ImGuiKey_B;
	case GLFW_KEY_C: return ImGuiKey_C; case GLFW_KEY_D: return ImGuiKey_D;
	case GLFW_KEY_E: return ImGuiKey_E; case GLFW_KEY_F: return ImGuiKey_F;
	case GLFW_KEY_G: return ImGuiKey_G; case GLFW_KEY_H: return ImGuiKey_H;
	case GLFW_KEY_I: return ImGuiKey_I; case GLFW_KEY_J: return ImGuiKey_J;
	case GLFW_KEY_K: return ImGuiKey_K; case GLFW_KEY_L: return ImGuiKey_L;
	case GLFW_KEY_M: return ImGuiKey_M; case GLFW_KEY_N: return ImGuiKey_N;
	case GLFW_KEY_O: return ImGuiKey_O; case GLFW_KEY_P: return ImGuiKey_P;
	case GLFW_KEY_Q: return ImGuiKey_Q; case GLFW_KEY_R: return ImGuiKey_R;
	case GLFW_KEY_S: return ImGuiKey_S; case GLFW_KEY_T: return ImGuiKey_T;
	case GLFW_KEY_U: return ImGuiKey_U; case GLFW_KEY_V: return ImGuiKey_V;
	case GLFW_KEY_W: return ImGuiKey_W; case GLFW_KEY_X: return ImGuiKey_X;
	case GLFW_KEY_Y: return ImGuiKey_Y; case GLFW_KEY_Z: return ImGuiKey_Z;
	default: return ImGuiKey_None;
	}
}

static void glfwCharCallback(GLFWwindow*, unsigned int codepoint)
{
	ImGui::GetIO().AddInputCharacter(codepoint);
}

static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// GLFW_REPEAT is ignored — ImGui handles key-repeat timing internally
	if (action != GLFW_PRESS && action != GLFW_RELEASE)
		return;

	// Forward all key events to ImGui
	ImGuiIO& io = ImGui::GetIO();
	const bool pressed = (action == GLFW_PRESS);
	ImGuiKey imguiKey = glfwKeyToImGui(key);
	if (imguiKey != ImGuiKey_None)
		io.AddKeyEvent(imguiKey, pressed);

	// Modifier state
	io.AddKeyEvent(ImGuiMod_Ctrl,  (mods & GLFW_MOD_CONTROL) != 0);
	io.AddKeyEvent(ImGuiMod_Shift, (mods & GLFW_MOD_SHIFT)   != 0);
	io.AddKeyEvent(ImGuiMod_Alt,   (mods & GLFW_MOD_ALT)     != 0);
	io.AddKeyEvent(ImGuiMod_Super, (mods & GLFW_MOD_SUPER)   != 0);

	// Escape toggles mouse capture (only when ImGui doesn't want it)
	if (key == GLFW_KEY_ESCAPE && pressed && !io.WantTextInput)
	{
		mouseCaptureMode = !mouseCaptureMode;
		glfwSetInputMode(window, GLFW_CURSOR,
			mouseCaptureMode ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
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
	glfwSetCharCallback(window, glfwCharCallback);

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

