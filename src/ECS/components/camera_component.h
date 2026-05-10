#pragma once
#pragma once 

#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <GLFW/glfw3.h>

enum class CameraMovement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN
};

class Camera {
private:
	glm::vec3 position;
	glm::vec3 front;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 worldUp;

	float yaw;
	float pitch;

	float movementSpeed;
	float mouseSensitivity;
	float zoom;

	void updateCameraVectors();

public:

	Camera(
		glm::vec3 position = glm::vec3(0.0f, 0.0f, 2.0f),
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
		float yaw = -90.0f,                                 // Look along negative Z-axis (OpenGL convention)
		float pitch = 0.0f
	)
		: position(position), worldUp(up), yaw(yaw), pitch(pitch),
		movementSpeed(14.5f), mouseSensitivity(0.1f), zoom(45.0f)
	{
	};

	glm::mat4 getViewMatrix() const;
	glm::mat4 getProjectionMatrix(float aspectRatio, float nearPlane = 0.1f, float farPlane = 3000.0f) const;

	void processKeyboard(CameraMovement direction, float deltaTime);
	void processMouseMovement(float xOffset, float yOffset, bool constrainPitch = true);
	void processMouseScroll(float yOffset);

	glm::vec3 getPosition() const { return position; }
	glm::vec3 getFront() const { return front; }
	float getZoom() const { return zoom; }
	float getMovementSpeed() const { return movementSpeed; }
	float getMouseSensitivity() const { return mouseSensitivity; }

	void setPosition(glm::vec3 newPos) { position = newPos; }
	void setYaw(float newYaw) { yaw = newYaw; }
	void setPitch(float newPitch) { pitch = newPitch; }
	void setMovementSpeed(float speed) { movementSpeed = speed; }
	void setMouseSensitivity(float sensitivity) { mouseSensitivity = sensitivity; }
	void setZoom(float newZoom) { zoom = newZoom; }

	void processInput(GLFWwindow* window, Camera& camera, float deltaTime) {
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::FORWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::BACKWARD, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::LEFT, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::RIGHT, deltaTime);

		// Vertical movement controls for 3D navigation
		// Space and Control provide intuitive up/down movement
		if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::UP, deltaTime);
		if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
				camera.processKeyboard(CameraMovement::DOWN, deltaTime);
		}

	static void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
		// Static variables maintain state between callback invocations
		static bool firstMouse = true;
		static float lastX = 0.0f, lastY = 0.0f;  // Previous mouse position for delta calculation

		if (firstMouse) {
			lastX = static_cast<float>(xpos);               // Initialize previous position
			lastY = static_cast<float>(ypos);
			firstMouse = false;         // Disable special handling for subsequent calls
			return;
		}

		// Calculate mouse movement deltas since last callback
		// These deltas represent the amount and direction of mouse movement
		float xoffset = static_cast<float>(xpos - lastX);                   // Horizontal movement (left-right)
		float yoffset = static_cast<float>(lastY - ypos);                   // Vertical movement (inverted: screen Y increases downward, camera pitch increases upward)

		// Update state for next callback iteration
		lastX = static_cast<float>(xpos);
		lastY = static_cast<float>(ypos);

		// Retrieve the camera pointer from the window user pointer
		Camera* camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
		if (camera) {
			// Convert mouse movement to camera rotation
			camera->processMouseMovement(xoffset, yoffset);
		}
	}

	// Scroll wheel callback for zoom control
	// Provides intuitive field-of-view adjustment through scroll wheel interaction
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
		// Retrieve the camera pointer from the window user pointer
		Camera* camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
		if (camera) {
			// Direct scroll-to-zoom mapping
			// Positive yoffset (scroll up) zooms in, negative (scroll down) zooms out
			camera->processMouseScroll(static_cast<float>(yoffset));
		}
	}

	// Input system initialization — no longer registers GLFW callbacks directly;
	// the Renderer owns the window user pointer and routes input here.
	void setupInputCallbacks(GLFWwindow* window) {
		// Configure mouse capture mode for first-person camera behavior
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
};

