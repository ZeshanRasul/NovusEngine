#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

constexpr uint32_t WIDTH = 1920;
constexpr uint32_t HEIGHT = 1080;

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	void initVulkan() {
		createInstance();
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() {
		glfwDestroyWindow(window);

		glfwTerminate();
	}

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Novus Engine", nullptr, nullptr);
	}

	void createInstance() {
		constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Novus Engine",
											  .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
											  .pEngineName = "No Engine",
											  .engineVersion = VK_MAKE_VERSION(1, 0, 0),
											  .apiVersion = vk::ApiVersion14 };

		uint32_t glfwExtensionCount = 0;
		auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		auto extensionProperties = context.enumerateInstanceExtensionProperties();
		for (uint32_t i = 0; i < glfwExtensionCount; ++i)
		{
			if (std::ranges::none_of(extensionProperties,
				[glfwExtension = glfwExtensions[i]](auto const& extensionProperty)
				{ return strcmp(extensionProperty.extensionName, glfwExtension) == 0; }))
			{
				throw std::runtime_error("Required GLFW extension not supported: " + std::string(glfwExtensions[i]));
			}
		}

		auto extensions = context.enumerateInstanceExtensionProperties();

		for (const auto& extension : extensions) {
			std::cout << '\t' << extension.extensionName << '\n';
		}

		vk::InstanceCreateInfo createInfo{
			.pApplicationInfo = &appInfo,
			.enabledExtensionCount = glfwExtensionCount,
			.ppEnabledExtensionNames = glfwExtensions };

		instance = vk::raii::Instance(context, createInfo);
	}

private:
	GLFWwindow* window;

	vk::raii::Context context;
	vk::raii::Instance instance = nullptr;
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}