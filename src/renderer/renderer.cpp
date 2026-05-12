#include "renderer/renderer.h"
#include <ktx.h>
#include <filesystem>
#include <glm/gtc/type_ptr.hpp>
#include "../vulkan/command_pool.h"
#include "../vulkan/command_buffer.h"
#include "../vulkan/buffer.h"
#include "../vulkan/texture.h"
#include "../vulkan/image.h"
#include "../vulkan/image_view.h"
#include "../vulkan/pipeline.h"
#include "../vulkan/swap_chain.h"
#include "../vulkan/depth_target.h"
#include "../vulkan/sync.h"
#include "../vulkan/material.h"
#include "../model/model.h"
#include "../vulkan/uniform_buffer.h"
#include "../vulkan/descriptors.h"
#include "../ECS/entity.h"
#include "../model/AssimpModel.h"
#include "../model/AssimpInstance.h"
#include "../model/InstanceSettings.h"
#include "../../lib/ImGuiFileDialog.h"

// ---------------------------------------------------------------------------
// Public
// ---------------------------------------------------------------------------

void Renderer::run()
{
   initWindow();
	try
	{
		initVulkan();
		mainLoop();
	}
	catch (...)
	{
		cleanup();
		throw;
	}
	cleanup();
}

// ---------------------------------------------------------------------------
// Initialisation / shutdown
// ---------------------------------------------------------------------------

void Renderer::initWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Novus Engine", nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

	camera.setupInputCallbacks(window);
	glfwSetWindowUserPointer(window, &camera); // Set the user pointer for the InputSystem callbacks
	InputSystem::Initialize(window, &camera);
    camera.setPosition(glm::vec3(0.0f, -90.0f, 18.0f));
	camera.setYaw(90.0f);
	camera.setPitch(-5.0f);
	camera.setMovementSpeed(40.0f);
	camera.setZoom(55.0f);
	camera.getViewMatrix();
	camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 600.0f);
}

void Renderer::initVulkan()
{
	deviceInit();
    SwapChain::createSwapChain(physicalDevice, device, surface, window, swapChain, swapChainImages, swapChainExtent, swapChainSurfaceFormat);
	SwapChain::createImageViews(device, swapChainImages, swapChainSurfaceFormat.format, swapChainImageViews);
	DescriptorSetLayout::createEntityDescriptorSetLayout(device, descriptorSetLayout, 7);
	DescriptorSetLayout::createEntityDescriptorSetLayout(device, shadowDescriptorSetLayout, 7);

	if (!createPBRPipeline())
	{
		std::cerr << "Failed to create PBR pipeline" << std::endl;
	}

	ShadowPass::createPipeline(device, physicalDevice, shadowPipelineLayout, shadowPipeline, shadowDescriptorSetLayout);

	CommandPool::init(device, queueIndex, commandPool);
	DepthTarget::createDepthResources(device, physicalDevice, swapChainExtent, depthImage, depthImageMemory, depthImageView);
	for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
		ShadowPass::createResources(device, physicalDevice, shadowImages[i], shadowImageMemories[i], shadowImageViews[i], shadowSampler);
	createTextureSampler();
	createDefaultTextures();
	setupGameObjects();
	for (auto& entityPtr : entities)
	{
		createVertexBuffer(*entityPtr->GetComponent<RenderableComponent>());
		createIndexBuffer(*entityPtr->GetComponent<RenderableComponent>());
	}
	UniformBuffer::createUniformBuffers(entities, device, physicalDevice, MAX_FRAMES_IN_FLIGHT);
	DescriptorPool::createDescriptorPool(device, entities, descriptorPool, MAX_FRAMES_IN_FLIGHT);
 std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowViews = {
		*shadowImageViews[0], *shadowImageViews[1], *shadowImageViews[2], *shadowImageViews[3], *shadowImageViews[4]
	};
	DescriptorSet::createDescriptorSets(device, entities, descriptorPool, descriptorSetLayout, defaultTextureView, defaultNormalView, textureSampler, shadowViews, shadowSampler, MAX_FRAMES_IN_FLIGHT);
	CommandBuffer::init(device, queueIndex, commandPool, commandBuffers, MAX_FRAMES_IN_FLIGHT);
    Sync::createSyncObjects(device, swapChainImages.size(), MAX_FRAMES_IN_FLIGHT, presentCompleteSemaphores, renderFinishedSemaphores, inFlightFences);

	imGui = new ImGuiVulkanUtil(
		device,
		physicalDevice,
		queue,
		queueIndex
	);

	imGui->init(swapChainExtent.width, swapChainExtent.height);
	imGui->initResources(); // No renderPass needed with dynamic rendering

	mModelInstData.miModelCheckCallbackFunction = [this](std::string fileName) { return hasModel(fileName); };
	mModelInstData.miModelAddCallbackFunction = [this](std::string fileName) { return addModel(fileName); };
	mModelInstData.miModelDeleteCallbackFunction = [this](std::string modelName) { deleteModel(modelName); };

	mModelInstData.miInstanceAddCallbackFunction = [this](std::shared_ptr<AssimpModel> model) { return addInstance(model); };
	mModelInstData.miInstanceAddManyCallbackFunction = [this](std::shared_ptr<AssimpModel> model, int numInstances) { addInstances(model, numInstances); };
	mModelInstData.miInstanceDeleteCallbackFunction = [this](std::shared_ptr<AssimpInstance> instance) { deleteInstance(instance);};
	mModelInstData.miInstanceCloneCallbackFunction = [this](std::shared_ptr<AssimpInstance> instance) { cloneInstance(instance); };

	initAssimpRenderData();

	if (!mModelInstData.miModelAddCallbackFunction("D:\\dev\\Graphics\\NovusEngine\\models\\Woman.gltf")) {
		Logger::log(1, "%s error: unable to load model file '%s', unknown error \n", __FUNCTION__, "D:\\dev\\Graphics\\NovusEngine\\models\\Woman.gltf");
	}
	else {
		/* select new model and new instance */
		mModelInstData.miSelectedModel = mModelInstData.miModelList.size() - 1;
		mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
	}
}

bool Renderer::hasModel(std::string modelFileName) {
	auto modelIter = std::find_if(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(),
		[modelFileName](const auto& model) {
			return model->getModelFileNamePath() == modelFileName || model->getModelFileName() == modelFileName;
		});
	return modelIter != mModelInstData.miModelList.end();
}

std::shared_ptr<AssimpModel> Renderer::getModel(std::string modelFileName) {
	auto modelIter = std::find_if(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(),
		[modelFileName](const auto& model) {
			return model->getModelFileNamePath() == modelFileName || model->getModelFileName() == modelFileName;
		});
	if (modelIter != mModelInstData.miModelList.end()) {
		return *modelIter;
	}
	return nullptr;
}

bool Renderer::addModel(std::string modelFileName) {
	if (hasModel(modelFileName)) {
		return false;
	}

	std::shared_ptr<AssimpModel> model = std::make_shared<AssimpModel>();
	if (!model->loadModel(mRenderData, modelFileName)) {
		return false;
	}

	mModelInstData.miModelList.emplace_back(model);

	/* also add a new instance here to see the model */
	addInstance(model);

	return true;
}

void Renderer::deleteModel(std::string modelFileName) {
	std::string shortModelFileName = std::filesystem::path(modelFileName).filename().generic_string();

	if (!mModelInstData.miAssimpInstances.empty()) {
		mModelInstData.miAssimpInstances.erase(
			std::remove_if(
				mModelInstData.miAssimpInstances.begin(),
				mModelInstData.miAssimpInstances.end(),
				[shortModelFileName](std::shared_ptr<AssimpInstance> instance) {
					return instance->getModel()->getModelFileName() == shortModelFileName;
				}
			),
			mModelInstData.miAssimpInstances.end()
		);
	}

	if (mModelInstData.miAssimpInstancesPerModel.count(shortModelFileName) > 0) {
		mModelInstData.miAssimpInstancesPerModel[shortModelFileName].clear();
		mModelInstData.miAssimpInstancesPerModel.erase(shortModelFileName);
	}

	/* add models to pending delete list */
	for (const auto& model : mModelInstData.miModelList) {
		if (model && (model->getTriangleCount() > 0)) {
			mModelInstData.miPendingDeleteAssimpModels.insert(model);
		}
	}

	mModelInstData.miModelList.erase(
		std::remove_if(
			mModelInstData.miModelList.begin(),
			mModelInstData.miModelList.end(),
			[modelFileName](std::shared_ptr<AssimpModel> model) {
				return model->getModelFileName() == modelFileName;
			}
		)
	);

	updateTriangleCount();
}

std::shared_ptr<AssimpInstance> Renderer::addInstance(std::shared_ptr<AssimpModel> model) {
	std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(model);
	mModelInstData.miAssimpInstances.emplace_back(newInstance);
	mModelInstData.miAssimpInstancesPerModel[model->getModelFileName()].emplace_back(newInstance);

	if (*skinningPipeline != VK_NULL_HANDLE) {
		createAssimpInstanceGPUData(newInstance);
	}

	updateTriangleCount();

	return newInstance;
}

void Renderer::addInstances(std::shared_ptr<AssimpModel> model, int numInstances) {
	size_t animClipNum = model->getAnimClips().size();
	for (int i = 0; i < numInstances; ++i) {
		int xPos = std::rand() % 50 - 25;
		int zPos = std::rand() % 50 - 25;
		int rotation = std::rand() % 360 - 180;
		int clipNr = std::rand() % animClipNum;

		std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(model, glm::vec3(xPos, 0.0f, zPos), glm::vec3(0.0f, rotation, 0.0f));
		if (animClipNum > 0) {
			InstanceSettings instSettings = newInstance->getInstanceSettings();
			instSettings.isAnimClipNr = clipNr;
			newInstance->setInstanceSettings(instSettings);
		}

		mModelInstData.miAssimpInstances.emplace_back(newInstance);
		mModelInstData.miAssimpInstancesPerModel[model->getModelFileName()].emplace_back(newInstance);
	}
	updateTriangleCount();
}

void Renderer::deleteInstance(std::shared_ptr<AssimpInstance> instance) {
	std::shared_ptr<AssimpModel> currentModel = instance->getModel();
	std::string currentModelName = currentModel->getModelFileName();

	deleteAssimpInstanceGPUData(instance);

	mModelInstData.miAssimpInstances.erase(
		std::remove_if(
			mModelInstData.miAssimpInstances.begin(),
			mModelInstData.miAssimpInstances.end(),
			[instance](std::shared_ptr<AssimpInstance> inst) {
				return inst == instance;
			}
		));


	mModelInstData.miAssimpInstancesPerModel[currentModelName].erase(
		std::remove_if(
			mModelInstData.miAssimpInstancesPerModel[currentModelName].begin(),
			mModelInstData.miAssimpInstancesPerModel[currentModelName].end(),
			[instance](std::shared_ptr<AssimpInstance> inst) {
				return inst == instance;
			}
		));

	updateTriangleCount();
}

void Renderer::cloneInstance(std::shared_ptr<AssimpInstance> instance) {
	std::shared_ptr<AssimpModel> currentModel = instance->getModel();
	std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(currentModel);
	InstanceSettings newInstanceSettings = instance->getInstanceSettings();

	/* slight offset to see new instance */
	newInstanceSettings.isWorldPosition += glm::vec3(1.0f, 0.0f, -1.0f);
	newInstance->setInstanceSettings(newInstanceSettings);

	mModelInstData.miAssimpInstances.emplace_back(newInstance);
	mModelInstData.miAssimpInstancesPerModel[currentModel->getModelFileName()].emplace_back(newInstance);

	if (*skinningPipeline != VK_NULL_HANDLE) {
		createAssimpInstanceGPUData(newInstance);
	}

	updateTriangleCount();
}

void Renderer::updateTriangleCount()
{
	mRenderData.rdTriangleCount = 0;
	for (const auto& instance : mModelInstData.miAssimpInstances) {
		mRenderData.rdTriangleCount += instance->getModel()->getTriangleCount();
	}
}

void Renderer::mainLoop()
{
	lastFrameTime = 0.0f;
	while (!glfwWindowShouldClose(window))
	{
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto        currentTime = std::chrono::high_resolution_clock::now();
		float       time = std::chrono::duration<float>(currentTime - startTime).count();
		float       deltaTime = time - lastFrameTime;
		lastFrameTime = time;
		InputSystem::Update(deltaTime);

		camera.processInput(window, camera, deltaTime);
		updateAssimpAnimations(deltaTime);
		drawFrame();
	}

	device.waitIdle();
}

void Renderer::cleanup()
{
   if (mCleanupDone)
		return;
	mCleanupDone = true;

	try
	{
		if (*device != VK_NULL_HANDLE)
			device.waitIdle();
	}
	catch (...)
	{
		// Ignore waitIdle failures (e.g. device lost) and continue best-effort cleanup.
	}

		// Release Assimp per-instance GPU resources (descriptor sets + mapped buffers)
		// before destroying the skinning descriptor pool.
		for (auto& gpuData : mAssimpGPUData)
		{
			if (gpuData.boneMapped)
			{
				gpuData.boneBufferMemory.unmapMemory();
				gpuData.boneMapped = nullptr;
			}

			for (size_t f = 0; f < gpuData.uboMemories.size() && f < gpuData.uboMapped.size(); ++f)
			{
				if (gpuData.uboMapped[f])
				{
					gpuData.uboMemories[f].unmapMemory();
					gpuData.uboMapped[f] = nullptr;
				}
			}
		}
		mAssimpGPUData.clear();

		// Now descriptor pool/pipeline resources can be safely destroyed.
		skinningWhiteView = nullptr;
		skinningWhiteImage = nullptr;
		skinningWhiteMemory = nullptr;
		skinningSampler = nullptr;
		skinningPipeline = nullptr;
		skinningPipelineLayout = nullptr;
		skinningDescriptorSetLayout = nullptr;
		skinningDescriptorPool = nullptr;

		// Cleanup all loaded Assimp model resources (VMA buffers/images).
		std::unordered_set<AssimpModel*> cleanedModels;
		for (const auto& model : mModelInstData.miModelList)
		{
			if (model && cleanedModels.insert(model.get()).second)
				model->cleanup(mRenderData);
		}
		for (const auto& model : mModelInstData.miPendingDeleteAssimpModels)
		{
			if (model && cleanedModels.insert(model.get()).second)
				model->cleanup(mRenderData);
		}

		mModelInstData.miAssimpInstances.clear();
		mModelInstData.miAssimpInstancesPerModel.clear();
		mModelInstData.miModelList.clear();
		mModelInstData.miPendingDeleteAssimpModels.clear();

  if (mRenderData.rdAllocator != VK_NULL_HANDLE)
	{
		vmaDestroyAllocator(mRenderData.rdAllocator);
		mRenderData.rdAllocator = VK_NULL_HANDLE;
	}

	if (imGui)
	{
		delete imGui;
		imGui = nullptr;
	}

	glfwDestroyWindow(window);
	glfwTerminate();
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void Renderer::framebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/)
{
	auto app = reinterpret_cast<Renderer*>(glfwGetWindowUserPointer(window));
	app->framebufferResized = true;
}

void Renderer::mousePosCallback(GLFWwindow* window, double xpos, double ypos)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddMousePosEvent(static_cast<float>(xpos), static_cast<float>(ypos));

	if (!io.WantCaptureMouse)
		Camera::mouseCallback(window, xpos, ypos);
}

void Renderer::mouseButtonCallback(GLFWwindow* window, int button, int action, int /*mods*/)
{
	ImGuiIO& io = ImGui::GetIO();
	if (button >= 0 && button < ImGuiMouseButton_COUNT)
		io.AddMouseButtonEvent(button, action == GLFW_PRESS);
}

void Renderer::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGuiIO& io = ImGui::GetIO();
	io.AddMouseWheelEvent(static_cast<float>(xoffset), static_cast<float>(yoffset));

	if (!io.WantCaptureMouse)
		Camera::scrollCallback(window, xoffset, yoffset);
}

VKAPI_ATTR vk::Bool32 VKAPI_CALL
Renderer::debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT  severity,
	vk::DebugUtilsMessageTypeFlagsEXT          type,
	const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void*)
{
	if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
		severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
	{
		std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;
	}
	return vk::False;
}

void Renderer::renderImgui()
{
	if (imGui->newFrame()) {
	}

	// Create a window for camera controls
	ImGui::Begin("Camera Controls");

	// Add a button to reset camera position
	if (ImGui::Button("Reset Camera")) {
        camera.setPosition(glm::vec3(0.0f, -90.0f, 18.0f));
		camera.setYaw(90.0f);
		camera.setPitch(-5.0f);
		camera.setMovementSpeed(40.0f);
		camera.setZoom(55.0f);
	}

	// Add sliders for camera settings
	float movementSpeed = camera.getMovementSpeed();
	if (ImGui::SliderFloat("Movement Speed", &movementSpeed, 1.0f, 100.0f)) {
		camera.setMovementSpeed(movementSpeed);
	}

	float sensitivity = camera.getMouseSensitivity();
	if (ImGui::SliderFloat("Mouse Sensitivity", &sensitivity, 0.1f, 1.0f)) {
		camera.setMouseSensitivity(sensitivity);
	}

  float zoom = camera.getZoom();
	if (ImGui::SliderFloat("Zoom", &zoom, 1.0f, 90.0f)) {
		camera.setZoom(zoom);
	}

	ImGui::End();

	ImGui::Begin("Shadow Tuning");
	ImGui::SliderFloat("Shadow Distance", &shadowSettings.shadowMaxDistance, 50.0f, 600.0f);
	ImGui::SliderFloat("Lambda", &shadowSettings.lambda, 0.0f, 1.0f);
	ImGui::SliderFloat("Bias Scale", &shadowSettings.biasScale, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Bias Min", &shadowSettings.biasMin, 0.00001f, 0.005f, "%.5f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Cascade Blend", &shadowSettings.cascadeBlendFactor, 0.0f, 0.5f);
	ImGui::SliderFloat("Coverage Padding", &shadowSettings.coveragePaddingFactor, 0.0f, 0.5f);
	ImGui::SliderFloat("Depth Padding", &shadowSettings.depthPaddingFactor, 0.0f, 1.0f);
    ImGui::SliderFloat("Caster Padding", &shadowSettings.casterPadding, 0.0f, 250.0f);
    ImGui::SliderFloat("Far Cascade Expansion", &shadowSettings.farCascadeExpansion, 1.0f, 4.0f);
	ImGui::SliderFloat("Base Padding", &shadowSettings.shadowPadding, 0.0f, 100.0f);
	ImGui::SliderFloat3("Light Direction", &shadowSettings.lightDirection.x, -1.0f, 1.0f);
	ImGui::Checkbox("Cascade Debug View", reinterpret_cast<bool*>(&shadowSettings.cascadeDebugView));
	if (ImGui::Button("Reset Shadows")) {
		shadowSettings = ShadowSettings{};
	}

	if (ImGui::CollapsingHeader("Models")) {
		/* state is changed during model deletion, so save it first */
		bool modelListEmtpy = mModelInstData.miModelList.empty();
		std::string selectedModelName;

		if (!modelListEmtpy) {
			selectedModelName = mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str();
		}

		if (modelListEmtpy) {
			ImGui::BeginDisabled();
		}

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Models :");
		ImGui::SameLine();
		ImGui::PushItemWidth(200);
		if (ImGui::BeginCombo("##ModelCombo",
			// avoid access the empty model vector
			selectedModelName.c_str())) {
			for (int i = 0; i < mModelInstData.miModelList.size(); ++i) {
				const bool isSelected = (mModelInstData.miSelectedModel == i);
				if (ImGui::Selectable(mModelInstData.miModelList.at(i)->getModelFileName().c_str(), isSelected)) {
					mModelInstData.miSelectedModel = i;
					selectedModelName = mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str();
				}

				if (isSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::PopItemWidth();

		if (modelListEmtpy) {
			ImGui::EndDisabled();
		}


		if (ImGui::Button("Import Model")) {
			IGFD::FileDialogConfig config;
			config.path = ".";
			config.countSelectionMax = 1;
			config.flags = ImGuiFileDialogFlags_Modal;
			ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
			ImGuiFileDialog::Instance()->OpenDialog("ChooseModelFile", "Choose Model File",
				"Supported Model Files{.gltf,.glb,.obj,.fbx,.dae,.mdl,.md3,.pk3}", config);
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseModelFile")) {
			if (ImGuiFileDialog::Instance()->IsOk()) {
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

				/* try to construct a relative path */
				std::filesystem::path currentPath = std::filesystem::current_path();
				std::string relativePath = std::filesystem::relative(filePathName, currentPath).generic_string();

				if (!relativePath.empty()) {
					filePathName = relativePath;
				}
				/* Windows does understand forward slashes, but std::filesystem preferres backslashes... */
				std::replace(filePathName.begin(), filePathName.end(), '\\', '/');

				if (!mModelInstData.miModelAddCallbackFunction(filePathName)) {
					Logger::log(1, "%s error: unable to load model file '%s', unknown error \n", __FUNCTION__, filePathName.c_str());
				}
				else {
					/* select new model and new instance */
					mModelInstData.miSelectedModel = mModelInstData.miModelList.size() - 1;
					mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
				}
			}
			ImGuiFileDialog::Instance()->Close();
		}

		if (modelListEmtpy) {
			ImGui::BeginDisabled();
		}

		ImGui::SameLine();
		if (ImGui::Button("Delete Model")) {
			ImGui::OpenPopup("Delete Model?");
		}

		if (ImGui::BeginPopupModal("Delete Model?", nullptr, ImGuiChildFlags_AlwaysAutoResize)) {
			ImGui::Text("Delete Model '%s'?", mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str());

			/* cheating a bit to get buttons more to the center */
			ImGui::Indent();
			ImGui::Indent();
			if (ImGui::Button("OK") || ImGui::IsKeyPressed(ImGuiKey_Enter)) {
				mModelInstData.miModelDeleteCallbackFunction(mModelInstData.miModelList.at(mModelInstData.miSelectedModel)->getModelFileName().c_str());

				/* decrement selected model index to point to model that is in list before the deleted one */
				if (mModelInstData.miSelectedModel > 0) {
					mModelInstData.miSelectedModel -= 1;
				}

				/* reset model instance to first instnace - if we have instances */
				if (!mModelInstData.miAssimpInstances.empty()) {
					mModelInstData.miSelectedInstance = 0;
				}
				ImGui::CloseCurrentPopup();
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		ImGui::SameLine();
		if (ImGui::Button("Create Instance")) {
			std::shared_ptr<AssimpModel> currentModel = mModelInstData.miModelList[mModelInstData.miSelectedModel];
			mModelInstData.miInstanceAddCallbackFunction(currentModel);
			/* select new instance */
			mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
		}

		if (ImGui::Button("Create Multiple Instances")) {
			std::shared_ptr<AssimpModel> currentModel = mModelInstData.miModelList[mModelInstData.miSelectedModel];
			mModelInstData.miInstanceAddManyCallbackFunction(currentModel, mManyInstanceCreateNum);
			mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
		}
		ImGui::SameLine();
		ImGui::SliderInt("##MassInstanceCreation", &mManyInstanceCreateNum, 1, 100, "%d");

		if (modelListEmtpy) {
			ImGui::EndDisabled();
		}
	}

	if (ImGui::CollapsingHeader("Instances")) {
		size_t numberOfInstances = mModelInstData.miAssimpInstances.size();

		ImGui::Text("Number of Instances: %ld", numberOfInstances);

		if (numberOfInstances == 0) {
			ImGui::BeginDisabled();
		}

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Selected Instance  :");
		ImGui::SameLine();
		ImGui::PushButtonRepeat(true);
		if (ImGui::ArrowButton("##Left", ImGuiDir_Left) &&
			mModelInstData.miSelectedInstance > 0) {
			mModelInstData.miSelectedInstance--;
		}
		ImGui::SameLine();
		ImGui::PushItemWidth(30);
		ImGui::DragInt("##SelInst", &mModelInstData.miSelectedInstance, 1, 0,
			mModelInstData.miAssimpInstances.size() - 1, "%3d");
		ImGui::PopItemWidth();
		ImGui::SameLine();
		if (ImGui::ArrowButton("##Right", ImGuiDir_Right) &&
			mModelInstData.miSelectedInstance < (mModelInstData.miAssimpInstances.size() - 1)) {
			mModelInstData.miSelectedInstance++;
		}
		ImGui::PopButtonRepeat();

		InstanceSettings settings;
		if (numberOfInstances > 0) {
			settings = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getInstanceSettings();
		}

		ImGui::SameLine();
		if (ImGui::Button("Clone Instance")) {
			std::shared_ptr<AssimpInstance> currentInstance = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance);
			mModelInstData.miInstanceCloneCallbackFunction(currentInstance);

			/* reset to last position for now */
			mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;

			/* read back settings for UI */
			settings = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getInstanceSettings();
		}

		/* we MUST retain the last model */
		unsigned int numberOfInstancesPerModel = 0;
		if (!mModelInstData.miAssimpInstances.empty()) {
			std::shared_ptr<AssimpInstance> currentInstance = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance);
			std::string currentModelName = currentInstance->getModel()->getModelFileName();
			numberOfInstancesPerModel = mModelInstData.miAssimpInstancesPerModel[currentModelName].size();
		}

		if (numberOfInstancesPerModel < 2) {
			ImGui::BeginDisabled();
		}

		ImGui::SameLine();
		if (ImGui::Button("Delete Instance")) {
			std::shared_ptr<AssimpInstance> currentInstance = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance);
			mModelInstData.miInstanceDeleteCallbackFunction(currentInstance);

			/* hard reset for now */
			if (mModelInstData.miSelectedInstance > 0) {
				mModelInstData.miSelectedInstance -= 1;
			}
			settings = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getInstanceSettings();
		}

		if (numberOfInstancesPerModel < 2) {
			ImGui::EndDisabled();
		}

		if (numberOfInstances == 0) {
			ImGui::EndDisabled();
		}

		/* get the new size, in case of a deletion */
		numberOfInstances = mModelInstData.miAssimpInstances.size();

		std::string baseModelName = "None";
		if (numberOfInstances > 0) {
			baseModelName = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getModel()->getModelFileName();
		}
		ImGui::Text("Base Model: %s", baseModelName.c_str());

		if (numberOfInstances == 0) {
			ImGui::BeginDisabled();
		}

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Swap Y and Z axes:     ");
		ImGui::SameLine();
		ImGui::Checkbox("##ModelAxisSwap", &settings.isSwapYZAxis);

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Model Pos (X/Y/Z):     ");
		ImGui::SameLine();
		ImGui::SliderFloat3("##ModelPos", glm::value_ptr(settings.isWorldPosition),
			-25.0f, 25.0f, "%.3f");

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Model Rotation (X/Y/Z):");
		ImGui::SameLine();
		ImGui::SliderFloat3("##ModelRot", glm::value_ptr(settings.isWorldRotation),
			-180.0f, 180.0f, "%.3f");

		ImGui::AlignTextToFramePadding();
		ImGui::Text("Model Scale:           ");
		ImGui::SameLine();
		ImGui::SliderFloat("##ModelScale", &settings.isScale,
			0.001f, 10.0f, "%.4f");

		if (ImGui::Button("Reset Instance Values")) {
			InstanceSettings defaultSettings{};
			settings = defaultSettings;
		}

		if (numberOfInstances == 0) {
			ImGui::EndDisabled();
		}

		if (numberOfInstances > 0) {
			mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->setInstanceSettings(settings);
		}
	}

	if (ImGui::CollapsingHeader("Animations")) {
		size_t numberOfInstances = mModelInstData.miAssimpInstances.size();

		InstanceSettings settings;
		size_t numberOfClips = 0;
		if (numberOfInstances > 0) {
			settings = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getInstanceSettings();
			numberOfClips = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getModel()->getAnimClips().size();
		}

		if (numberOfInstances > 0 && numberOfClips > 0) {
			std::vector<std::shared_ptr<AssimpAnimClip>> animClips = mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->getModel()->getAnimClips();

			ImGui::AlignTextToFramePadding();
			ImGui::Text("Animation Clip:");
			ImGui::SameLine();
			if (ImGui::BeginCombo("##ClipCombo",
				animClips.at(settings.isAnimClipNr)->getClipName().c_str())) {
				for (int i = 0; i < animClips.size(); ++i) {
					const bool isSelected = (settings.isAnimClipNr == i);
					if (ImGui::Selectable(animClips.at(i)->getClipName().c_str(), isSelected)) {
						settings.isAnimClipNr = i;
					}

					if (isSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			ImGui::AlignTextToFramePadding();
			ImGui::Text("Replay Speed:  ");
			ImGui::SameLine();
			ImGui::SliderFloat("##ClipSpeed", &settings.isAnimSpeedFactor, 0.0f, 2.0f, "%.3f");
		}
		else {
			/* TODO: better solution if no instances or no clips are found */
			ImGui::BeginDisabled();

			ImGui::AlignTextToFramePadding();
			ImGui::Text("Animation Clip:");
			ImGui::SameLine();
			ImGui::BeginCombo("##ClipComboDisabled", "None");

			float playSpeed = 1.0f;
			ImGui::AlignTextToFramePadding();
			ImGui::Text("Replay Speed:  ");
			ImGui::SameLine();
			ImGui::SliderFloat("##ClipSpeedDisabled", &playSpeed, 0.0f, 2.0f, "%.3f");

			ImGui::EndDisabled();
		}

		if (numberOfInstances > 0) {
			mModelInstData.miAssimpInstances.at(mModelInstData.miSelectedInstance)->setInstanceSettings(settings);
		}
	}

	ImGui::End();

	// End the frame
	ImGui::EndFrame();

	// Render to generate draw data
	ImGui::Render();
		//ImDrawData* drawData = ImGui::GetDrawData();
	//if (drawData && drawData->CmdListsCount > 0) {
	//	if (drawData->TotalVtxCount > vertexCount || drawData->TotalIdxCount > indexCount) {
	//		needsUpdateBuffers = true;
	//		return true;
	//	}
	//}
	imGui->updateBuffers();
}

// ---------------------------------------------------------------------------
// Instance / device
// ---------------------------------------------------------------------------

std::vector<const char*> Renderer::getRequiredInstanceExtensions()
{
	uint32_t glfwExtensionCount = 0;
	auto     glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enableValidationLayers)
		extensions.push_back(vk::EXTDebugUtilsExtensionName);

	return extensions;
}

bool Renderer::deviceInit()
{
	createInstance();
	setupDebugMessenger();
	createSurface();
	pickPhysicalDevice();
	createLogicalDevice();

	return true;
}

void Renderer::createInstance()
{
	constexpr vk::ApplicationInfo appInfo{ .pApplicationName = "Novus Engine",
										   .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
										   .pEngineName = "No Engine",
										   .engineVersion = VK_MAKE_VERSION(1, 0, 0),
										   .apiVersion = vk::ApiVersion14 };

	std::vector<char const*> requiredLayers;
	if (enableValidationLayers)
		requiredLayers.assign(validationLayers.begin(), validationLayers.end());

	auto layerProperties = context.enumerateInstanceLayerProperties();
	if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
		return std::ranges::none_of(layerProperties, [requiredLayer](auto const& lp) {
			return strcmp(lp.layerName, requiredLayer) == 0;
			});
		}))
	{
		throw std::runtime_error("One or more required layers are not supported!");
	}

	auto requiredExtensions = getRequiredInstanceExtensions();

	auto extensionProperties = context.enumerateInstanceExtensionProperties();
	auto unsupportedIt = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const& re) {
		return std::ranges::none_of(extensionProperties, [re](auto const& ep) {
			return strcmp(ep.extensionName, re) == 0;
			});
		});
	if (unsupportedIt != requiredExtensions.end())
		throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedIt));

	vk::InstanceCreateInfo createInfo{
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
		.ppEnabledLayerNames = requiredLayers.data(),
		.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
		.ppEnabledExtensionNames = requiredExtensions.data() };

	instance = vk::raii::Instance(context, createInfo);
}

void Renderer::setupDebugMessenger()
{
	if (!enableValidationLayers)
		return;

	vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
		vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
	vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
		vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
		vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
		.messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback };
	debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void Renderer::createSurface()
{
	VkSurfaceKHR _surface;
	if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0)
		throw std::runtime_error("failed to create window surface!");
	surface = vk::raii::SurfaceKHR(instance, _surface);
}

bool Renderer::isDeviceSuitable(vk::raii::PhysicalDevice const& pd)
{
	bool supportsVulkan1_3 = pd.getProperties().apiVersion >= vk::ApiVersion13;

	auto queueFamilies = pd.getQueueFamilyProperties();
	bool supportsGraphics = std::ranges::any_of(queueFamilies, [](auto const& qfp) {
		return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
		});

	auto availableExts = pd.enumerateDeviceExtensionProperties();
	bool supportsAllRequiredExts = std::ranges::all_of(requiredDeviceExtension, [&availableExts](auto const& req) {
		return std::ranges::any_of(availableExts, [req](auto const& avail) {
			return strcmp(avail.extensionName, req) == 0;
			});
		});

	auto features = pd.template getFeatures2<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>();
	bool supportsRequiredFeatures =
		features.template get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
		features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
		features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
		features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState &&
		features.template get<vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>().dynamicRenderingLocalRead;

	return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExts && supportsRequiredFeatures;
}

void Renderer::pickPhysicalDevice()
{
	std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
	auto const devIter = std::ranges::find_if(physicalDevices, [&](auto const& pd) { return isDeviceSuitable(pd); });
	if (devIter == physicalDevices.end())
		throw std::runtime_error("failed to find a suitable GPU!");
	physicalDevice = *devIter;
}

void Renderer::createLogicalDevice()
{
	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

	queueIndex = ~0u;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
	{
		if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
			physicalDevice.getSurfaceSupportKHR(i, *surface))
		{
			queueIndex = i;
			break;
		}
	}

	if (queueIndex == ~0u)
		throw std::runtime_error("Could not find a queue for graphics and present -> terminating");

	vk::StructureChain<vk::PhysicalDeviceFeatures2,
		vk::PhysicalDeviceVulkan11Features,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
		vk::PhysicalDeviceDynamicRenderingLocalReadFeaturesKHR>
		featureChain = {
			{.features = {.samplerAnisotropy = true } },
			{.shaderDrawParameters = true },
			{.synchronization2 = true, .dynamicRendering = true },
			{.extendedDynamicState = true },
			{.dynamicRenderingLocalRead = true }
	};

	float queuePriority = 0.5f;
	vk::DeviceQueueCreateInfo deviceQueueCreateInfo{ .queueFamilyIndex = queueIndex,
													  .queueCount = 1,
													  .pQueuePriorities = &queuePriority };

	vk::DeviceCreateInfo deviceCreateInfo{
		.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount = 1,
		.pQueueCreateInfos = &deviceQueueCreateInfo,
		.enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
		.ppEnabledExtensionNames = requiredDeviceExtension.data()
	};

	device = vk::raii::Device(physicalDevice, deviceCreateInfo);
	queue = vk::raii::Queue(device, queueIndex, 0);
}
void Renderer::recreateSwapChain()
{
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0)
	{
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	device.waitIdle();
	SwapChain::cleanupSwapChain(swapChainImageViews, swapChain);
	SwapChain::createSwapChain(physicalDevice, device, surface, window, swapChain, swapChainImages, swapChainExtent, swapChainSurfaceFormat);
	SwapChain::createImageViews(device, swapChainImages, swapChainSurfaceFormat.format, swapChainImageViews);
	DepthTarget::createDepthResources(device, physicalDevice, swapChainExtent, depthImage, depthImageMemory, depthImageView);
}

void Renderer::createGraphicsPipeline()
{
	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\slang.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\slang.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};

	config.vertexBindings = { bindingDescription };
	config.vertexAttributes = {
		attributeDescriptions.begin(),
		attributeDescriptions.end()
	};

	config.descriptorSetLayouts = { *descriptorSetLayout };
	config.colorAttachmentFormats = { swapChainSurfaceFormat.format };
	config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);

	auto pipelineBundle = Pipeline::createPipeline(device, config);
	pipelineLayout = std::move(pipelineBundle.layout);
	graphicsPipeline = std::move(pipelineBundle.pipeline);
}

bool Renderer::createPBRPipeline()
{
	try
	{
		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PushConstantRange pushConstantRange{
			.stageFlags = vk::ShaderStageFlagBits::eFragment,
			.offset = 0,
           .size = sizeof(::PushConstantBlock)
		};

		Pipeline::PipelineConfig config{};
		config.shaderStages = {
			{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\pbr.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
			{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\pbr.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
		};

		config.vertexBindings = { bindingDescription };
		config.vertexAttributes = {
			attributeDescriptions.begin(),
			attributeDescriptions.end()
		};

		config.descriptorSetLayouts = { *descriptorSetLayout };
		config.pushConstantRanges = { pushConstantRange };
		config.colorAttachmentFormats = { swapChainSurfaceFormat.format };
       config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);

		auto pipelineBundle = Pipeline::createPipeline(device, config);
		pbrPipelineLayout = std::move(pipelineBundle.layout);
		pbrPipeline = std::move(pipelineBundle.pipeline);

		return true;
	}
	catch (std::exception const& e)
	{
		std::cerr << "Error creating PBR pipeline: " << e.what() << std::endl;
		return false;
	}
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

void Renderer::recordCommandBuffer(uint32_t imageIndex)
{
	auto& commandBuffer = commandBuffers[frameIndex];
	commandBuffer.begin({});

	// Transition all cascade shadow maps to depth attachment, render each, then transition to shader read
	for (uint32_t cascade = 0; cascade < SHADOW_CASCADE_COUNT; ++cascade)
	{
		transition_image_layout(*shadowImages[cascade],
			shadowImageLayouts[cascade],
			vk::ImageLayout::eDepthAttachmentOptimal,
			vk::AccessFlags2{},
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::PipelineStageFlagBits2::eTopOfPipe,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::ImageAspectFlagBits::eDepth);
		shadowImageLayouts[cascade] = vk::ImageLayout::eDepthAttachmentOptimal;

		recordShadowPass(commandBuffer, cascade);
		commandBuffer.endRendering();

		transition_image_layout(*shadowImages[cascade],
			vk::ImageLayout::eDepthAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
			vk::AccessFlagBits2::eShaderSampledRead,
			vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eDepth);
		shadowImageLayouts[cascade] = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	beginMainPass(commandBuffer, imageIndex);
	recordScenePass(commandBuffer);
	commandBuffer.endRendering();
	recordImguiPass(commandBuffer, imageIndex);
	endMainPass(commandBuffer, imageIndex);
	commandBuffer.end();
}

void Renderer::beginMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{

	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);

	transition_image_layout(*depthImage,
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::ImageAspectFlagBits::eDepth);

	vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

	vk::RenderingAttachmentInfo attachmentInfo = {
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = clearColor
	};

	vk::RenderingAttachmentInfo depthAttachmentInfo = {
		.imageView = depthImageView,
		.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eDontCare,
		.clearValue = clearDepth
	};

	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &attachmentInfo,
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pbrPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
}

void Renderer::recordShadowPass(vk::raii::CommandBuffer& commandBuffer, uint32_t cascadeIndex)
{
	vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
	vk::RenderingAttachmentInfo depthAttachmentInfo = {
	.imageView = shadowImageViews[cascadeIndex],
	.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
	.loadOp = vk::AttachmentLoadOp::eClear,
	.storeOp = vk::AttachmentStoreOp::eStore,
	.clearValue = clearDepth
	};

	vk::RenderingInfo renderingInfo = {
		.renderArea = {.offset = { 0, 0 }, .extent = vk::Extent2D(2048, 2048)},
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pColorAttachments = nullptr,
		.pDepthAttachment = &depthAttachmentInfo
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(2048), static_cast<float>(2048), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(2048, 2048)));

	// Push the cascade index so the shadow vertex shader picks the right light matrix
	int cascadeIndexInt = static_cast<int>(cascadeIndex);
	commandBuffer.pushConstants<int>(*shadowPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, cascadeIndexInt);

	for (auto& entityPtr : entities)
	{
		auto* renderable = entityPtr->GetComponent<RenderableComponent>();
		auto* transform = entityPtr->GetComponent<TransformComponent>();
		if (!renderable || !transform)
			continue;

        UniformBuffer::updateUniformBuffer(frameIndex, renderable, transform, &camera, swapChainExtent, shadowSettings);
		vk::Buffer     vertexBuffers[] = { renderable->vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*renderable->indexBuffer, 0, vk::IndexType::eUint32);

		for (const auto& mesh : renderable->meshes)
		{
		//	const Material& material = renderable->materials[mesh.materialIndex < renderable->materials.size() ? mesh.materialIndex : 0];
		//	MaterialPushConstants::push(commandBuffer, *shadowPipelineLayout, material);
			uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable->materialDescriptorSets.size() ? mesh.materialIndex : 0;

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*shadowPipelineLayout,
				0,
				*renderable->materialDescriptorSets[descriptorMaterialIndex][frameIndex],
				nullptr);

			commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
		}
	}

}



void Renderer::recordScenePass(vk::raii::CommandBuffer& commandBuffer)
{
	for (auto& entityPtr : entities)
	{
		auto* renderable = entityPtr->GetComponent<RenderableComponent>();
		auto* transform = entityPtr->GetComponent<TransformComponent>();
		if (!renderable || !transform)
			continue;

		UniformBuffer::updateUniformBuffer(frameIndex, renderable, transform, &camera, swapChainExtent, shadowSettings);
		vk::Buffer     vertexBuffers[] = { renderable->vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*renderable->indexBuffer, 0, vk::IndexType::eUint32);

	 for (const auto& mesh : renderable->meshes)
		{
		  const Material& material = renderable->materials[mesh.materialIndex < renderable->materials.size() ? mesh.materialIndex : 0];
			MaterialPushConstants::push(commandBuffer, *pbrPipelineLayout, material);
			uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable->materialDescriptorSets.size() ? mesh.materialIndex : 0;

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*pbrPipelineLayout,
				0,
				*renderable->materialDescriptorSets[descriptorMaterialIndex][frameIndex],
				nullptr);

			commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
		}
	}

	recordAssimpSkinnedPass(commandBuffer);
}




void Renderer::recordImguiPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	imGui->drawFrame(commandBuffer, *swapChainImageViews[imageIndex]);
}




void Renderer::endMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::ePresentSrcKHR,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		{},
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eBottomOfPipe,
		vk::ImageAspectFlagBits::eColor);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
{
	vk::raii::CommandBuffer commandCopyBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
	commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy{ .size = size });
	CommandBuffer::endSingleTimeCommands(std::move(commandCopyBuffer), queue);
}

void Renderer::createVertexBuffer(RenderableComponent& gameObj)
{
	vk::DeviceSize         bufferSize = sizeof(gameObj.vertices[0]) * gameObj.vertices.size();
	vk::raii::Buffer       stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	Buffer::createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, gameObj.vertices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	Buffer::createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		gameObj.vertexBuffer, gameObj.vertexBufferMemory);

	copyBuffer(stagingBuffer, gameObj.vertexBuffer, bufferSize);
}

void Renderer::createIndexBuffer(RenderableComponent& gameObj)
{
	vk::DeviceSize         bufferSize = sizeof(gameObj.indices[0]) * gameObj.indices.size();
	if (bufferSize == 0) return;

	vk::raii::Buffer       stagingBuffer({});
	vk::raii::DeviceMemory stagingBufferMemory({});
	Buffer::createBuffer(device, physicalDevice, bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		stagingBuffer, stagingBufferMemory);

	void* dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
	memcpy(dataStaging, gameObj.indices.data(), bufferSize);
	stagingBufferMemory.unmapMemory();

	Buffer::createBuffer(device, physicalDevice, bufferSize,
		vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		gameObj.indexBuffer, gameObj.indexBufferMemory);

	copyBuffer(stagingBuffer, gameObj.indexBuffer, bufferSize);
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

void Renderer::transition_image_layout(vk::Image               image,
	vk::ImageLayout         old_layout,
	vk::ImageLayout         new_layout,
	vk::AccessFlags2        src_access_mask,
	vk::AccessFlags2        dst_access_mask,
	vk::PipelineStageFlags2 src_stage_mask,
	vk::PipelineStageFlags2 dst_stage_mask,
	vk::ImageAspectFlags    image_aspect_flags)
{
	vk::ImageMemoryBarrier2 barrier = {
		.srcStageMask = src_stage_mask,
		.srcAccessMask = src_access_mask,
		.dstStageMask = dst_stage_mask,
		.dstAccessMask = dst_access_mask,
		.oldLayout = old_layout,
		.newLayout = new_layout,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = image,
		.subresourceRange = {
			.aspectMask = image_aspect_flags,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1
		}
	};
	vk::DependencyInfo dependency_info = {
		.dependencyFlags = {},
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers = &barrier
	};
	commandBuffers[frameIndex].pipelineBarrier2(dependency_info);
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

void Renderer::loadPBRTextures(const Material& material, RenderableComponent::PBRTextures& textures)
{
	std::cout << "Loading PBR textures for material: " << material.GetName() << std::endl;

	if (!material.albedoTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.albedoTexturePath, textures.baseColorImage, textures.baseColorMemory, textures.baseColorView, true);

	if (!material.metallicRoughnessTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.metallicRoughnessTexturePath, textures.metallicRoughnessImage, textures.metallicRoughnessMemory, textures.metallicRoughnessView, false);

	if (!material.normalTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.normalTexturePath, textures.normalImage, textures.normalMemory, textures.normalView, false);

	if (!material.occlusionTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.occlusionTexturePath, textures.occlusionImage, textures.occlusionMemory, textures.occlusionView, false);

	if (!material.emissiveTexturePath.empty())
		Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, material.emissiveTexturePath, textures.emissiveImage, textures.emissiveMemory, textures.emissiveView, true);
}

void Renderer::createDefaultTextures()
{
	// White 1x1
	{
		const uint32_t white = 0xFFFFFFFF;
		vk::DeviceSize imageSize = sizeof(uint32_t);

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		Buffer::createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, &white, imageSize);
		stagingBufferMemory.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, defaultTextureImage, defaultTextureMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuffer, defaultTextureImage, 1, 1);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

		defaultTextureView = ImageView::createImageView(device, *defaultTextureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}

	// Flat normal 1x1
	{
		const uint32_t flatNormal = 0xFFFF7F7F;
		vk::DeviceSize imageSize = sizeof(uint32_t);

		vk::raii::Buffer       stagingBuffer({});
		vk::raii::DeviceMemory stagingBufferMemory({});
		Buffer::createBuffer(device, physicalDevice, imageSize, vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuffer, stagingBufferMemory);

		void* data = stagingBufferMemory.mapMemory(0, imageSize);
		memcpy(data, &flatNormal, imageSize);
		stagingBufferMemory.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal, defaultNormalImage, defaultNormalMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuffer, defaultNormalImage, 1, 1);
		Image::transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

		defaultNormalView = ImageView::createImageView(device, *defaultNormalImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}
}

void Renderer::createTextureSampler()
{
	vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
	vk::SamplerCreateInfo samplerInfo{
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.anisotropyEnable = vk::True,
		.maxAnisotropy = properties.limits.maxSamplerAnisotropy,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways,
		.borderColor = vk::BorderColor::eIntOpaqueBlack,
		.unnormalizedCoordinates = vk::False
	};
	textureSampler = vk::raii::Sampler(device, samplerInfo);
}

void Renderer::drawFrame()
{
	auto fenceResult = device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
	if (fenceResult != vk::Result::eSuccess)
		throw std::runtime_error("Failed to wait for draw fence!");

	auto [result, imageIndex] = swapChain.acquireNextImage(UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

	if (result == vk::Result::eErrorOutOfDateKHR)
	{
		recreateSwapChain();
		return;
	}
	if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
	{
		assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
		throw std::runtime_error("failed to acquire swap chain image!");
	}

	device.resetFences(*inFlightFences[frameIndex]);
	commandBuffers[frameIndex].reset();

	renderImgui();

	recordCommandBuffer(imageIndex);

	queue.waitIdle();

	vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
	const vk::SubmitInfo submitInfo{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*presentCompleteSemaphores[frameIndex],
		.pWaitDstStageMask = &waitDestinationStageMask,
		.commandBufferCount = 1,
		.pCommandBuffers = &*commandBuffers[frameIndex],
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex]
	};

	queue.submit(submitInfo, *inFlightFences[frameIndex]);

	const vk::PresentInfoKHR presentInfoKHR{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex],
		.swapchainCount = 1,
		.pSwapchains = &*swapChain,
		.pImageIndices = &imageIndex
	};

	result = queue.presentKHR(presentInfoKHR);

	if ((result == vk::Result::eSuboptimalKHR) || (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized)
	{
		framebufferResized = false;
		recreateSwapChain();
	}
	else
	{
		assert(result == vk::Result::eSuccess);
	}

	frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ---------------------------------------------------------------------------
// Assimp / skinning
// ---------------------------------------------------------------------------

void Renderer::initAssimpRenderData()
{
	// ---- VMA allocator from raii handles ----
	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
	allocatorInfo.physicalDevice   = *physicalDevice;
	allocatorInfo.device           = *device;
	allocatorInfo.instance         = *instance;

	if (vmaCreateAllocator(&allocatorInfo, &mRenderData.rdAllocator) != VK_SUCCESS)
		throw std::runtime_error("Failed to create VMA allocator for Assimp render data");

	// Raw handles consumed by old VMA-based stack (VertexBuffer, IndexBuffer, Texture)
	mRenderData.rdDevice         = *device;
	mRenderData.rdPhysicalDevice = *physicalDevice;
	mRenderData.rdInstance       = *instance;
	mRenderData.rdGraphicsQueue  = *queue;
	mRenderData.rdCommandPool    = *commandPool;

	// ---- Skinning descriptor set layout ----
	// binding 0 = UBO          (vertex)
	// binding 1 = bone SSBO    (vertex)
	// binding 2 = diffuse tex  (fragment)
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
		{ 0, vk::DescriptorType::eUniformBuffer,        1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
		{ 1, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex   },
		{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }
	}};
	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings    = bindings.data()
	};
	skinningDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	mRenderData.rdAssimpTextureDescriptorLayout = *skinningDescriptorSetLayout;

	// ---- Descriptor pool ----
	std::array<vk::DescriptorPoolSize, 3> poolSizes = {{
		{ vk::DescriptorType::eUniformBuffer,        512 },
		{ vk::DescriptorType::eStorageBuffer,        512 },
		{ vk::DescriptorType::eCombinedImageSampler, 512 }
	}};
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets       = 512,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes    = poolSizes.data()
	};
	skinningDescriptorPool      = vk::raii::DescriptorPool(device, poolInfo);
	mRenderData.rdDescriptorPool = *skinningDescriptorPool;

	// ---- 1×1 white fallback texture ----
	{
		const uint32_t white = 0xFFFFFFFF;
		vk::raii::Buffer       stagingBuf({});
		vk::raii::DeviceMemory stagingMem({});
		Buffer::createBuffer(device, physicalDevice, sizeof(uint32_t),
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuf, stagingMem);
		void* d = stagingMem.mapMemory(0, sizeof(uint32_t));
		memcpy(d, &white, sizeof(uint32_t));
		stagingMem.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			skinningWhiteImage, skinningWhiteMemory);

		auto tmpCmd = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(tmpCmd, skinningWhiteImage,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(tmpCmd, stagingBuf, skinningWhiteImage, 1, 1);
		Image::transitionImageLayout(tmpCmd, skinningWhiteImage,
			vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(tmpCmd), queue);

		skinningWhiteView = ImageView::createImageView(device, *skinningWhiteImage,
			vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}

	// ---- Sampler ----
	vk::SamplerCreateInfo samplerInfo{
		.magFilter        = vk::Filter::eLinear,
		.minFilter        = vk::Filter::eLinear,
		.mipmapMode       = vk::SamplerMipmapMode::eLinear,
		.addressModeU     = vk::SamplerAddressMode::eRepeat,
		.addressModeV     = vk::SamplerAddressMode::eRepeat,
		.addressModeW     = vk::SamplerAddressMode::eRepeat,
		.anisotropyEnable = vk::False,
		.maxAnisotropy    = 1.0f,
		.compareEnable    = vk::False,
		.compareOp        = vk::CompareOp::eAlways,
		.borderColor      = vk::BorderColor::eIntOpaqueBlack
	};
	skinningSampler = vk::raii::Sampler(device, samplerInfo);

	createSkinningPipeline();
}

void Renderer::createSkinningPipeline()
{
	auto bindingDesc = SkinnedVertex::getBindingDescription();
	auto attribDescs = SkinnedVertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\skinning.spv", vk::ShaderStageFlagBits::eVertex,   "vertMain" },
		{ "D:\\dev\\Graphics\\NovusEngine\\shaders\\skinning.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};
	config.vertexBindings         = { bindingDesc };
	config.vertexAttributes       = { attribDescs.begin(), attribDescs.end() };
	config.descriptorSetLayouts   = { *skinningDescriptorSetLayout };
	config.colorAttachmentFormats = { swapChainSurfaceFormat.format };
	config.depthAttachmentFormat  = DepthTarget::findDepthFormat(physicalDevice);
	config.cullMode               = vk::CullModeFlagBits::eNone;

	auto bundle          = Pipeline::createPipeline(device, config);
	skinningPipelineLayout = std::move(bundle.layout);
	skinningPipeline       = std::move(bundle.pipeline);
}

void Renderer::createAssimpInstanceGPUData(std::shared_ptr<AssimpInstance> instance)
{
	AssimpInstanceGPUData gpuData;
	gpuData.instance = instance;

	auto& model     = *instance->getModel();
	uint32_t boneCount = static_cast<uint32_t>(model.getBoneList().size());
	if (boneCount == 0) boneCount = 1; // avoid zero-size buffer

	vk::DeviceSize boneBufferSize = sizeof(glm::mat4) * boneCount;

	Buffer::createBuffer(device, physicalDevice, boneBufferSize,
		vk::BufferUsageFlagBits::eStorageBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		gpuData.boneBuffer, gpuData.boneBufferMemory);

	gpuData.boneMapped = gpuData.boneBufferMemory.mapMemory(0, boneBufferSize);

	// Initialise to identity
	std::vector<glm::mat4> identity(boneCount, glm::mat4(1.0f));
	memcpy(gpuData.boneMapped, identity.data(), static_cast<size_t>(boneBufferSize));

	const auto& meshes = model.getModelMeshes();
	uint32_t meshCount = static_cast<uint32_t>(meshes.empty() ? 1 : meshes.size());

	// Allocate per-frame UBO buffers (persistent, updated each frame via mapped pointer)
	gpuData.uboBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
	gpuData.uboMemories.reserve(MAX_FRAMES_IN_FLIGHT);
	gpuData.uboMapped.resize(MAX_FRAMES_IN_FLIGHT, nullptr);
	for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
	{
		vk::raii::Buffer       uboBuf({});
		vk::raii::DeviceMemory uboMem({});
		Buffer::createBuffer(device, physicalDevice, sizeof(UniformBufferObject),
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			uboBuf, uboMem);
		gpuData.uboMapped[f] = uboMem.mapMemory(0, sizeof(UniformBufferObject));
		gpuData.uboBuffers.emplace_back(std::move(uboBuf));
		gpuData.uboMemories.emplace_back(std::move(uboMem));
	}

	// Allocate one descriptor set per frame per mesh: descriptorSets[frame][mesh]
	gpuData.descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
	for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
	{
		std::vector<vk::DescriptorSetLayout> layouts(meshCount, *skinningDescriptorSetLayout);
		vk::DescriptorSetAllocateInfo allocInfo{
			.descriptorPool     = *skinningDescriptorPool,
			.descriptorSetCount = meshCount,
			.pSetLayouts        = layouts.data()
		};
		auto sets = device.allocateDescriptorSets(allocInfo);
		gpuData.descriptorSets[f].reserve(meshCount);
		for (auto& s : sets)
			gpuData.descriptorSets[f].emplace_back(std::move(s));
	}

	// Write bone SSBO, UBO and per-mesh diffuse texture bindings (static per frame)
	for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
	{
		vk::DescriptorBufferInfo boneInfo{
			.buffer = *gpuData.boneBuffer, .offset = 0, .range = boneBufferSize
		};
		vk::DescriptorBufferInfo uboInfo{
			.buffer = *gpuData.uboBuffers[f], .offset = 0, .range = sizeof(UniformBufferObject)
		};

		for (uint32_t m = 0; m < meshCount; ++m)
		{
			// Resolve the diffuse image/view for this mesh
			vk::ImageView diffuseView = *skinningWhiteView;
			if (m < meshes.size())
			{
				const VkTextureData* tex = model.getDiffuseTexture(m);
				if (tex && tex->imageView != VK_NULL_HANDLE)
					diffuseView = tex->imageView;
			}

			vk::DescriptorImageInfo imgInfo{
				.sampler     = *skinningSampler,
				.imageView   = diffuseView,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array<vk::WriteDescriptorSet, 3> writes = {{
				{
					.dstSet          = *gpuData.descriptorSets[f][m],
					.dstBinding      = 0,
					.descriptorCount = 1,
					.descriptorType  = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo     = &uboInfo
				},
				{
					.dstSet          = *gpuData.descriptorSets[f][m],
					.dstBinding      = 1,
					.descriptorCount = 1,
					.descriptorType  = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo     = &boneInfo
				},
				{
					.dstSet          = *gpuData.descriptorSets[f][m],
					.dstBinding      = 2,
					.descriptorCount = 1,
					.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo      = &imgInfo
				}
			}};
			device.updateDescriptorSets(writes, nullptr);
		}
	}

	mAssimpGPUData.emplace_back(std::move(gpuData));
}

void Renderer::deleteAssimpInstanceGPUData(std::shared_ptr<AssimpInstance> instance)
{
	device.waitIdle();
	auto it = std::find_if(mAssimpGPUData.begin(), mAssimpGPUData.end(),
		[&instance](const AssimpInstanceGPUData& d) { return d.instance == instance; });
	if (it != mAssimpGPUData.end())
	{
		if (it->boneMapped)
			it->boneBufferMemory.unmapMemory();
		for (uint32_t f = 0; f < it->uboMemories.size(); ++f)
		{
			if (it->uboMapped[f])
				it->uboMemories[f].unmapMemory();
		}
		mAssimpGPUData.erase(it);
	}
}

void Renderer::updateAssimpAnimations(float deltaTime)
{
	for (auto& gpuData : mAssimpGPUData)
	{
		auto& inst  = gpuData.instance;
		auto& model = *inst->getModel();

		if (model.hasAnimations())
			inst->updateAnimation(deltaTime);
		else
			inst->updateModelRootMatrix();

		// Upload bone matrices to the persistent mapped buffer
		const std::vector<glm::mat4>& bones = inst->getBoneMatrices();
		if (!bones.empty() && gpuData.boneMapped)
			memcpy(gpuData.boneMapped, bones.data(), bones.size() * sizeof(glm::mat4));
	}
}

void Renderer::recordAssimpSkinnedPass(vk::raii::CommandBuffer& commandBuffer)
{
	if (mAssimpGPUData.empty() || *skinningPipeline == VK_NULL_HANDLE)
		return;

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *skinningPipeline);
	commandBuffer.setViewport(0, vk::Viewport(
		0.0f, 0.0f,
		static_cast<float>(swapChainExtent.width),
		static_cast<float>(swapChainExtent.height),
		0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

	float aspect = static_cast<float>(swapChainExtent.width)
				 / static_cast<float>(swapChainExtent.height);

	for (auto& gpuData : mAssimpGPUData)
	{
		auto& model        = *gpuData.instance->getModel();
		const auto& meshes = model.getModelMeshes();
		const auto& vbos   = model.getVertexBuffers();
		const auto& ibos   = model.getIndexBuffers();

		if (meshes.empty()) continue;

		// Update the persistent per-frame UBO
		UniformBufferObject ubo{};
		ubo.model                     = gpuData.instance->getWorldTransformMatrix();
		ubo.model = glm::rotate(ubo.model, glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.model = glm::scale(ubo.model, glm::vec3(500.0f)); // Sponza is huge, scale it down
		ubo.view                      = camera.getViewMatrix();
		ubo.proj                      = camera.getProjectionMatrix(aspect, 0.1f, 600.0f);
		ubo.directionalLightDirection = glm::vec4(glm::normalize(shadowSettings.lightDirection), 0.0f);
		ubo.directionalLightColor     = glm::vec4(1.0f);
		memcpy(gpuData.uboMapped[frameIndex], &ubo, sizeof(UniformBufferObject));

		// Draw each mesh with its own descriptor set (binding 0=UBO, 1=bones, 2=diffuse)
		for (size_t i = 0; i < meshes.size(); ++i)
		{
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				*skinningPipelineLayout, 0,
				*gpuData.descriptorSets[frameIndex][i], nullptr);

			vk::Buffer     vb  = vbos[i].buffer;
			vk::DeviceSize off = 0;
			commandBuffer.bindVertexBuffers(0, vb, off);
			commandBuffer.bindIndexBuffer(ibos[i].buffer, 0, vk::IndexType::eUint32);
			commandBuffer.drawIndexed(
				static_cast<uint32_t>(meshes[i].indices.size()), 1, 0, 0, 0);
		}
	}
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

void Renderer::setupGameObjects()
{
	auto makeEntity = [&](const std::string& name,
		glm::vec3 position, glm::vec3 rotation, glm::vec3 scale,
		const std::string& modelPath) -> Entity& {
			entities.push_back(std::make_unique<Entity>(name));
			Entity& entity = *entities.back();

			auto* transform = entity.AddComponent<TransformComponent>();
			transform->SetPosition(position);
			transform->SetRotation(glm::quat(rotation));
			transform->SetScale(scale);

			auto* renderable = entity.AddComponent<RenderableComponent>();
			Model::loadModel(modelPath, *renderable);
			for (size_t i = 0; i < renderable->materials.size(); ++i)
				loadPBRTextures(renderable->materials[i], renderable->materialTextures[i]);

			return entity;
		};

	//makeEntity("FlightHelmet_Left",
	//	{ -13.0f, -10.5f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 10.0f, 10.0f, 10.0f },
	//	"../models/FlightHelmet.gltf");

	//{
	//	Entity& e = makeEntity("DamagedHelmet",
	//		{ 13.0f, -5.0f, 10.0f }, { -90.0f, 0.0f, 0.0f }, { 10.0f, 10.0f, 10.0f },
	//		"../models/DamagedHelmet.gltf");
	//}

	makeEntity("Sponza",
		{ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 0.3f, 0.3f, 0.3f },
		"models/Sponza.gltf");
}

