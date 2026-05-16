#include "renderer/renderer.h"
#include <ktx.h>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/component_wise.hpp>
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
#include "../model/AssimpModel.h"
#include "../model/AssimpInstance.h"
#include "../model/InstanceSettings.h"
#include "../../lib/ImGuiFileDialog.h"
#include "../../include/imgui_internal.h"

namespace {
	using json = nlohmann::json;

	std::vector<uint32_t> readSpvU32(const std::string& filePath)
	{
		std::ifstream file(filePath, std::ios::ate | std::ios::binary);
		if (!file.is_open())
			throw std::runtime_error("Failed to open SPIR-V file: " + filePath);

		size_t fileSize = static_cast<size_t>(file.tellg());
		if ((fileSize % sizeof(uint32_t)) != 0)
			throw std::runtime_error("Invalid SPIR-V size for file: " + filePath);

		std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(fileSize));
		return buffer;
	}

	std::string makeUniqueEntityName(entt::registry& registry, const std::string& baseName)
	{
		auto nameExists = [&](const std::string& candidate) {
			for (auto [entity, tag] : registry.view<EnttTagComponent>().each())
			{
				(void)entity;
				if (tag.name == candidate)
					return true;
			}
			return false;
			};

		if (!nameExists(baseName))
			return baseName;

		for (int suffix = 2; ; ++suffix)
		{
			std::string candidate = baseName + " (" + std::to_string(suffix) + ")";
			if (!nameExists(candidate))
				return candidate;
		}
	}

}


void Renderer::createAssimpInstanceForEntity(std::shared_ptr<AssimpModel> model, entt::entity entity)
{
	if (!model)
		return;

	// Create an AssimpInstance and attach it to the entity
	std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(model);
	auto& registry = mEnttScene.getRegistry();
	auto& animation = registry.emplace_or_replace<AnimationComponent>(entity);
	InstanceSettings newSettings = newInstance->getInstanceSettings();
	newSettings.isAnimClipNr = animation.clipIndex;
	newSettings.isAnimSpeedFactor = animation.speed;
	newInstance->setInstanceSettings(newSettings);
	AssimpSystems::RegisterInstance(mModelInstData, registry, entity, newInstance,
		[this](const std::shared_ptr<AssimpInstance>& instance) {
			if (*skinningPipeline != VK_NULL_HANDLE)
			{
				createAssimpInstanceGPUData(instance);
			}
		});
}

void Renderer::removeInstanceFromEntity(entt::entity entity)
{
	auto& registry = mEnttScene.getRegistry();
	if (!registry.valid(entity)) return;
	if (!registry.any_of<AssimpInstanceComponent>(entity)) return;
	auto& comp = registry.get<AssimpInstanceComponent>(entity);
	if (!comp.instance) return;

	// Full cleanup via the shared path (GPU data, instance lists, entity map)
	onAssimpInstanceDestroyed(comp.instance.get());

	// Remove the component so the inspector shows "Add Instance" again
	if (registry.valid(entity))
		registry.remove<AssimpInstanceComponent>(entity);
}





void Renderer::onAssimpInstanceComponentDestroyed(entt::registry& reg, entt::entity e)
{
	auto* comp = reg.try_get<AssimpInstanceComponent>(e);
	if (comp && comp->instance) {
		onAssimpInstanceDestroyed(comp->instance.get());
	}
}

void Renderer::createSkinningPipeline()
{
	auto bindingDesc = SkinnedVertex::getBindingDescription();
	auto attribDescs = SkinnedVertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "shaders\\skinning.spv", vk::ShaderStageFlagBits::eVertex,   "vertMain" },
		{ "shaders\\skinning.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};
	config.vertexBindings = { bindingDesc };
	config.vertexAttributes = { attribDescs.begin(), attribDescs.end() };
	config.descriptorSetLayouts = { *skinningDescriptorSetLayout };
	config.colorAttachmentFormats = { vk::Format::eR16G16B16A16Sfloat };
	config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);
	config.depthTestEnable = true;
	config.depthWriteEnable = true;
	config.cullMode = vk::CullModeFlagBits::eNone;
	config.depthCompareOp = vk::CompareOp::eLess;

	auto bundle = Pipeline::createPipeline(device, config);
	skinningPipelineLayout = std::move(bundle.layout);
	skinningPipeline = std::move(bundle.pipeline);
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

void Renderer::setupGameObjects()
{
	auto& registry = mEnttScene.getRegistry();
	auto makeEntity = [&](const std::string& name,
		glm::vec3 position, glm::vec3 rotation, glm::vec3 scale,
		const std::string& modelPath) -> entt::entity {
			auto ecsEntity = mEnttScene.createEntity(name);
			auto& transform = registry.emplace_or_replace<TransformComponent>(ecsEntity);
			transform.SetPosition(position);
			transform.SetRotation(glm::quat(rotation));
			transform.SetScale(scale);

			if (!modelPath.empty())
			{
				auto& renderable = registry.emplace_or_replace<RenderableComponent>(ecsEntity);
				renderable.sourceModelFile = modelPath;
				Model::loadModel(modelPath, renderable);
				for (size_t i = 0; i < renderable.materials.size(); ++i)
					loadPBRTextures(renderable.materials[i], renderable.materialTextures[i]);
			}

			return ecsEntity;
		};

	auto addPhysics = [&](entt::entity entity,
		RigidBodyType bodyType,
		ColliderShapeType shapeType,
		float mass,
		float friction,
		float restitution,
		bool useGravity,
		const glm::vec3& halfExtents,
		float radius,
		float halfHeight,
		const glm::vec3& linearVelocity) {
			auto& rigidBody = registry.emplace_or_replace<RigidBodyComponent>(entity);
			rigidBody.bodyType = bodyType;
			rigidBody.mass = mass;
			rigidBody.friction = friction;
			rigidBody.restitution = restitution;
			rigidBody.useGravity = useGravity;
			rigidBody.linearVelocity = linearVelocity;

			auto& collider = registry.emplace_or_replace<ColliderComponent>(entity);
			collider.shapeType = shapeType;
			collider.halfExtents = halfExtents;
			collider.radius = radius;
			collider.halfHeight = halfHeight;

			physicsSystem.registerEntity(entity, registry);
		};

	makeEntity("Sponza",
		{ 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f },
		"models/Sponza.gltf");

	const entt::entity floor = makeEntity(
		"Physics Arena Floor",
		{ 0.0f, 2.5f, 0.0f },
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f },
		"");
	addPhysics(floor, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.9f, 0.15f, false,
		{ 860.0f, 18.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f });

	const entt::entity wallPosX = makeEntity("Physics Arena Wall +X", { 860.0f, -260.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallPosX, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 18.0f, 300.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f });

	const entt::entity wallNegX = makeEntity("Physics Arena Wall -X", { -860.0f, -260.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallNegX, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 18.0f, 300.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f });

	const entt::entity wallPosZ = makeEntity("Physics Arena Wall +Z", { 0.0f, -260.0f, 860.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallPosZ, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 860.0f, 300.0f, 18.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f });

	const entt::entity wallNegZ = makeEntity("Physics Arena Wall -Z", { 0.0f, -260.0f, -860.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallNegZ, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 860.0f, 300.0f, 18.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f });

	const int spawnCount = std::max(12, physicsSpawnCount);
	const float spawnBaseHeight = physicsSpawnHeight;
	constexpr float twoPi = 6.28318530718f;
	for (int i = 0; i < spawnCount; ++i)
	{
		const bool damagedHelmet = (i % 2) == 0;
		const float angle = (static_cast<float>(i) / static_cast<float>(spawnCount)) * twoPi;
		const float ringRadius = 260.0f + static_cast<float>(i % 4) * 95.0f;
		const float height = spawnBaseHeight - static_cast<float>(i % 5) * 90.0f;
		const glm::vec3 position = glm::vec3(glm::cos(angle) * ringRadius, height, glm::sin(angle) * ringRadius);
		const glm::vec3 tangent = glm::normalize(glm::vec3(-glm::sin(angle), 0.0f, glm::cos(angle)));
		const glm::vec3 inward = glm::normalize(glm::vec3(-glm::cos(angle), 0.0f, -glm::sin(angle)));
		const glm::vec3 startVelocity = tangent * 125.0f + inward * 40.0f + glm::vec3(0.0f, (i % 3 == 0) ? 15.0f : -8.0f, 0.0f);

		const std::string namePrefix = damagedHelmet ? "Damaged Helmet " : "Flight Helmet ";
		const std::string modelPath = damagedHelmet ? "models/DamagedHelmet.gltf" : "models/FlightHelmet.gltf";
		const glm::vec3 scale = damagedHelmet ? glm::vec3(35.0f) : glm::vec3(100.0f);
		const glm::vec3 rotation = damagedHelmet ? glm::vec3(0.0f, angle, 0.0f) : glm::vec3(0.0f, angle + glm::radians(90.0f), 0.0f);

		entt::entity helmet = makeEntity(namePrefix + std::to_string(i + 1), position, rotation, scale, modelPath);
		addPhysics(helmet,
			RigidBodyType::Dynamic,
			ColliderShapeType::Sphere,
			damagedHelmet ? 1.2f : 2.8f,
			0.45f,
			0.6f,
			true,
			glm::vec3(0.5f),
			damagedHelmet ? 18.0f : 46.0f,
			0.5f,
			startVelocity);
	}
}

void Renderer::rebuildRenderableRuntimeResources()
{
	auto& registry = mEnttScene.getRegistry();

	for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
		(void)entity;
		if (renderable.vertices.empty() || renderable.indices.empty())
			continue;

		createVertexBuffer(renderable);
		createIndexBuffer(renderable);
		for (size_t i = 0; i < renderable.materials.size(); ++i)
			loadPBRTextures(renderable.materials[i], renderable.materialTextures[i]);
	}

	UniformBuffer::createUniformBuffers(registry, device, physicalDevice, MAX_FRAMES_IN_FLIGHT);

	DescriptorPool::createDescriptorPool(device, registry, descriptorPool, MAX_FRAMES_IN_FLIGHT);
	std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowViews = {
		   *shadowImageViews[0], *shadowImageViews[1], *shadowImageViews[2], *shadowImageViews[3], *shadowImageViews[4]
	};
	DescriptorSet::createDescriptorSets(device, registry, descriptorPool, descriptorSetLayout, defaultTextureView, defaultNormalView, textureSampler, shadowViews, shadowSampler, MAX_FRAMES_IN_FLIGHT);
}

struct FxaaPushConstantsCPU
{
	glm::vec2 rcpFrame;
	float exposure;
	float gamma;
	float bloomIntensity;
	int debugMode;
	glm::vec2 _pad{};
};

struct BloomExtractPushConstantsCPU
{
	float threshold;
	float softKnee;
	float prefilterScale;
	float _pad = 0.0f;
};

struct BloomBlurPushConstantsCPU
{
	glm::vec2 direction;
	float blurScale;
	float _pad = 0.0f;
};


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
	InputSystem::Initialize(window, &camera);
	camera.setPosition(glm::vec3(400.0f, -120.0f, 0.0f));
	camera.setYaw(180.0f);
	camera.setPitch(-5.0f);
	camera.setMovementSpeed(140.0f);
	camera.setZoom(55.0f);
	camera.getViewMatrix();
	camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f);

}

void Renderer::initVulkan()
{
	deviceInit();
	physicsSystem.initialize();
	initEnttDemoScene();
	SwapChain::createSwapChain(physicalDevice, device, surface, window, swapChain, swapChainImages, swapChainExtent, swapChainSurfaceFormat);
	SwapChain::createImageViews(device, swapChainImages, swapChainSurfaceFormat.format, swapChainImageViews);
	Image::createImage(device, physicalDevice,
		swapChainExtent.width, swapChainExtent.height,
		vk::Format::eR16G16B16A16Sfloat,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		fxaaImage, fxaaImageMemory);
	fxaaImageView = ImageView::createImageView(device, fxaaImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor);

	Image::createImage(device, physicalDevice,
		swapChainExtent.width, swapChainExtent.height,
		swapChainSurfaceFormat.format,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		viewportPreviewImage, viewportPreviewImageMemory);
	viewportPreviewImageView = ImageView::createImageView(device, viewportPreviewImage, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eUndefined;
	DescriptorSetLayout::createEntityDescriptorSetLayout(device, descriptorSetLayout, 7);
	DescriptorSetLayout::createEntityDescriptorSetLayout(device, shadowDescriptorSetLayout, 7);
	DescriptorSetLayout::createFxaaDescriptorSetLayout(device, fxaaDescriptorSetLayout);
	createBloomResources();
	if (!createPBRPipeline())
	{
		std::cerr << "Failed to create PBR pipeline" << std::endl;
	}
	std::array<glm::vec4, 4> pointLightPositions = {
	   glm::vec4(0.0f, -45.0f, 0.0f, 1.0f),
	   glm::vec4(-70.0f, -80.0f, 5.0f, 1.0f),
	   glm::vec4(10.0f, -50.0f, -75.0f, 1.0f),
	   glm::vec4(20.0f, 40.0f, -10.0f, 1.0f)
	};
	std::array<glm::vec4, 4> pointLightColors = {
		glm::vec4(1000.0f, 1000.0f, 1000.0f, 1.0f),
		glm::vec4(800.0f, 200.0f, 200.0f, 1.0f),
		glm::vec4(200.0f, 200.0f, 800.0f, 1.0f),
		glm::vec4(200.0f, 800.0f, 200.0f, 1.0f)
	};

	for (int i = 0; i < 4; ++i)
	{
		auto& registry = mEnttScene.getRegistry();
		auto pointLightEntity = mEnttScene.createEntity("Point Light");
		auto& lightTransform = registry.emplace_or_replace<TransformComponent>(pointLightEntity);
		lightTransform.SetPosition(glm::vec3(pointLightPositions[i]));
		auto& light = registry.emplace_or_replace<PointLightComponent>(pointLightEntity);
		glm::vec3 legacyColor = glm::vec3(pointLightColors[i]);
		float legacyIntensity = std::max(1.0f, std::max(legacyColor.x, std::max(legacyColor.y, legacyColor.z)));
		light.color = legacyColor / legacyIntensity;
		light.intensity = legacyIntensity;
		light.range = 120.0f;
		light.enabled = true;
	}
	ShadowPass::createPipeline(device, physicalDevice, shadowPipelineLayout, shadowPipeline, shadowDescriptorSetLayout);

	auto pipelineBundle = Pipeline::createFxaaPipeline(device, swapChainSurfaceFormat.format, fxaaDescriptorSetLayout);
	fxaaPipelineLayout = std::move(pipelineBundle.layout);
	fxaaPipeline = std::move(pipelineBundle.pipeline);


	CommandPool::init(device, queueIndex, commandPool);
	DepthTarget::createDepthResources(device, physicalDevice, swapChainExtent, depthImage, depthImageMemory, depthImageView);
	for (uint32_t i = 0; i < SHADOW_CASCADE_COUNT; ++i)
		ShadowPass::createResources(device, physicalDevice, shadowImages[i], shadowImageMemories[i], shadowImageViews[i], shadowSampler);
	createTextureSampler();
	createFxaaSampler();
	createBloomDescriptorSets();
	createBloomPipelines();
	createDefaultTextures();
	setupGameObjects();
	auto& registry = mEnttScene.getRegistry();
	for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
		(void)entity;
		createVertexBuffer(renderable);
		createIndexBuffer(renderable);
	}
	UniformBuffer::createUniformBuffers(registry, device, physicalDevice, MAX_FRAMES_IN_FLIGHT);
	DescriptorPool::createDescriptorPool(device, registry, descriptorPool, MAX_FRAMES_IN_FLIGHT);
	DescriptorPool::createFxaaDescriptorPool(device, fxaaDescriptorPool, MAX_FRAMES_IN_FLIGHT);
	std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowViews = {
		   *shadowImageViews[0], *shadowImageViews[1], *shadowImageViews[2], *shadowImageViews[3], *shadowImageViews[4]
	};
	DescriptorSet::createDescriptorSets(device, registry, descriptorPool, descriptorSetLayout, defaultTextureView, defaultNormalView, textureSampler, shadowViews, shadowSampler, MAX_FRAMES_IN_FLIGHT);
	DescriptorSet::createFxaaDescriptorSets(device, fxaaDescriptorPool, fxaaDescriptorSetLayout, fxaaImageView, bloomImageAView, depthImageView, fxaaSampler, MAX_FRAMES_IN_FLIGHT, fxaaDescriptorSets);
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
	mViewportTextureId = imGui->registerTexture(*viewportPreviewImageView, *fxaaSampler, vk::ImageLayout::eShaderReadOnlyOptimal);

	mModelInstData.miModelCheckCallbackFunction = [this](std::string fileName) { return hasModel(fileName); };
	mModelInstData.miModelAddCallbackFunction = [this](std::string fileName) { return addModel(fileName); };
	mModelInstData.miModelDeleteCallbackFunction = [this](std::string modelName) { deleteModel(modelName); };

	mModelInstData.miInstanceAddCallbackFunction = [this](std::shared_ptr<AssimpModel> model) { return addInstance(model); };
	mModelInstData.miInstanceAddManyCallbackFunction = [this](std::shared_ptr<AssimpModel> model, int numInstances) { addInstances(model, numInstances); };
	mModelInstData.miInstanceDeleteCallbackFunction = [this](std::shared_ptr<AssimpInstance> instance) { deleteInstance(instance);};
	mModelInstData.miInstanceCloneCallbackFunction = [this](std::shared_ptr<AssimpInstance> instance) { cloneInstance(instance); };

	initAssimpRenderData();

	if (!mModelInstData.miModelAddCallbackFunction("models\\Knight.glb")) {
		Logger::log(1, "%s error: unable to load model file '%s', unknown error \n", __FUNCTION__, "models\\Knight.glb");
	}
	else {
		/* select new model and new instance */
		mModelInstData.miSelectedModel = mModelInstData.miModelList.size() - 1;
		mModelInstData.miSelectedInstance = mModelInstData.miAssimpInstances.size() - 1;
	}
}

void Renderer::initEnttDemoScene()
{
	auto& registry = mEnttScene.getRegistry();
	if (!registry.view<EnttTagComponent>().empty())
		return;

	mEnttSelectedEntity = mEnttScene.createEntity("Main Camera");

	// Register a callback in the EnttScene so destroyEntity will notify us when
	// an entity holding an AssimpInstanceComponent is destroyed.
	mEnttScene.setAssimpInstanceDestroyCallback([this](std::shared_ptr<AssimpInstance> inst) {
		if (inst) {
			onAssimpInstanceDestroyed(inst.get());
		}
		});

	mUndoSnapshots.clear();
	mRedoSnapshots.clear();
	pushUndoSnapshot();
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
	auto newInstance = addInstance(model);

	// Apply known import presets for assets authored with different up/scale conventions.
	std::string fileNameLower = std::filesystem::path(modelFileName).filename().generic_string();
	std::transform(fileNameLower.begin(), fileNameLower.end(), fileNameLower.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });

	InstanceSettings settings = newInstance->getInstanceSettings();
	if (fileNameLower == "woman.gltf") {
		settings.isScale = 30.0f;
		settings.isWorldRotation.y = -90.0f;
		settings.isWorldRotation.z = 180.0f;
	}
	else if (fileNameLower == "man.gltf") {
		settings.isSwapYZAxis = true;
		settings.isWorldRotation.z = -90.0f;
		settings.isScale = 10.0f;
	}
	else if (fileNameLower == "knight.glb") {
		settings.isWorldRotation.y = -90.0f;
		settings.isWorldRotation.z = 180.0f;
		settings.isScale = 5000.0f;
	}
	newInstance->setInstanceSettings(settings);

	// Select the entity that createAssimpEnttEntity already created for this instance and
	// sync its ECS transform to match the applied presets.
	auto& registry = mEnttScene.getRegistry();
	entt::entity instanceEntity = AssimpSystems::FindEntityForInstance(registry, newInstance.get());
	if (instanceEntity != entt::null) {
		mEnttSelectedEntity = instanceEntity;
		if (auto* transform = registry.try_get<TransformComponent>(mEnttSelectedEntity)) {
			transform->SetPosition(settings.isWorldPosition);
			transform->SetRotation(glm::quat(glm::radians(settings.isWorldRotation)));
			transform->SetScale(glm::vec3(settings.isScale));
		}
	}

	return true;
}

void Renderer::deleteModel(std::string modelFileName) {
	std::string shortModelFileName = std::filesystem::path(modelFileName).filename().generic_string();

	if (!mModelInstData.miAssimpInstances.empty()) {
		std::vector<std::shared_ptr<AssimpInstance>> instancesToDestroy;
		for (const auto& instance : mModelInstData.miAssimpInstances)
		{
			if (instance && instance->getModel() && instance->getModel()->getModelFileName() == shortModelFileName)
				instancesToDestroy.emplace_back(instance);
		}

		for (const auto& instance : instancesToDestroy)
		{
			destroyAssimpEnttEntity(instance);
		}
	}

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

	/* add the deleted model to pending delete list so its GPU resources are freed next frame */
	for (const auto& model : mModelInstData.miModelList) {
		if (model && model->getModelFileName() == shortModelFileName && model->getTriangleCount() > 0) {
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
	createAssimpEnttEntity(newInstance);

	updateTriangleCount();

	return newInstance;
}

void Renderer::addInstances(std::shared_ptr<AssimpModel> model, int numInstances) {
	size_t animClipNum = model->getAnimClips().size();
	for (int i = 0; i < numInstances; ++i) {
		int xPos = std::rand() % 50 - 25;
		int zPos = std::rand() % 50 - 25;
		int rotation = std::rand() % 360 - 180;
		int clipNr = animClipNum > 0 ? std::rand() % animClipNum : 0;

		std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(model, glm::vec3(xPos, 0.0f, zPos), glm::vec3(0.0f, rotation, 0.0f));
		if (animClipNum > 0) {
			InstanceSettings instSettings = newInstance->getInstanceSettings();
			instSettings.isAnimClipNr = clipNr;
			newInstance->setInstanceSettings(instSettings);
		}

		createAssimpEnttEntity(newInstance);
	}
	updateTriangleCount();
}

void Renderer::deleteInstance(std::shared_ptr<AssimpInstance> instance) {
	if (!instance)
		return;

	destroyAssimpEnttEntity(instance);
	if (AssimpSystems::FindInstance(mModelInstData, instance.get())) {
		onAssimpInstanceDestroyed(instance.get());
	}
}

void Renderer::cloneInstance(std::shared_ptr<AssimpInstance> instance) {
	std::shared_ptr<AssimpModel> currentModel = instance->getModel();
	std::shared_ptr<AssimpInstance> newInstance = std::make_shared<AssimpInstance>(currentModel);
	InstanceSettings newInstanceSettings = instance->getInstanceSettings();

	/* slight offset to see new instance */
	newInstanceSettings.isWorldPosition += glm::vec3(1.0f, 0.0f, -1.0f);
	newInstance->setInstanceSettings(newInstanceSettings);

	createAssimpEnttEntity(newInstance);

	updateTriangleCount();
}

void Renderer::updateTriangleCount()
{
	mRenderData.rdTriangleCount = 0;
	for (const auto& instance : mModelInstData.miAssimpInstances) {
		mRenderData.rdTriangleCount += instance->getModel()->getTriangleCount();
	}
}

entt::entity Renderer::createAssimpEnttEntity(const std::shared_ptr<AssimpInstance>& instance, const std::string& namePrefix)
{
	if (!instance)
		return entt::null;

	auto* key = instance.get();
	auto& registry = mEnttScene.getRegistry();

	entt::entity existingEntity = AssimpSystems::FindEntityForInstance(registry, key);
	if (existingEntity != entt::null && registry.valid(existingEntity))
	{
		AssimpSystems::RegisterInstance(mModelInstData, registry, existingEntity, instance,
			[this](const std::shared_ptr<AssimpInstance>& registeredInstance) {
				if (*skinningPipeline != VK_NULL_HANDLE)
				{
					createAssimpInstanceGPUData(registeredInstance);
				}
			});
		auto* transform = registry.try_get<TransformComponent>(existingEntity);
		if (transform)
		{
			InstanceSettings settings = instance->getInstanceSettings();
			transform->SetPosition(settings.isWorldPosition);
			transform->SetRotation(glm::quat(glm::radians(settings.isWorldRotation)));
			transform->SetScale(glm::vec3(settings.isScale));
			auto& animation = registry.emplace_or_replace<AnimationComponent>(existingEntity);
			animation.clipIndex = settings.isAnimClipNr;
			animation.speed = settings.isAnimSpeedFactor;
		}
		return existingEntity;
	}

	std::string modelName = "Instance";
	if (instance->getModel())
		modelName = instance->getModel()->getModelFileName();

	std::string entityName = makeUniqueEntityName(registry, namePrefix + modelName);
	entt::entity entity = mEnttScene.createEntity(entityName);
	auto& transform = registry.emplace_or_replace<TransformComponent>(entity);
	InstanceSettings settings = instance->getInstanceSettings();
	transform.SetPosition(settings.isWorldPosition);
	transform.SetRotation(glm::quat(glm::radians(settings.isWorldRotation)));
	transform.SetScale(glm::vec3(settings.isScale));
	auto& animation = registry.emplace_or_replace<AnimationComponent>(entity);
	animation.clipIndex = settings.isAnimClipNr;
	animation.speed = settings.isAnimSpeedFactor;

	AssimpSystems::RegisterInstance(mModelInstData, registry, entity, instance,
		[this](const std::shared_ptr<AssimpInstance>& registeredInstance) {
			if (*skinningPipeline != VK_NULL_HANDLE)
			{
				createAssimpInstanceGPUData(registeredInstance);
			}
		});
	return entity;
}

void Renderer::destroyAssimpEnttEntity(const std::shared_ptr<AssimpInstance>& instance)
{
	if (!instance)
		return;

	auto& registry = mEnttScene.getRegistry();
	entt::entity entity = AssimpSystems::FindEntityForInstance(registry, instance.get());
	if (entity == entt::null)
		return;

	if (mEnttScene.isValid(entity))
		mEnttScene.destroyEntity(entity);

	if (mEnttSelectedEntity == entity)
		mEnttSelectedEntity = entt::null;
}

void Renderer::mainLoop()
{
	lastFrameTime = 0.0f;
	currentFrameIndex = 0;
	while (!glfwWindowShouldClose(window))
	{
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto        currentTime = std::chrono::high_resolution_clock::now();
		float       time = std::chrono::duration<float>(currentTime - startTime).count();
		float       deltaTime = time - lastFrameTime;
		lastFrameTime = time;
		InputSystem::Update(deltaTime);

		camera.processInput(window, camera, deltaTime);
		auto& reg = mEnttScene.getRegistry();

		if (sceneState == SceneState::PLAY)
		{
			physicsSystem.step(deltaTime, reg);
			updateAssimpAnimations(deltaTime);
		}
		updateAssimpAnimations(0.0f);

		drawFrame();
		currentFrameIndex++;
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
	shadowSkinningPipeline = nullptr;
	shadowSkinningPipelineLayout = nullptr;
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

	physicsSystem.shutdown();

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

void Renderer::buildEditorDockspace()
{
	const ImGuiViewport* mainViewport = ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(mainViewport->Pos);
	ImGui::SetNextWindowSize(mainViewport->Size);
	ImGui::SetNextWindowViewport(mainViewport->ID);

	ImGuiWindowFlags rootWindowFlags =
		ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_NoResize |
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoBringToFrontOnFocus |
		ImGuiWindowFlags_NoNavFocus |
		ImGuiWindowFlags_NoBackground |
		ImGuiWindowFlags_MenuBar;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
	ImGui::Begin("##EditorRoot", nullptr, rootWindowFlags);
	ImGui::PopStyleVar(3);

	ImGuiID dockspaceId = ImGui::GetID("EditorDockspace");
	ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_None);

	if (ImGui::BeginMenuBar())
	{
		ImGui::TextUnformatted("NovusEngine Editor");
		ImGui::EndMenuBar();
	}

	if (!mEditorDockLayoutInitialized)
	{
		mEditorDockLayoutInitialized = true;

		ImGui::DockBuilderRemoveNode(dockspaceId);
		ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
		ImGui::DockBuilderSetNodeSize(dockspaceId, mainViewport->Size);

		ImGuiID dockMain = dockspaceId;
		ImGuiID dockLeft = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left, 0.20f, nullptr, &dockMain);
		ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right, 0.28f, nullptr, &dockMain);
		ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down, 0.30f, nullptr, &dockMain);
		ImGuiID dockLeftBottom = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Down, 0.45f, nullptr, &dockLeft);

		ImGui::DockBuilderDockWindow("Viewport", dockMain);
		ImGui::DockBuilderDockWindow("ECS Scene", dockLeft);
		ImGui::DockBuilderDockWindow("ECS Lights", dockLeftBottom);
		ImGui::DockBuilderDockWindow("ECS Inspector", dockRight);
		ImGui::DockBuilderDockWindow("Camera Controls", dockRight);
		ImGui::DockBuilderDockWindow("Shadow Tuning", dockBottom);
		ImGui::DockBuilderDockWindow("Animation Controls", dockBottom);
		ImGui::DockBuilderDockWindow("Post Processing", dockBottom);
		ImGui::DockBuilderDockWindow("Physics Demo", dockBottom);

		ImGui::DockBuilderFinish(dockspaceId);
	}

	ImGui::End();
}

void Renderer::renderViewportPanel()
{
	if (!uiShowViewport)
		return;

	ImGui::Begin("Viewport");
	mViewportFocused = ImGui::IsWindowFocused();
	mViewportHovered = ImGui::IsWindowHovered();

	const ImVec2 imagePos = ImGui::GetCursorScreenPos();
	const ImVec2 available = ImGui::GetContentRegionAvail();
	const float targetAspect = static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height);
	const float availableAspect = (available.y > 0.0f) ? (available.x / available.y) : targetAspect;
	ImVec2 drawSize = available;
	if (availableAspect > targetAspect)
	{
		drawSize.x = available.y * targetAspect;
		drawSize.y = available.y;
	}
	else
	{
		drawSize.x = available.x;
		drawSize.y = (targetAspect > 0.0f) ? (available.x / targetAspect) : available.y;
	}
	const ImVec2 drawPos = ImVec2(
		imagePos.x + (available.x - drawSize.x) * 0.5f,
		imagePos.y + (available.y - drawSize.y) * 0.5f);
	ImGuizmo::SetDrawlist(ImGui::GetWindowDrawList());
	ImGuizmo::SetRect(drawPos.x, drawPos.y, drawSize.x, drawSize.y);
	if (mViewportTextureId && drawSize.x > 1.0f && drawSize.y > 1.0f)
	{
		if (drawPos.y > imagePos.y)
			ImGui::Dummy(ImVec2(0.0f, drawPos.y - imagePos.y));

		const float indentX = drawPos.x - ImGui::GetCursorScreenPos().x;
		if (indentX > 0.0f)
			ImGui::Indent(indentX);

		ImGui::Image(mViewportTextureId, drawSize, ImVec2(0.0f, 0.0f), ImVec2(1.0f, 1.0f));

		if (indentX > 0.0f)
			ImGui::Unindent(indentX);

		const float bottomPad = (imagePos.y + available.y) - (drawPos.y + drawSize.y);
		if (bottomPad > 0.0f)
			ImGui::Dummy(ImVec2(0.0f, bottomPad));
	}
	else
	{
		ImGui::TextUnformatted("Scene output unavailable");
	}

	ImGui::End();
}

void Renderer::renderImgui()
{
	imGui->newFrame();
	buildEditorDockspace();
	renderViewportPanel();
	auto& registry = mEnttScene.getRegistry();
	const bool isEditMode = (sceneState == SceneState::EDIT);

	ImGui::Begin("Play Mode");
	ImGui::Checkbox("Viewport", &uiShowViewport);
	ImGui::SameLine();
	ImGui::Checkbox("Camera", &uiShowCameraControls);
	ImGui::SameLine();
	ImGui::Checkbox("Play HUD", &uiShowPlayHud);
	ImGui::SameLine();
	ImGui::Checkbox("Post UI", &uiShowPostProcessingWindow);
	ImGui::SameLine();
	ImGui::Checkbox("Shadow UI", &uiShowShadowTuningWindow);
	ImGui::SameLine();
	ImGui::Checkbox("Physics UI", &uiShowPhysicsWindow);

	ImGui::Checkbox("Render Shadows", &renderEnableShadows);
	ImGui::SameLine();
	ImGui::Checkbox("Post Processing", &renderEnablePostProcessing);
	ImGui::SameLine();
	ImGui::BeginDisabled(!renderEnablePostProcessing);
	ImGui::Checkbox("FXAA", &renderEnableFxaa);
	ImGui::SameLine();
	ImGui::Checkbox("Bloom", &renderEnableBloom);
	ImGui::EndDisabled();

	ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
	ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f, 8.0f));

	const bool canPlay = isEditMode;
	const bool canStop = !isEditMode;

	if (canPlay)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.20f, 0.58f, 0.28f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.68f, 0.33f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.48f, 0.23f, 1.0f));
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
	}
	if (ImGui::Button(ICON_FA_PLAY "  Play") && canPlay)
		enterPlayMode();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();

	if (canStop)
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.72f, 0.22f, 0.22f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.82f, 0.27f, 0.27f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.62f, 0.16f, 0.16f, 1.0f));
	}
	else
	{
		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.16f, 0.16f, 0.16f, 1.0f));
	}
	if (ImGui::Button(ICON_FA_STOP "  Stop") && canStop)
		exitPlayMode();
	ImGui::PopStyleColor(3);

	ImGui::SameLine();
	ImGui::TextUnformatted(sceneState == SceneState::PLAY ? "State: PLAY" : "State: EDIT");
	if (sceneState == SceneState::PLAY)
	{
		ImGui::SameLine();
		ImGui::Checkbox("Show Debug UI", &playShowDebugUI);
	}

	ImGui::PopStyleVar(2);
	ImGui::End();

	shadowSettings.enabled = renderEnableShadows ? 1.0f : 0.0f;

	// Create a window for camera controls
	if (uiShowCameraControls)
	{
		ImGui::SetNextWindowBgAlpha(1.0f);
		ImGui::Begin("Camera Controls");

		// Add a button to reset camera position
		if (ImGui::Button("Reset Camera")) {
			camera.setPosition(glm::vec3(400.0f, -120.0f, 0.0f));
			camera.setYaw(180.0f);
			camera.setPitch(-5.0f);
			camera.setMovementSpeed(140.0f);
			camera.setZoom(55.0f);
			camera.getViewMatrix();
			camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f);
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

		const glm::vec3 camPos = camera.getPosition();
		ImGui::Text("Position: (%.2f, %.2f, %.2f)", camPos.x, camPos.y, camPos.z);

		ImGuiIO& io = ImGui::GetIO();
		const float fps = io.Framerate;
		const float frameMs = fps > 0.0f ? (1000.0f / fps) : 0.0f;
		ImGui::Text("FPS: %.1f (%.2f ms)", fps, frameMs);

		ImGui::End();
	}

	if (isEditMode)
	{
		renderEnttEditor(camera.getViewMatrix(), camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f));

		if (uiShowShadowTuningWindow)
		{
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

			ImGui::End();
		}

		ImGui::Begin("Animation Controls");

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

		ImGui::Separator();
		ImGui::TextUnformatted("Instance and animation editing moved to ECS Inspector.");

		ImGui::End();
	}
	else
	{
		if (uiShowPlayHud)
			ImGui::Begin("Play HUD");
		else
			ImGui::Begin("Play HUD", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);

		if (uiShowPlayHud)
		{
			ImGui::TextUnformatted("Runtime UI");
			ImGui::Separator();
			ImGui::TextUnformatted("W/A/S/D + Mouse: Move camera");
			ImGui::TextUnformatted("Space/Ctrl: Up/Down");
			ImGui::TextUnformatted("Esc: Toggle mouse capture");

			int rigidBodyCount = 0;
			for (auto entity : registry.view<RigidBodyComponent>()) {
				(void)entity;
				++rigidBodyCount;
			}
			int colliderCount = 0;
			for (auto entity : registry.view<ColliderComponent>()) {
				(void)entity;
				++colliderCount;
			}
			ImGui::Text("RigidBodies: %d", rigidBodyCount);
			ImGui::Text("Colliders: %d", colliderCount);
		}
		ImGui::End();
	}

	if ((isEditMode || playShowDebugUI) && uiShowPostProcessingWindow)
	{
		ImGui::Begin("Post Processing");
		ImGui::SliderFloat("FXAA Exposure", &fxaaExposure, 0.1f, 8.0f, "%.2f");
		ImGui::SliderFloat("FXAA Gamma", &fxaaGamma, 1.0f, 3.0f, "%.2f");
		ImGui::Checkbox("Bloom Enabled", &bloomEnabled);
		ImGui::SliderFloat("Bloom Threshold", &bloomThreshold, 0.05f, 4.0f, "%.2f");
		ImGui::SliderFloat("Bloom Soft Knee", &bloomSoftKnee, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat("Bloom Prefilter", &bloomPrefilterScale, 0.5f, 6.0f, "%.2f");
		ImGui::SliderFloat("Bloom Intensity", &bloomIntensity, 0.0f, 4.0f, "%.3f");
		ImGui::SliderFloat("Bloom Blur Scale", &bloomBlurScale, 0.25f, 3.0f, "%.2f");
		ImGui::SliderInt("Bloom Blur Passes", &bloomBlurPasses, 1, 8);
		const char* debugModes[] = { "Final", "Scene HDR", "Bloom Only" };
		ImGui::Combo("Post Debug", &postProcessDebugMode, debugModes, IM_ARRAYSIZE(debugModes));
		ImGui::End();

		if (uiShowPhysicsWindow)
			ImGui::Begin("Physics Demo");
		else
			ImGui::Begin("Physics Demo", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoInputs);

		if (uiShowPhysicsWindow)
		{
			ImGui::Checkbox("Pause Physics", &physicsPaused);
			physicsSystem.setPaused(physicsPaused);

			glm::vec3 gravity = physicsSystem.getGravity();
			if (ImGui::SliderFloat3("Gravity", &gravity.x, -30.0f, 30.0f, "%.2f"))
			{
				physicsSystem.setGravity(gravity);
			}

			ImGui::SliderInt("Spawn Count", &physicsSpawnCount, 1, 128);
			ImGui::SliderFloat("Spawn Height", &physicsSpawnHeight, 0.0f, -3120.0f, "%.1f");

			int rigidBodyCount = 0;
			for (auto entity : registry.view<RigidBodyComponent>()) {
				(void)entity;
				++rigidBodyCount;
			}
			int colliderCount = 0;
			for (auto entity : registry.view<ColliderComponent>()) {
				(void)entity;
				++colliderCount;
			}
			ImGui::Text("RigidBodies: %d", rigidBodyCount);
			ImGui::Text("Colliders: %d", colliderCount);

			if (ImGui::Button("Rebuild Physics Registration"))
			{
				physicsSystem.clear();
				for (auto [entity, rb, col, tr] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each())
				{
					(void)rb;
					(void)col;
					(void)tr;
					entt::entity e = entity;
					physicsSystem.registerEntity(e, registry);
				}
			}

			if (ImGui::Button("Reset Physics"))
			{
				physicsSystem.reset();
			}
		}
		ImGui::End();
	}

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

void Renderer::enterPlayMode()
{
	std::ofstream outFile(mEditorSceneFilePath, std::ios::out | std::ios::trunc);
	if (outFile.is_open())
		outFile << serializeEnttScene();

	sceneState = SceneState::PLAY;
	currentFrameIndex = 0;
	physicsSystem.setPaused(false);
}

void Renderer::exitPlayMode()
{
	std::ifstream inFile(mEditorSceneFilePath);
	if (inFile.is_open()) {
		std::string jsonContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
		pushUndoSnapshot();
		deserializeEnttScene(jsonContent);
	}

	sceneState = SceneState::EDIT;
	currentFrameIndex = 0;
}

void Renderer::renderEnttEditor(glm::mat4 view, glm::mat4 projection)
{
	auto& registry = mEnttScene.getRegistry();
	static entt::entity sNameEditEntity = entt::null;
	static char sNameEditBuffer[256]{};
	auto isInMultiSelection = [&](entt::entity entity) {
		return std::find(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), entity) != mEnttMultiSelection.end();
		};
	auto removeFromMultiSelection = [&](entt::entity entity) {
		auto it = std::remove(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), entity);
		if (it != mEnttMultiSelection.end())
			mEnttMultiSelection.erase(it, mEnttMultiSelection.end());
		};
	auto detachHierarchy = [&](entt::entity entity) {
		auto* hc = registry.try_get<HierarchyComponent>(entity);
		if (!hc)
			return;

		if (hc->parent != entt::null && registry.valid(hc->parent)) {
			auto* parentHc = registry.try_get<HierarchyComponent>(hc->parent);
			if (parentHc) {
				auto it = std::remove(parentHc->children.begin(), parentHc->children.end(), entity);
				if (it != parentHc->children.end())
					parentHc->children.erase(it, parentHc->children.end());
			}
		}

		for (auto child : hc->children) {
			if (!registry.valid(child))
				continue;
			auto& childHc = registry.emplace_or_replace<HierarchyComponent>(child);
			childHc.parent = entt::null;
		}
		hc->children.clear();
		hc->parent = entt::null;
		};
	auto isDescendantOf = [&](entt::entity child, entt::entity possibleAncestor) {
		entt::entity cursor = child;
		while (cursor != entt::null && registry.valid(cursor)) {
			auto* hc = registry.try_get<HierarchyComponent>(cursor);
			if (!hc)
				return false;
			cursor = hc->parent;
			if (cursor == possibleAncestor)
				return true;
		}
		return false;
		};
	auto recalcLocalFromParent = [&](entt::entity child) {
		TransformSystems::RecalculateLocalFromParent(registry, child);
		};
	std::function<void(entt::entity)> applyWorldFromParent = [&](entt::entity parentEntity) {
		TransformSystems::ApplyWorldFromParent(registry, parentEntity);
		};
	auto setParent = [&](entt::entity child, entt::entity newParent) {
		if (!registry.valid(child))
			return;
		if (newParent == child)
			return;
		if (newParent != entt::null && isDescendantOf(newParent, child))
			return;

		auto& childHc = registry.emplace_or_replace<HierarchyComponent>(child);
		if (childHc.parent != entt::null && registry.valid(childHc.parent)) {
			auto* oldParentHc = registry.try_get<HierarchyComponent>(childHc.parent);
			if (oldParentHc) {
				auto it = std::remove(oldParentHc->children.begin(), oldParentHc->children.end(), child);
				if (it != oldParentHc->children.end())
					oldParentHc->children.erase(it, oldParentHc->children.end());
			}
		}

		childHc.parent = newParent;
		if (newParent != entt::null && registry.valid(newParent)) {
			auto& parentHc = registry.emplace_or_replace<HierarchyComponent>(newParent);
			if (std::find(parentHc.children.begin(), parentHc.children.end(), child) == parentHc.children.end()) {
				parentHc.children.push_back(child);
			}
		}

		recalcLocalFromParent(child);
		if (newParent != entt::null)
			applyWorldFromParent(newParent);
		};
	auto selectEntityAndSync = [&](entt::entity entity) {
		mEnttSelectedEntity = entity;
		mEnttMultiSelection.clear();
		mEnttMultiSelection.push_back(entity);
		sNameEditEntity = entt::null;

		if (auto* assimp = registry.try_get<AssimpInstanceComponent>(entity); assimp && assimp->instance) {
			auto instanceIt = std::find(mModelInstData.miAssimpInstances.begin(), mModelInstData.miAssimpInstances.end(), assimp->instance);
			if (instanceIt != mModelInstData.miAssimpInstances.end()) {
				mModelInstData.miSelectedInstance = static_cast<int>(std::distance(mModelInstData.miAssimpInstances.begin(), instanceIt));
			}

			auto model = assimp->instance->getModel();
			if (model) {
				auto modelIt = std::find(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(), model);
				if (modelIt != mModelInstData.miModelList.end()) {
					mModelInstData.miSelectedModel = static_cast<int>(std::distance(mModelInstData.miModelList.begin(), modelIt));
				}
			}
		}
		};

	// Keep selection state clean to avoid deleting stale entities.
	mEnttMultiSelection.erase(
		std::remove_if(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), [&](entt::entity e) {
			return !mEnttScene.isValid(e);
			}),
		mEnttMultiSelection.end());
	std::sort(mEnttMultiSelection.begin(), mEnttMultiSelection.end(), [](entt::entity a, entt::entity b) {
		return entt::to_integral(a) < entt::to_integral(b);
		});
	mEnttMultiSelection.erase(std::unique(mEnttMultiSelection.begin(), mEnttMultiSelection.end()), mEnttMultiSelection.end());

	if (mEnttScene.isValid(mEnttSelectedEntity) && mEnttMultiSelection.empty()) {
		mEnttMultiSelection.push_back(mEnttSelectedEntity);
	}

	TransformSystems::UpdateHierarchyFromRoots(registry);

	auto duplicateEntity = [&](entt::entity sourceEntity) -> entt::entity {
		if (!mEnttScene.isValid(sourceEntity))
			return entt::null;

		auto* sourceTag = registry.try_get<EnttTagComponent>(sourceEntity);
		auto* sourceTransform = registry.try_get<TransformComponent>(sourceEntity);
		auto* sourceAnimation = registry.try_get<AnimationComponent>(sourceEntity);
		auto* sourceAssimp = registry.try_get<AssimpInstanceComponent>(sourceEntity);

		if (sourceAssimp && sourceAssimp->instance) {
			size_t previousInstanceCount = mModelInstData.miAssimpInstances.size();
			cloneInstance(sourceAssimp->instance);
			if (mModelInstData.miAssimpInstances.size() <= previousInstanceCount)
				return entt::null;

			auto duplicatedInstance = mModelInstData.miAssimpInstances.back();
			entt::entity duplicatedEntity = AssimpSystems::FindEntityForInstance(registry, duplicatedInstance.get());
			if (duplicatedEntity == entt::null)
				return entt::null;

			if (auto* duplicatedTag = registry.try_get<EnttTagComponent>(duplicatedEntity); duplicatedTag && sourceTag) {
				duplicatedTag->name = makeUniqueEntityName(registry, sourceTag->name + " Copy");
			}

			return duplicatedEntity;
		}

		std::string duplicatedName = sourceTag ? makeUniqueEntityName(registry, sourceTag->name + " Copy") : makeUniqueEntityName(registry, "Entity Copy");
		entt::entity duplicatedEntity = mEnttScene.createEntity(duplicatedName);

		if (sourceTransform) {
			auto& duplicatedTransform = registry.emplace_or_replace<TransformComponent>(duplicatedEntity);
			duplicatedTransform.SetPosition(sourceTransform->GetPosition() + glm::vec3(1.0f, 0.0f, -1.0f));
			duplicatedTransform.SetRotation(sourceTransform->GetRotation());
			duplicatedTransform.SetScale(sourceTransform->GetScale());
		}

		if (sourceAnimation) {
			registry.emplace_or_replace<AnimationComponent>(duplicatedEntity, *sourceAnimation);
		}

		return duplicatedEntity;
		};

	ImGui::Begin("ECS Scene");
	if (ImGui::Button("Undo"))
		performUndo();
	ImGui::SameLine();
	if (ImGui::Button("Redo"))
		performRedo();
	ImGui::SameLine();
	if (ImGui::Button("Save Scene")) {
		std::ofstream outFile(mSceneFilePath, std::ios::out | std::ios::trunc);
		if (outFile.is_open()) {
			outFile << serializeEnttScene();
		}
	}
	ImGui::SameLine();
	if (ImGui::Button("Load Scene")) {
		std::ifstream inFile(mSceneFilePath);
		if (inFile.is_open()) {
			std::string jsonContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
			pushUndoSnapshot();
			deserializeEnttScene(jsonContent);
		}
	}

	if (ImGui::Button("Create Entity"))
	{
		pushUndoSnapshot();
		mEnttSelectedEntity = mEnttScene.createEntity("New Entity");
		mEnttMultiSelection.clear();
		mEnttMultiSelection.push_back(mEnttSelectedEntity);
	}	ImGui::Separator();

	registry.view<EnttTagComponent>().each([&](entt::entity entity, EnttTagComponent& tag)
		{
			ImGui::PushID(static_cast<int>(entt::to_integral(entity)));
			const bool isSelected = isInMultiSelection(entity);
			if (ImGui::Selectable(tag.name.c_str(), isSelected)) {
				const bool ctrlDown = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl) || ImGui::GetIO().KeyCtrl;
				if (ctrlDown) {
					if (isInMultiSelection(entity)) {
						removeFromMultiSelection(entity);
						if (mEnttSelectedEntity == entity) {
							mEnttSelectedEntity = mEnttMultiSelection.empty() ? entt::null : mEnttMultiSelection.back();
						}
					}
					else {
						mEnttMultiSelection.push_back(entity);
						mEnttSelectedEntity = entity;
					}
				}
				else {
					selectEntityAndSync(entity);
				}
			}
			ImGui::PopID();
		});
	ImGui::Text("Selected: %d", static_cast<int>(mEnttMultiSelection.size()));
	ImGui::End();

	ImGui::Begin("ECS Lights");
	if (ImGui::Button("Create Point Light")) {
		pushUndoSnapshot();
		auto lightEntity = mEnttScene.createEntity("Point Light");
		auto& lightTransform = registry.emplace_or_replace<TransformComponent>(lightEntity);
		lightTransform.SetPosition(camera.getPosition() + camera.getFront() * 8.0f);
		auto& light = registry.emplace_or_replace<PointLightComponent>(lightEntity);
		light.color = glm::vec3(1.0f);
		light.intensity = 800.0f;
		light.range = 100.0f;
		light.enabled = true;
		selectEntityAndSync(lightEntity);
	}

	entt::entity lightEntityToDelete = entt::null;
	for (auto [entity, tag, light, transform] : registry.view<EnttTagComponent, PointLightComponent, TransformComponent>().each())
	{
		ImGui::PushID(static_cast<int>(entt::to_integral(entity)));
		const bool selected = (mEnttSelectedEntity == entity);
		if (ImGui::Selectable(tag.name.c_str(), selected)) {
			selectEntityAndSync(entity);
		}
		ImGui::SameLine();
		ImGui::Checkbox("##Enabled", &light.enabled);
		ImGui::SameLine();
		ImGui::Text("I: %.0f", light.intensity);
		ImGui::SameLine();
		if (ImGui::SmallButton("Delete")) {
			lightEntityToDelete = entity;
		}
		ImGui::PopID();
	}

	if (lightEntityToDelete != entt::null && mEnttScene.isValid(lightEntityToDelete)) {
		pushUndoSnapshot();
		detachHierarchy(lightEntityToDelete);
		mEnttScene.destroyEntity(lightEntityToDelete);
		if (mEnttSelectedEntity == lightEntityToDelete) {
			mEnttSelectedEntity = entt::null;
			mEnttMultiSelection.clear();
		}
	}
	ImGui::End();

	ImGui::SetNextWindowBgAlpha(1.0f);
	ImGui::Begin("ECS Inspector");
	if (mEnttScene.isValid(mEnttSelectedEntity))
	{
		auto* tag = registry.try_get<EnttTagComponent>(mEnttSelectedEntity);
		auto* transform = registry.try_get<TransformComponent>(mEnttSelectedEntity);
		auto* animation = registry.try_get<AnimationComponent>(mEnttSelectedEntity);

		if (tag)
		{
			if (sNameEditEntity != mEnttSelectedEntity) {
				sNameEditEntity = mEnttSelectedEntity;
				std::snprintf(sNameEditBuffer, sizeof(sNameEditBuffer), "%s", tag->name.c_str());
			}

			if (ImGui::InputText("Name", sNameEditBuffer, sizeof(sNameEditBuffer))) {
				if (ImGui::IsItemActivated()) {
					pushUndoSnapshot();
				}
				tag->name = sNameEditBuffer;
			}
		}

		if (transform)
		{

			if (ImGui::IsKeyPressed(ImGuiKey_T))
				currentOperation = ImGuizmo::OPERATION::TRANSLATE;
			if (ImGui::IsKeyPressed(ImGuiKey_E))
				currentOperation = ImGuizmo::OPERATION::ROTATE;
			if (ImGui::IsKeyPressed(ImGuiKey_S))
				currentOperation = ImGuizmo::OPERATION::SCALE;
			if (ImGui::RadioButton("Translate", currentOperation == ImGuizmo::OPERATION::TRANSLATE))
			{
				currentOperation = ImGuizmo::OPERATION::TRANSLATE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate", currentOperation == ImGuizmo::OPERATION::ROTATE))
			{
				currentOperation = ImGuizmo::OPERATION::ROTATE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale", currentOperation == ImGuizmo::OPERATION::SCALE))
			{
				currentOperation = ImGuizmo::OPERATION::SCALE;
				ImGuizmo::Enable(true);
			}
			ImGui::SameLine();
			if (ImGui::RadioButton("None", currentOperation == -1))
				ImGuizmo::Enable(false);

			glm::vec3 position = transform->GetPosition();
			glm::vec3 rotation = glm::eulerAngles(transform->GetRotation());
			glm::vec3 scale = transform->GetScale();

			if (ImGui::DragFloat3("Translation", &position.x, 0.1f))
				transform->SetPosition(position);
			if (ImGui::DragFloat3("Rotation", &rotation.x, 0.1f))
				transform->SetRotation(glm::quat(rotation));
			if (ImGui::DragFloat3("Scale", &scale.x, 0.1f, 0.01f, 100.0f))
				transform->SetScale(scale);

			glm::mat4 transformMatrix = transform->GetTransformMatrix();
			glm::mat4 gizmoProjection = projection;
			gizmoProjection[1][1] *= -1.0f;
			const bool manipulated = ImGuizmo::Manipulate(glm::value_ptr(view),
				glm::value_ptr(gizmoProjection),
				currentOperation,
				currentMode,
				glm::value_ptr(transformMatrix));

			if ((manipulated || ImGuizmo::IsUsing()) && currentOperation != static_cast<ImGuizmo::OPERATION>(-1))
			{
				glm::vec3 pos, rot, scale;
				ImGuizmo::DecomposeMatrixToComponents(glm::value_ptr(transformMatrix),
					glm::value_ptr(pos),
					glm::value_ptr(rot),
					glm::value_ptr(scale));

				switch (currentOperation)
				{
				case ImGuizmo::OPERATION::TRANSLATE:
					transform->SetPosition(pos);
					break;
				case ImGuizmo::OPERATION::ROTATE:
					transform->SetRotation(glm::quat(glm::radians(rot)));
					break;
				case ImGuizmo::OPERATION::SCALE:
					transform->SetScale(scale);
					break;
				default:
					break;
				}

				if (auto* assimp = registry.try_get<AssimpInstanceComponent>(mEnttSelectedEntity); assimp && assimp->instance)
				{
					InstanceSettings settings = assimp->instance->getInstanceSettings();
					settings.isWorldPosition = transform->GetPosition();
					settings.isWorldRotation = glm::degrees(glm::eulerAngles(transform->GetRotation()));
					settings.isScale = transform->GetScale().x;
					assimp->instance->setInstanceSettings(settings);
				}
			}
		}

		ImGui::Separator();
		ImGui::TextUnformatted("Components");
		const bool hasAnimationComp = registry.any_of<AnimationComponent>(mEnttSelectedEntity);
		if (!hasAnimationComp) {
			if (ImGui::Button("Add AnimationComponent")) {
				pushUndoSnapshot();
				registry.emplace_or_replace<AnimationComponent>(mEnttSelectedEntity);
				animation = registry.try_get<AnimationComponent>(mEnttSelectedEntity);
			}
		}
		else {
			if (ImGui::Button("Remove AnimationComponent")) {
				pushUndoSnapshot();
				registry.remove<AnimationComponent>(mEnttSelectedEntity);
				animation = nullptr;
			}
		}

		if (!registry.any_of<HierarchyComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add HierarchyComponent")) {
				pushUndoSnapshot();
				registry.emplace_or_replace<HierarchyComponent>(mEnttSelectedEntity);
			}
		}
		else {
			if (ImGui::Button("Remove HierarchyComponent")) {
				pushUndoSnapshot();
				detachHierarchy(mEnttSelectedEntity);
				registry.remove<HierarchyComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<PointLightComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add PointLightComponent")) {
				pushUndoSnapshot();
				auto& light = registry.emplace_or_replace<PointLightComponent>(mEnttSelectedEntity);
				light.color = glm::vec3(1.0f);
				light.intensity = 800.0f;
				light.range = 100.0f;
				light.enabled = true;
			}
		}
		else {
			if (ImGui::Button("Remove PointLightComponent")) {
				pushUndoSnapshot();
				registry.remove<PointLightComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<RigidBodyComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add RigidBodyComponent")) {
				pushUndoSnapshot();
				auto& rb = registry.emplace_or_replace<RigidBodyComponent>(mEnttSelectedEntity);
				rb.bodyType = RigidBodyType::Dynamic;
				rb.mass = 1.0f;
				rb.friction = 0.5f;
				rb.restitution = 0.2f;
				rb.useGravity = true;
				rb.linearVelocity = glm::vec3(0.0f);
				rb.registeredInWorld = false;
				rb.bodyId = -1;

				if (!registry.any_of<ColliderComponent>(mEnttSelectedEntity)) {
					registry.emplace_or_replace<ColliderComponent>(mEnttSelectedEntity);
				}

				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.registerEntity(entity, registry);
			}
		}
		else {
			if (ImGui::Button("Remove RigidBodyComponent")) {
				pushUndoSnapshot();
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				registry.remove<RigidBodyComponent>(mEnttSelectedEntity);
			}
		}

		if (!registry.any_of<ColliderComponent>(mEnttSelectedEntity)) {
			if (ImGui::Button("Add ColliderComponent")) {
				pushUndoSnapshot();
				auto& col = registry.emplace_or_replace<ColliderComponent>(mEnttSelectedEntity);
				col.shapeType = ColliderShapeType::Box;
				col.halfExtents = glm::vec3(0.5f);
				col.radius = 0.5f;
				col.halfHeight = 0.5f;

				if (registry.any_of<RigidBodyComponent>(mEnttSelectedEntity)) {
					entt::entity entity = mEnttSelectedEntity;
					physicsSystem.registerEntity(entity, registry);
				}
			}
		}
		else {
			if (ImGui::Button("Remove ColliderComponent")) {
				pushUndoSnapshot();
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				registry.remove<ColliderComponent>(mEnttSelectedEntity);
			}
		}

		if (registry.any_of<HierarchyComponent>(mEnttSelectedEntity)) {
			auto& hc = registry.get<HierarchyComponent>(mEnttSelectedEntity);
			std::vector<entt::entity> candidates{};
			std::vector<std::string> candidateLabels{};
			std::vector<const char*> candidateCStr{};
			candidates.push_back(entt::null);
			candidateLabels.emplace_back("<None>");
			for (auto [entity, entityTag] : registry.view<EnttTagComponent>().each()) {
				if (entity == mEnttSelectedEntity)
					continue;
				if (isDescendantOf(entity, mEnttSelectedEntity))
					continue;
				candidates.push_back(entity);
				candidateLabels.push_back(entityTag.name);
			}
			for (auto& label : candidateLabels)
				candidateCStr.push_back(label.c_str());

			int selectedParentIdx = 0;
			for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
				if (candidates[i] == hc.parent) {
					selectedParentIdx = i;
					break;
				}
			}

			if (ImGui::Combo("Parent", &selectedParentIdx, candidateCStr.data(), static_cast<int>(candidateCStr.size()))) {
				pushUndoSnapshot();
				setParent(mEnttSelectedEntity, candidates[selectedParentIdx]);
			}

			ImGui::Text("Children: %d", static_cast<int>(hc.children.size()));
			for (auto child : hc.children) {
				if (!registry.valid(child))
					continue;
				auto* childTag = registry.try_get<EnttTagComponent>(child);
				if (childTag) {
					ImGui::BulletText("%s", childTag->name.c_str());
				}
			}
		}

		if (auto* pointLight = registry.try_get<PointLightComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Point Light");
			ImGui::Checkbox("Enabled", &pointLight->enabled);
			ImGui::DragFloat3("Light Color", &pointLight->color.x, 0.01f, 0.0f, 1000.0f, "%.2f");
			ImGui::DragFloat("Intensity", &pointLight->intensity, 1.0f, 0.0f, 50000.0f, "%.1f");
			ImGui::DragFloat("Range", &pointLight->range, 0.5f, 0.0f, 1000.0f, "%.1f");
		}

		if (auto* rigidBody = registry.try_get<RigidBodyComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Rigid Body");

			bool rebuildBody = false;
			int bodyTypeIndex = static_cast<int>(rigidBody->bodyType);
			const char* bodyTypes[] = { "Static", "Dynamic", "Kinematic" };
			if (ImGui::Combo("Body Type", &bodyTypeIndex, bodyTypes, IM_ARRAYSIZE(bodyTypes))) {
				rigidBody->bodyType = static_cast<RigidBodyType>(bodyTypeIndex);
				rebuildBody = true;
			}

			if (ImGui::DragFloat("Mass", &rigidBody->mass, 0.05f, 0.001f, 10000.0f, "%.3f")) {
				rigidBody->mass = std::max(0.001f, rigidBody->mass);
				rebuildBody = true;
			}
			if (ImGui::SliderFloat("Friction", &rigidBody->friction, 0.0f, 1.0f, "%.3f")) {
				rebuildBody = true;
			}
			if (ImGui::SliderFloat("Restitution", &rigidBody->restitution, 0.0f, 1.0f, "%.3f")) {
				rebuildBody = true;
			}
			if (ImGui::Checkbox("Use Gravity", &rigidBody->useGravity)) {
				rebuildBody = true;
			}
			if (ImGui::DragFloat3("Linear Velocity", &rigidBody->linearVelocity.x, 0.05f, -500.0f, 500.0f, "%.3f")) {
				rebuildBody = true;
			}

			ImGui::Text("Body ID: %d", rigidBody->bodyId);
			ImGui::Text("Registered: %s", rigidBody->registeredInWorld ? "Yes" : "No");

			if (rebuildBody && registry.any_of<ColliderComponent>(mEnttSelectedEntity) && registry.any_of<TransformComponent>(mEnttSelectedEntity)) {
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				physicsSystem.registerEntity(entity, registry);
			}
		}

		if (auto* collider = registry.try_get<ColliderComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::TextUnformatted("Collider");

			bool rebuildBody = false;
			int shapeIndex = static_cast<int>(collider->shapeType);
			const char* shapeTypes[] = { "Box", "Sphere", "Capsule" };
			if (ImGui::Combo("Shape", &shapeIndex, shapeTypes, IM_ARRAYSIZE(shapeTypes))) {
				collider->shapeType = static_cast<ColliderShapeType>(shapeIndex);
				rebuildBody = true;
			}

			if (collider->shapeType == ColliderShapeType::Box) {
				if (ImGui::DragFloat3("Half Extents", &collider->halfExtents.x, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->halfExtents = glm::max(collider->halfExtents, glm::vec3(0.001f));
					rebuildBody = true;
				}
			}
			else if (collider->shapeType == ColliderShapeType::Sphere) {
				if (ImGui::DragFloat("Radius", &collider->radius, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->radius = std::max(0.001f, collider->radius);
					rebuildBody = true;
				}
			}
			else {
				if (ImGui::DragFloat("Radius", &collider->radius, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->radius = std::max(0.001f, collider->radius);
					rebuildBody = true;
				}
				if (ImGui::DragFloat("Half Height", &collider->halfHeight, 0.05f, 0.001f, 10000.0f, "%.3f")) {
					collider->halfHeight = std::max(0.001f, collider->halfHeight);
					rebuildBody = true;
				}
			}

			if (rebuildBody && registry.any_of<RigidBodyComponent>(mEnttSelectedEntity) && registry.any_of<TransformComponent>(mEnttSelectedEntity)) {
				entt::entity entity = mEnttSelectedEntity;
				physicsSystem.unregisterEntity(entity, registry);
				physicsSystem.registerEntity(entity, registry);
			}
		}

		// Assimp instance controls (add/remove instance for this entity)
		if (!registry.any_of<AssimpInstanceComponent>(mEnttSelectedEntity)) {
			ImGui::Separator();
			ImGui::Text("Assimp Instance:");
			if (mModelInstData.miModelList.empty()) {
				ImGui::Text("No models loaded. Import a model first.");
			}
			else {
				static int selModelIdx = 0;
				std::vector<const char*> modelNames;
				modelNames.reserve(mModelInstData.miModelList.size());
				for (auto& m : mModelInstData.miModelList) modelNames.push_back(m->getModelFileName().c_str());
				ImGui::PushItemWidth(200);
				if (ImGui::Combo("Model to Add", &selModelIdx, modelNames.data(), static_cast<int>(modelNames.size()))) {}
				ImGui::PopItemWidth();
				ImGui::SameLine();
				if (ImGui::Button("Add Instance")) {
					pushUndoSnapshot();
					auto model = mModelInstData.miModelList.at(selModelIdx);
					createAssimpInstanceForEntity(model, mEnttSelectedEntity);
				}
			}
		}
		else {
			ImGui::Separator();
			ImGui::Text("Assimp Instance Attached");
			auto& comp = registry.get<AssimpInstanceComponent>(mEnttSelectedEntity);
			if (comp.instance) {
				InstanceSettings instSet = comp.instance->getInstanceSettings();
				if (ImGui::DragFloat3("Instance Pos", glm::value_ptr(instSet.isWorldPosition), 0.1f)) {
					comp.instance->setTranslation(instSet.isWorldPosition);
				}
				glm::vec3 rot = instSet.isWorldRotation;
				if (ImGui::DragFloat3("Instance Rot", glm::value_ptr(rot), 1.0f)) {
					comp.instance->setRotation(rot);
				}
				if (ImGui::DragFloat("Instance Scale", &instSet.isScale, 0.01f, 0.001f, 100.0f)) {
					comp.instance->setScale(instSet.isScale);
				}

				auto model = comp.instance->getModel();
				if (animation && model) {
					auto animClips = model->getAnimClips();
					if (!animClips.empty()) {
						if (animation->clipIndex >= animClips.size()) {
							animation->clipIndex = static_cast<unsigned int>(animClips.size() - 1);
						}

						std::vector<std::string> clipNameStorage;
						clipNameStorage.reserve(animClips.size());
						for (const auto& clip : animClips) {
							clipNameStorage.push_back(clip->getClipName());
						}

						std::vector<const char*> clipNames;
						clipNames.reserve(clipNameStorage.size());
						for (const auto& clipName : clipNameStorage) {
							clipNames.push_back(clipName.c_str());
						}

						int clipIndex = static_cast<int>(animation->clipIndex);
						if (ImGui::Combo("Animation Clip", &clipIndex, clipNames.data(), static_cast<int>(clipNames.size()))) {
							animation->clipIndex = static_cast<unsigned int>(clipIndex);
							InstanceSettings updatedSettings = comp.instance->getInstanceSettings();
							updatedSettings.isAnimClipNr = animation->clipIndex;
							comp.instance->setInstanceSettings(updatedSettings);
						}

						float speed = animation->speed;
						if (ImGui::SliderFloat("Animation Speed", &speed, 0.0f, 2.0f, "%.3f")) {
							animation->speed = speed;
							InstanceSettings updatedSettings = comp.instance->getInstanceSettings();
							updatedSettings.isAnimSpeedFactor = animation->speed;
							comp.instance->setInstanceSettings(updatedSettings);
						}
					}
				}
				ImGui::SameLine();
				if (ImGui::Button("Remove Instance")) {
					pushUndoSnapshot();
					removeInstanceFromEntity(mEnttSelectedEntity);
				}
			}
		}

		if (ImGui::Button("Duplicate Entity"))
		{
			pushUndoSnapshot();
			std::vector<entt::entity> sources = mEnttMultiSelection.empty() ? std::vector<entt::entity>{mEnttSelectedEntity} : mEnttMultiSelection;
			mEnttMultiSelection.clear();
			for (auto src : sources) {
				entt::entity duplicatedEntity = duplicateEntity(src);
				if (duplicatedEntity != entt::null) {
					mEnttMultiSelection.push_back(duplicatedEntity);
					mEnttSelectedEntity = duplicatedEntity;
				}
			}
			if (!mEnttMultiSelection.empty()) {
				selectEntityAndSync(mEnttMultiSelection.back());
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Delete Entity"))
		{
			pushUndoSnapshot();
			std::vector<entt::entity> targets;
			if (!mEnttMultiSelection.empty() && isInMultiSelection(mEnttSelectedEntity)) {
				targets = mEnttMultiSelection;
			}
			else if (mEnttScene.isValid(mEnttSelectedEntity)) {
				targets.push_back(mEnttSelectedEntity);
			}

			for (auto entity : targets) {
				if (!mEnttScene.isValid(entity))
					continue;
				physicsSystem.unregisterEntity(entity, registry);
				detachHierarchy(entity);
				mEnttScene.destroyEntity(entity);
				removeFromMultiSelection(entity);
			}
			mEnttSelectedEntity = entt::null;
			mEnttMultiSelection.clear();
		}
	}
	else
	{
		ImGui::TextUnformatted("Select an entity from ECS Scene.");
	}
	ImGui::End();
}

std::string Renderer::serializeEnttScene() const
{
	const auto& registry = mEnttScene.getRegistry();
	json root;
	root["entities"] = json::array();
	root["selected"] = (mEnttSelectedEntity != entt::null) ? static_cast<uint32_t>(entt::to_integral(mEnttSelectedEntity)) : 0u;

	for (auto [entity, tag] : registry.view<EnttTagComponent>().each())
	{
		json node;
		node["id"] = static_cast<uint32_t>(entt::to_integral(entity));
		node["name"] = tag.name;

		if (const auto* transform = registry.try_get<TransformComponent>(entity)) {
			const auto& pos = transform->GetPosition();
			const auto& rot = transform->GetRotation();
			const auto& scale = transform->GetScale();
			node["transform"] = {
				{ "position", { pos.x, pos.y, pos.z } },
				{ "rotation", { rot.x, rot.y, rot.z, rot.w } },
				{ "scale", { scale.x, scale.y, scale.z } }
			};
		}

		if (const auto* animation = registry.try_get<AnimationComponent>(entity)) {
			node["animation"] = {
				{ "clipIndex", animation->clipIndex },
				{ "speed", animation->speed }
			};
		}

		if (const auto* assimpComp = registry.try_get<AssimpInstanceComponent>(entity); assimpComp && assimpComp->instance) {
			json assimp;
			auto model = assimpComp->instance->getModel();
			assimp["model"] = model ? model->getModelFileNamePath() : "";

			InstanceSettings settings = assimpComp->instance->getInstanceSettings();
			assimp["settings"] = {


				{ "position", { settings.isWorldPosition.x, settings.isWorldPosition.y, settings.isWorldPosition.z } },
				{ "rotation", { settings.isWorldRotation.x, settings.isWorldRotation.y, settings.isWorldRotation.z } },
				{ "scale", settings.isScale },
				{ "swapYZ", settings.isSwapYZAxis },
				{ "clipIndex", settings.isAnimClipNr },
				{ "playTime", settings.isAnimPlayTimePos },
				{ "speed", settings.isAnimSpeedFactor }
			};
			node["assimp"] = assimp;
		}

		if (const auto* renderableComp = registry.try_get<RenderableComponent>(entity)) {
			json renderable;
			renderable["modelPath"] = renderableComp->sourceModelFile;
			node["renderable"] = renderable;
		}

		if (const auto* hierarchy = registry.try_get<HierarchyComponent>(entity)) {
			node["hierarchy"] = {
				{ "parent", hierarchy->parent == entt::null ? -1 : static_cast<int64_t>(entt::to_integral(hierarchy->parent)) }
			};
		}

		if (const auto* pointLight = registry.try_get<PointLightComponent>(entity)) {
			node["pointLight"] = {
				{ "color", { pointLight->color.x, pointLight->color.y, pointLight->color.z } },
				{ "intensity", pointLight->intensity },
				{ "range", pointLight->range },
				{ "enabled", pointLight->enabled }
			};
		}

		if (const auto* rigidBody = registry.try_get<RigidBodyComponent>(entity)) {
			node["rigidBody"] = {
				{ "bodyType", static_cast<int>(rigidBody->bodyType) },
				{ "mass", rigidBody->mass },
				{ "friction", rigidBody->friction },
				{ "restitution", rigidBody->restitution },
				{ "useGravity", rigidBody->useGravity },
				{ "linearVelocity", { rigidBody->linearVelocity.x, rigidBody->linearVelocity.y, rigidBody->linearVelocity.z } }
			};
		}

		if (const auto* collider = registry.try_get<ColliderComponent>(entity)) {
			node["collider"] = {
				{ "shapeType", static_cast<int>(collider->shapeType) },
				{ "halfExtents", { collider->halfExtents.x, collider->halfExtents.y, collider->halfExtents.z } },
				{ "radius", collider->radius },
				{ "halfHeight", collider->halfHeight }
			};
		}

		root["entities"].push_back(node);
	}

	return root.dump(2);
}

std::shared_ptr<AssimpModel> Renderer::ensureModelLoadedForScene(const std::string& modelFileName)
{
	if (modelFileName.empty())
		return nullptr;

	if (auto existing = getModel(modelFileName)) {
		return existing;
	}

	std::shared_ptr<AssimpModel> model = std::make_shared<AssimpModel>();
	if (!model->loadModel(mRenderData, modelFileName)) {
		return nullptr;
	}

	mModelInstData.miModelList.emplace_back(model);
	return model;
}

bool Renderer::deserializeEnttScene(const std::string& sceneJson)
{
	json root = json::parse(sceneJson, nullptr, false);
	if (root.is_discarded() || !root.contains("entities") || !root["entities"].is_array()) {
		return false;
	}

	auto& registry = mEnttScene.getRegistry();
	std::vector<entt::entity> existingEntities;
	registry.view<EnttTagComponent>().each([&](entt::entity entity, EnttTagComponent&) {
		existingEntities.push_back(entity);
		});

	for (auto entity : existingEntities) {
		mEnttScene.destroyEntity(entity);
	}

	mEnttSelectedEntity = entt::null;
	mEnttMultiSelection.clear();

	std::unordered_map<uint32_t, entt::entity> entityMap;
	struct PendingHierarchy {
		entt::entity child = entt::null;
		int64_t parentId = -1;
	};
	std::vector<PendingHierarchy> pendingHierarchy;

	for (const auto& node : root["entities"]) {
		const uint32_t sourceId = node.value("id", 0u);
		const std::string name = node.value("name", std::string("Entity"));
		entt::entity entity = mEnttScene.createEntity(name);
		entityMap[sourceId] = entity;

		if (node.contains("transform")) {
			auto& transform = registry.emplace_or_replace<TransformComponent>(entity);
			const auto& t = node["transform"];
			if (t.contains("position") && t["position"].is_array() && t["position"].size() == 3) {
				transform.SetPosition(glm::vec3(t["position"][0].get<float>(), t["position"][1].get<float>(), t["position"][2].get<float>()));
			}
			if (t.contains("rotation") && t["rotation"].is_array() && t["rotation"].size() == 4) {
				transform.SetRotation(glm::quat(t["rotation"][3].get<float>(), t["rotation"][0].get<float>(), t["rotation"][1].get<float>(), t["rotation"][2].get<float>()));
			}
			if (t.contains("scale") && t["scale"].is_array() && t["scale"].size() == 3) {
				transform.SetScale(glm::vec3(t["scale"][0].get<float>(), t["scale"][1].get<float>(), t["scale"][2].get<float>()));
			}
		}

		if (node.contains("animation")) {
			auto& anim = registry.emplace_or_replace<AnimationComponent>(entity);
			anim.clipIndex = node["animation"].value("clipIndex", 0u);
			anim.speed = node["animation"].value("speed", 1.0f);
		}

		if (node.contains("renderable") && node["renderable"].is_object()) {
			const auto& renderableNode = node["renderable"];
			const std::string modelPath = renderableNode.value("modelPath", std::string());
			if (!modelPath.empty()) {
				auto& renderableComp = registry.emplace_or_replace<RenderableComponent>(entity);
				Model::loadModel(modelPath, renderableComp);
				renderableComp.sourceModelFile = modelPath;
			}
		}

		if (node.contains("assimp") && node["assimp"].is_object()) {
			const auto& assimpNode = node["assimp"];
			const std::string modelName = assimpNode.value("model", std::string());
			auto model = ensureModelLoadedForScene(modelName);
			if (model) {
				createAssimpInstanceForEntity(model, entity);
				auto* assimpComp = registry.try_get<AssimpInstanceComponent>(entity);
				if (assimpComp && assimpComp->instance && assimpNode.contains("settings")) {
					InstanceSettings settings = assimpComp->instance->getInstanceSettings();
					const auto& s = assimpNode["settings"];
					if (s.contains("position") && s["position"].is_array() && s["position"].size() == 3) {
						settings.isWorldPosition = glm::vec3(s["position"][0].get<float>(), s["position"][1].get<float>(), s["position"][2].get<float>());
					}
					if (s.contains("rotation") && s["rotation"].is_array() && s["rotation"].size() == 3) {
						settings.isWorldRotation = glm::vec3(s["rotation"][0].get<float>(), s["rotation"][1].get<float>(), s["rotation"][2].get<float>());
					}
					settings.isScale = s.value("scale", settings.isScale);
					settings.isSwapYZAxis = s.value("swapYZ", settings.isSwapYZAxis);
					settings.isAnimClipNr = s.value("clipIndex", settings.isAnimClipNr);
					settings.isAnimPlayTimePos = s.value("playTime", settings.isAnimPlayTimePos);
					settings.isAnimSpeedFactor = s.value("speed", settings.isAnimSpeedFactor);
					assimpComp->instance->setInstanceSettings(settings);

					auto& anim = registry.emplace_or_replace<AnimationComponent>(entity);
					anim.clipIndex = settings.isAnimClipNr;
					anim.speed = settings.isAnimSpeedFactor;

					auto& transform = registry.emplace_or_replace<TransformComponent>(entity);
					transform.SetPosition(settings.isWorldPosition);
					transform.SetRotation(glm::quat(glm::radians(settings.isWorldRotation)));
					transform.SetScale(glm::vec3(settings.isScale));
				}
			}
		}

		if (node.contains("hierarchy") && node["hierarchy"].is_object()) {
			const int64_t parentId = node["hierarchy"].value("parent", -1ll);
			registry.emplace_or_replace<HierarchyComponent>(entity);
			pendingHierarchy.push_back({ entity, parentId });
		}

		if (node.contains("pointLight") && node["pointLight"].is_object()) {
			auto& light = registry.emplace_or_replace<PointLightComponent>(entity);
			const auto& pl = node["pointLight"];
			if (pl.contains("color") && pl["color"].is_array() && pl["color"].size() == 3) {
				light.color = glm::vec3(pl["color"][0].get<float>(), pl["color"][1].get<float>(), pl["color"][2].get<float>());
			}
			light.intensity = pl.value("intensity", light.intensity);
			light.range = pl.value("range", light.range);
			light.enabled = pl.value("enabled", light.enabled);
		}

		if (node.contains("rigidBody") && node["rigidBody"].is_object()) {
			auto& rb = registry.emplace_or_replace<RigidBodyComponent>(entity);
			const auto& rbNode = node["rigidBody"];
			rb.bodyType = static_cast<RigidBodyType>(rbNode.value("bodyType", static_cast<int>(rb.bodyType)));
			rb.mass = rbNode.value("mass", rb.mass);
			rb.friction = rbNode.value("friction", rb.friction);
			rb.restitution = rbNode.value("restitution", rb.restitution);
			rb.useGravity = rbNode.value("useGravity", rb.useGravity);
			if (rbNode.contains("linearVelocity") && rbNode["linearVelocity"].is_array() && rbNode["linearVelocity"].size() == 3) {
				rb.linearVelocity = glm::vec3(rbNode["linearVelocity"][0].get<float>(), rbNode["linearVelocity"][1].get<float>(), rbNode["linearVelocity"][2].get<float>());
			}
			rb.registeredInWorld = false;
			rb.bodyId = -1;
		}

		if (node.contains("collider") && node["collider"].is_object()) {
			auto& col = registry.emplace_or_replace<ColliderComponent>(entity);
			const auto& colNode = node["collider"];
			col.shapeType = static_cast<ColliderShapeType>(colNode.value("shapeType", static_cast<int>(col.shapeType)));
			if (colNode.contains("halfExtents") && colNode["halfExtents"].is_array() && colNode["halfExtents"].size() == 3) {
				col.halfExtents = glm::vec3(colNode["halfExtents"][0].get<float>(), colNode["halfExtents"][1].get<float>(), colNode["halfExtents"][2].get<float>());
			}
			col.radius = colNode.value("radius", col.radius);
			col.halfHeight = colNode.value("halfHeight", col.halfHeight);
		}
	}

	for (const auto& entry : pendingHierarchy) {
		if (!registry.valid(entry.child))
			continue;
		auto& childHierarchy = registry.emplace_or_replace<HierarchyComponent>(entry.child);
		childHierarchy.parent = entt::null;
		if (entry.parentId < 0)
			continue;
		auto it = entityMap.find(static_cast<uint32_t>(entry.parentId));
		if (it == entityMap.end())
			continue;
		entt::entity parent = it->second;
		if (!registry.valid(parent) || parent == entry.child)
			continue;

		childHierarchy.parent = parent;
		auto& parentHierarchy = registry.emplace_or_replace<HierarchyComponent>(parent);
		if (std::find(parentHierarchy.children.begin(), parentHierarchy.children.end(), entry.child) == parentHierarchy.children.end()) {
			parentHierarchy.children.push_back(entry.child);
		}
	}

	const uint32_t selectedId = root.value("selected", 0u);
	auto selectedIt = entityMap.find(selectedId);
	if (selectedIt != entityMap.end() && mEnttScene.isValid(selectedIt->second)) {
		mEnttSelectedEntity = selectedIt->second;
		mEnttMultiSelection.push_back(mEnttSelectedEntity);
	}

	for (auto [entity, rb, col, tr] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each()) {
		(void)rb;
		(void)col;
		(void)tr;
		entt::entity e = entity;
		physicsSystem.registerEntity(e, registry);
	}

	rebuildRenderableRuntimeResources();

	return true;
}

void Renderer::pushUndoSnapshot()
{
	if (mHistoryMuted)
		return;

	mUndoSnapshots.push_back(serializeEnttScene());
	if (mUndoSnapshots.size() > 100) {
		mUndoSnapshots.erase(mUndoSnapshots.begin());
	}
	mRedoSnapshots.clear();
}

void Renderer::performUndo()
{
	if (mUndoSnapshots.empty())
		return;

	mRedoSnapshots.push_back(serializeEnttScene());
	std::string targetSnapshot = mUndoSnapshots.back();
	mUndoSnapshots.pop_back();

	const bool prevMute = mHistoryMuted;
	mHistoryMuted = true;
	deserializeEnttScene(targetSnapshot);
	mHistoryMuted = prevMute;
}

void Renderer::performRedo()
{
	if (mRedoSnapshots.empty())
		return;

	mUndoSnapshots.push_back(serializeEnttScene());
	std::string targetSnapshot = mRedoSnapshots.back();
	mRedoSnapshots.pop_back();

	const bool prevMute = mHistoryMuted;
	mHistoryMuted = true;
	deserializeEnttScene(targetSnapshot);
	mHistoryMuted = prevMute;
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
		vk::PhysicalDeviceVulkan11Features,
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

	Image::createImage(device, physicalDevice,
		swapChainExtent.width, swapChainExtent.height,
		vk::Format::eR16G16B16A16Sfloat,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		fxaaImage, fxaaImageMemory);
	fxaaImageView = ImageView::createImageView(device, fxaaImage, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor);

	createBloomResources();
	createBloomDescriptorSets();
	Image::createImage(device, physicalDevice,
		swapChainExtent.width, swapChainExtent.height,
		swapChainSurfaceFormat.format,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		viewportPreviewImage, viewportPreviewImageMemory);
	viewportPreviewImageView = ImageView::createImageView(device, viewportPreviewImage, swapChainSurfaceFormat.format, vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eUndefined;

	fxaaDescriptorSets.clear();
	fxaaDescriptorPool = nullptr;
	DescriptorPool::createFxaaDescriptorPool(device, fxaaDescriptorPool, MAX_FRAMES_IN_FLIGHT);
	DescriptorSet::createFxaaDescriptorSets(device, fxaaDescriptorPool, fxaaDescriptorSetLayout, fxaaImageView, bloomImageAView, depthImageView, fxaaSampler, MAX_FRAMES_IN_FLIGHT, fxaaDescriptorSets);

	ImGuiIO& io = ImGui::GetIO();
	io.DisplaySize = ImVec2(static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height));

	imGui->clearRegisteredTextures();
	mViewportTextureId = imGui->registerTexture(*viewportPreviewImageView, *fxaaSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Renderer::createGraphicsPipeline()
{
	auto bindingDescription = Vertex::getBindingDescription();
	auto attributeDescriptions = Vertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "shaders\\slang.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "shaders\\slang.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
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
			{ "shaders\\pbr.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
			{ "shaders\\pbr.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
		};

		config.vertexBindings = { bindingDescription };
		config.vertexAttributes = {
			attributeDescriptions.begin(),
			attributeDescriptions.end()
		};

		config.descriptorSetLayouts = { *descriptorSetLayout };
		config.pushConstantRanges = { pushConstantRange };
		config.colorAttachmentFormats = { vk::Format::eR16G16B16A16Sfloat };
		config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);
		config.depthTestEnable = true;
		config.depthWriteEnable = true;
		config.depthCompareOp = vk::CompareOp::eLess;

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

	if (renderEnableShadows)
	{
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
			recordAssimpShadowPass(commandBuffer, cascade);
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
	}

	beginMainPass(commandBuffer, imageIndex);
	recordScenePass(commandBuffer);
	commandBuffer.endRendering();

	if (renderEnablePostProcessing)
	{
		if (renderEnableBloom && bloomEnabled)
			recordBloomPasses(commandBuffer);

		if (renderEnableFxaa)
			recordFxaaPass(commandBuffer, imageIndex);
		else
			recordSceneCopyPass(commandBuffer, imageIndex);
	}
	else
	{
		recordSceneCopyPass(commandBuffer, imageIndex);
	}

	recordImguiPass(commandBuffer, imageIndex);

	endMainPass(commandBuffer, imageIndex);

	commandBuffer.end();
}

void Renderer::beginMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{

	transition_image_layout(*fxaaImage,
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
		.imageView = fxaaImageView,
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

	auto& registry = mEnttScene.getRegistry();
	std::array<glm::vec4, 4> pointLightPositions = {
		glm::vec4(0.0f, -45.0f, 0.0f, 1.0f),
		glm::vec4(-70.0f, -80.0f, 5.0f, 1.0f),
		glm::vec4(10.0f, -50.0f, -75.0f, 1.0f),
		glm::vec4(20.0f, 40.0f, -10.0f, 1.0f)
	};
	std::array<glm::vec4, 4> pointLightColors = {
		glm::vec4(1000.0f, 1000.0f, 1000.0f, 1.0f),
		glm::vec4(800.0f, 200.0f, 200.0f, 1.0f),
		glm::vec4(200.0f, 200.0f, 800.0f, 1.0f),
		glm::vec4(200.0f, 800.0f, 200.0f, 1.0f)
	};
	int lightIndex = 0;
	for (auto [lightEntity, light, lightTransform] : registry.view<PointLightComponent, TransformComponent>().each())
	{
		(void)lightEntity;
		if (lightIndex >= 4)
			break;
		if (!light.enabled)
			continue;

		pointLightPositions[lightIndex] = glm::vec4(lightTransform.GetPosition(), 1.0f);
		pointLightColors[lightIndex] = glm::vec4(light.color * light.intensity, 1.0f);
		++lightIndex;
	}
	for (; lightIndex < 4; ++lightIndex) {
		pointLightColors[lightIndex] = glm::vec4(0.0f);
	}

	for (auto [entity, renderable, transform] : registry.view<RenderableComponent, TransformComponent>().each())
	{
		(void)entity;

		UniformBuffer::updateUniformBuffer(frameIndex, &renderable, &transform, &camera, swapChainExtent, shadowSettings, pointLightPositions, pointLightColors);
		vk::Buffer     vertexBuffers[] = { renderable.vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*renderable.indexBuffer, 0, vk::IndexType::eUint32);

		for (const auto& mesh : renderable.meshes)
		{
			uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable.materialDescriptorSets.size() ? mesh.materialIndex : 0;

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*shadowPipelineLayout,
				0,
				*renderable.materialDescriptorSets[descriptorMaterialIndex][frameIndex],
				nullptr);

			commandBuffer.drawIndexed(mesh.indexCount, 1, mesh.firstIndex, 0, 0);
		}
	}

}

void Renderer::recordAssimpShadowPass(vk::raii::CommandBuffer& commandBuffer, uint32_t cascadeIndex)
{
	if (mAssimpGPUData.empty() || mModelInstData.miAssimpInstances.empty() || *shadowSkinningPipeline == VK_NULL_HANDLE)
		return;

	const UniformBufferObject* shadowTemplateUbo = nullptr;
	auto& registry = mEnttScene.getRegistry();
	for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
		(void)entity;
		if (renderable.uniformBuffersMapped.size() <= frameIndex)
			continue;
		if (!renderable.uniformBuffersMapped[frameIndex])
			continue;

		shadowTemplateUbo = reinterpret_cast<const UniformBufferObject*>(renderable.uniformBuffersMapped[frameIndex]);
		break;
	}

	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *shadowSkinningPipeline);

	int cascadeIndexInt = static_cast<int>(cascadeIndex);
	commandBuffer.pushConstants<int>(*shadowSkinningPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, cascadeIndexInt);

	for (auto& gpuData : mAssimpGPUData)
	{
		UniformBufferObject ubo{};
		if (shadowTemplateUbo)
			ubo = *shadowTemplateUbo;
		ubo.model = gpuData.instance->getWorldTransformMatrix();
		memcpy(gpuData.uboMapped[frameIndex], &ubo, sizeof(UniformBufferObject));

		auto& model = *gpuData.instance->getModel();
		const auto& meshes = model.getModelMeshes();
		const auto& vbos = model.getVertexBuffers();
		const auto& ibos = model.getIndexBuffers();

		for (size_t i = 0; i < meshes.size(); ++i)
		{
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				*shadowSkinningPipelineLayout, 0,
				*gpuData.descriptorSets[frameIndex][i], nullptr);

			vk::Buffer vb = vbos[i].buffer;
			vk::DeviceSize off = 0;
			commandBuffer.bindVertexBuffers(0, vb, off);
			commandBuffer.bindIndexBuffer(ibos[i].buffer, 0, vk::IndexType::eUint32);
			commandBuffer.drawIndexed(static_cast<uint32_t>(meshes[i].indices.size()), 1, 0, 0, 0);
		}
	}
}

void Renderer::createFxaaSampler()
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
	fxaaSampler = vk::raii::Sampler(device, samplerInfo);
}

void Renderer::createBloomResources()
{
	bloomExtent = {
		  .width = std::max(1u, swapChainExtent.width / 2),
		  .height = std::max(1u, swapChainExtent.height / 2)
	};

	Image::createImage(device, physicalDevice,
		bloomExtent.width, bloomExtent.height,
		bloomFormat,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		bloomImageA, bloomImageAMemory);
	bloomImageAView = ImageView::createImageView(device, bloomImageA, bloomFormat, vk::ImageAspectFlagBits::eColor);

	Image::createImage(device, physicalDevice,
		bloomExtent.width, bloomExtent.height,
		bloomFormat,
		vk::ImageTiling::eOptimal,
		vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
		vk::MemoryPropertyFlagBits::eDeviceLocal,
		bloomImageB, bloomImageBMemory);
	bloomImageBView = ImageView::createImageView(device, bloomImageB, bloomFormat, vk::ImageAspectFlagBits::eColor);
}

void Renderer::createBloomDescriptorSets()
{
	std::array<vk::DescriptorSetLayoutBinding, 1> binding{ {
		{.binding = 0, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eFragment }
	} };
	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = static_cast<uint32_t>(binding.size()),
		.pBindings = binding.data()
	};
	bloomExtractDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	bloomBlurDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);

	std::array<vk::DescriptorPoolSize, 1> poolSizes{ {
		{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = 3 * MAX_FRAMES_IN_FLIGHT }
	} };
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 3 * MAX_FRAMES_IN_FLIGHT,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};
	bloomDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

	std::vector<vk::DescriptorSetLayout> extractLayouts(MAX_FRAMES_IN_FLIGHT, *bloomExtractDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo extractAllocInfo{
		.descriptorPool = *bloomDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(extractLayouts.size()),
		.pSetLayouts = extractLayouts.data()
	};
	bloomExtractDescriptorSets = device.allocateDescriptorSets(extractAllocInfo);

	std::vector<vk::DescriptorSetLayout> blurLayouts(MAX_FRAMES_IN_FLIGHT, *bloomBlurDescriptorSetLayout);
	vk::DescriptorSetAllocateInfo blurAllocInfo{
		.descriptorPool = *bloomDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(blurLayouts.size()),
		.pSetLayouts = blurLayouts.data()
	};
	bloomBlurFromADescriptorSets = device.allocateDescriptorSets(blurAllocInfo);
	bloomBlurFromBDescriptorSets = device.allocateDescriptorSets(blurAllocInfo);

	for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
	{
		vk::DescriptorImageInfo extractInfo{ .sampler = *fxaaSampler, .imageView = *fxaaImageView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::WriteDescriptorSet extractWrite{
			.dstSet = *bloomExtractDescriptorSets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eCombinedImageSampler,
			.pImageInfo = &extractInfo
		};
		device.updateDescriptorSets(extractWrite, nullptr);

		vk::DescriptorImageInfo blurFromAInfo{ .sampler = *fxaaSampler, .imageView = *bloomImageAView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::WriteDescriptorSet blurAWrite{
			.dstSet = *bloomBlurFromADescriptorSets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eCombinedImageSampler,
			.pImageInfo = &blurFromAInfo
		};
		device.updateDescriptorSets(blurAWrite, nullptr);

		vk::DescriptorImageInfo blurFromBInfo{ .sampler = *fxaaSampler, .imageView = *bloomImageBView, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::WriteDescriptorSet blurBWrite{
			.dstSet = *bloomBlurFromBDescriptorSets[i],
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = vk::DescriptorType::eCombinedImageSampler,
			.pImageInfo = &blurFromBInfo
		};
		device.updateDescriptorSets(blurBWrite, nullptr);
	}
}

void Renderer::createBloomPipelines()
{
	vk::PushConstantRange extractPushRange{
		.stageFlags = vk::ShaderStageFlagBits::eFragment,
		.offset = 0,
		.size = sizeof(BloomExtractPushConstantsCPU)
	};
	Pipeline::PipelineConfig extractConfig{};
	extractConfig.shaderStages = {
		{ "shaders\\bloom_extract.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "shaders\\bloom_extract.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};
	extractConfig.descriptorSetLayouts = { *bloomExtractDescriptorSetLayout };
	extractConfig.pushConstantRanges = { extractPushRange };
	extractConfig.colorAttachmentFormats = { bloomFormat };
	extractConfig.depthTestEnable = false;
	extractConfig.depthWriteEnable = false;
	extractConfig.blendEnable = false;
	extractConfig.cullMode = vk::CullModeFlagBits::eNone;
	auto extractBundle = Pipeline::createPipeline(device, extractConfig);
	bloomExtractPipelineLayout = std::move(extractBundle.layout);
	bloomExtractPipeline = std::move(extractBundle.pipeline);

	vk::PushConstantRange blurPushRange{
		.stageFlags = vk::ShaderStageFlagBits::eFragment,
		.offset = 0,
		.size = sizeof(BloomBlurPushConstantsCPU)
	};
	Pipeline::PipelineConfig blurConfig{};
	blurConfig.shaderStages = {
		{ "shaders\\bloom_blur.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "shaders\\bloom_blur.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};
	blurConfig.descriptorSetLayouts = { *bloomBlurDescriptorSetLayout };
	blurConfig.pushConstantRanges = { blurPushRange };
	blurConfig.colorAttachmentFormats = { bloomFormat };
	blurConfig.depthTestEnable = false;
	blurConfig.depthWriteEnable = false;
	blurConfig.blendEnable = false;
	blurConfig.cullMode = vk::CullModeFlagBits::eNone;
	auto blurBundle = Pipeline::createPipeline(device, blurConfig);
	bloomBlurPipelineLayout = std::move(blurBundle.layout);
	bloomBlurPipeline = std::move(blurBundle.pipeline);
}

void Renderer::recordBloomPasses(vk::raii::CommandBuffer& commandBuffer)
{
	if (!bloomEnabled)
		return;

	transition_image_layout(*fxaaImage,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eColor);

	transition_image_layout(*bloomImageA,
		bloomImageALayout,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);
	bloomImageALayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::RenderingAttachmentInfo bloomAttachmentA{
		.imageView = bloomImageAView,
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
	};

	vk::RenderingInfo bloomRenderInfoA{
	 .renderArea = {.offset = { 0, 0 }, .extent = bloomExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &bloomAttachmentA
	};

	vk::RenderingInfo bloomRenderInfoB{
		.renderArea = {.offset = { 0, 0 }, .extent = bloomExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = nullptr
	};
	commandBuffer.beginRendering(bloomRenderInfoA);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomExtractPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
		static_cast<float>(bloomExtent.width),
		static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		*bloomExtractPipelineLayout, 0,
		*bloomExtractDescriptorSets[frameIndex], nullptr);
	BloomExtractPushConstantsCPU extractPc{
		  .threshold = bloomThreshold,
		  .softKnee = bloomSoftKnee,
		  .prefilterScale = bloomPrefilterScale
	};
	commandBuffer.pushConstants<BloomExtractPushConstantsCPU>(
		*bloomExtractPipelineLayout,
		vk::ShaderStageFlagBits::eFragment,
		0,
		extractPc);
	commandBuffer.draw(3, 1, 0, 0);
	commandBuffer.endRendering();

	transition_image_layout(*bloomImageA,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eColor);
	bloomImageALayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	transition_image_layout(*bloomImageB,
		bloomImageBLayout,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);
	bloomImageBLayout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::RenderingAttachmentInfo bloomAttachmentB{
		 .imageView = bloomImageBView,
		 .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		 .loadOp = vk::AttachmentLoadOp::eClear,
		 .storeOp = vk::AttachmentStoreOp::eStore,
		 .clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
	};
	bloomRenderInfoB.pColorAttachments = &bloomAttachmentB;

	const int blurPassPairs = std::max(1, bloomBlurPasses);
	for (int i = 0; i < blurPassPairs; ++i)
	{
		if (i > 0)
		{
			transition_image_layout(*bloomImageB,
				vk::ImageLayout::eShaderReadOnlyOptimal,
				vk::ImageLayout::eColorAttachmentOptimal,
				vk::AccessFlagBits2::eShaderSampledRead,
				vk::AccessFlagBits2::eColorAttachmentWrite,
				vk::PipelineStageFlagBits2::eFragmentShader,
				vk::PipelineStageFlagBits2::eColorAttachmentOutput,
				vk::ImageAspectFlagBits::eColor);
			bloomImageBLayout = vk::ImageLayout::eColorAttachmentOptimal;
		}

		commandBuffer.beginRendering(bloomRenderInfoB);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomBlurPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
			static_cast<float>(bloomExtent.width),
			static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
			*bloomBlurPipelineLayout, 0,
			*bloomBlurFromADescriptorSets[frameIndex], nullptr);
		BloomBlurPushConstantsCPU blurPcX{
			.direction = {
				1.0f / static_cast<float>(bloomExtent.width),
				0.0f
			},
			.blurScale = bloomBlurScale
		};
		commandBuffer.pushConstants<BloomBlurPushConstantsCPU>(
			*bloomBlurPipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			blurPcX);
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRendering();

		transition_image_layout(*bloomImageB,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::AccessFlagBits2::eShaderSampledRead,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eColor);
		bloomImageBLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

		transition_image_layout(*bloomImageA,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::AccessFlagBits2::eShaderSampledRead,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::ImageAspectFlagBits::eColor);
		bloomImageALayout = vk::ImageLayout::eColorAttachmentOptimal;

		commandBuffer.beginRendering(bloomRenderInfoA);
		commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *bloomBlurPipeline);
		commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
			static_cast<float>(bloomExtent.width),
			static_cast<float>(bloomExtent.height), 0.0f, 1.0f));
		commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, bloomExtent));
		commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
			*bloomBlurPipelineLayout, 0,
			*bloomBlurFromBDescriptorSets[frameIndex], nullptr);
		BloomBlurPushConstantsCPU blurPcY{
			.direction = {
				0.0f,
				1.0f / static_cast<float>(bloomExtent.height)
			},
			.blurScale = bloomBlurScale
		};
		commandBuffer.pushConstants<BloomBlurPushConstantsCPU>(
			*bloomBlurPipelineLayout,
			vk::ShaderStageFlagBits::eFragment,
			0,
			blurPcY);
		commandBuffer.draw(3, 1, 0, 0);
		commandBuffer.endRendering();

		transition_image_layout(*bloomImageA,
			vk::ImageLayout::eColorAttachmentOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			vk::AccessFlagBits2::eColorAttachmentWrite,
			vk::AccessFlagBits2::eShaderSampledRead,
			vk::PipelineStageFlagBits2::eColorAttachmentOutput,
			vk::PipelineStageFlagBits2::eFragmentShader,
			vk::ImageAspectFlagBits::eColor);
		bloomImageALayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
}

void Renderer::recordFxaaPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	transition_image_layout(*depthImage,
		vk::ImageLayout::eDepthAttachmentOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eLateFragmentTests,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eDepth);

	// Transition swapchain image: undefined → color attachment for FXAA output
	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);

	vk::RenderingAttachmentInfo colorAttachment{
		.imageView = swapChainImageViews[imageIndex],
		.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		.loadOp = vk::AttachmentLoadOp::eClear,
		.storeOp = vk::AttachmentStoreOp::eStore,
		.clearValue = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f)
	};

	vk::RenderingInfo renderingInfo{
		.renderArea = {.offset = { 0, 0 }, .extent = swapChainExtent },
		.layerCount = 1,
		.colorAttachmentCount = 1,
		.pColorAttachments = &colorAttachment
	};

	commandBuffer.beginRendering(renderingInfo);
	commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *fxaaPipeline);
	commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f,
		static_cast<float>(swapChainExtent.width),
		static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
	commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, swapChainExtent));

	commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
		*fxaaPipelineLayout, 0,
		*fxaaDescriptorSets[frameIndex], nullptr);

	// Push rcpFrame so the FXAA shader knows the texel size
	FxaaPushConstantsCPU pc{
		.rcpFrame = {
			1.0f / static_cast<float>(swapChainExtent.width),
			1.0f / static_cast<float>(swapChainExtent.height)
		},
		.exposure = fxaaExposure,
	  .gamma = fxaaGamma,
	  .bloomIntensity = bloomEnabled ? bloomIntensity : 0.0f,
		.debugMode = postProcessDebugMode
	};

	commandBuffer.pushConstants<FxaaPushConstantsCPU>(
		*fxaaPipelineLayout,
		vk::ShaderStageFlagBits::eFragment,
		0,
		pc
	);

	// 3 vertices, no vertex buffer — the VS generates the fullscreen triangle
	commandBuffer.draw(3, 1, 0, 0);
	commandBuffer.endRendering();

	transition_image_layout(*viewportPreviewImage,
		viewportPreviewImageLayout,
		vk::ImageLayout::eTransferDstOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eTransferDstOptimal;

	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eTransferSrcOptimal,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::AccessFlagBits2::eTransferRead,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::ImageAspectFlagBits::eColor);

	vk::ImageCopy copyRegion{
		.srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.srcOffset = { 0, 0, 0 },
		.dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.dstOffset = { 0, 0, 0 },
		.extent = { swapChainExtent.width, swapChainExtent.height, 1 }
	};
	commandBuffer.copyImage(swapChainImages[imageIndex], vk::ImageLayout::eTransferSrcOptimal,
		*viewportPreviewImage, vk::ImageLayout::eTransferDstOptimal,
		copyRegion);

	transition_image_layout(*viewportPreviewImage,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eTransferWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eTransferSrcOptimal,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlagBits2::eTransferRead,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);

}

void Renderer::recordSceneCopyPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex)
{
	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eUndefined,
		vk::ImageLayout::eTransferDstOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::ImageAspectFlagBits::eColor);

	transition_image_layout(*fxaaImage,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eTransferSrcOptimal,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::AccessFlagBits2::eTransferRead,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::ImageAspectFlagBits::eColor);

	vk::ImageCopy copyRegion{
		.srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.srcOffset = { 0, 0, 0 },
		.dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
		.dstOffset = { 0, 0, 0 },
		.extent = { swapChainExtent.width, swapChainExtent.height, 1 }
	};
	commandBuffer.copyImage(*fxaaImage, vk::ImageLayout::eTransferSrcOptimal,
		swapChainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
		copyRegion);

	transition_image_layout(*viewportPreviewImage,
		viewportPreviewImageLayout,
		vk::ImageLayout::eTransferDstOptimal,
		vk::AccessFlags2{},
		vk::AccessFlagBits2::eTransferWrite,
		vk::PipelineStageFlagBits2::eTopOfPipe,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eTransferDstOptimal;

	commandBuffer.copyImage(*fxaaImage, vk::ImageLayout::eTransferSrcOptimal,
		*viewportPreviewImage, vk::ImageLayout::eTransferDstOptimal,
		copyRegion);

	transition_image_layout(*viewportPreviewImage,
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eShaderReadOnlyOptimal,
		vk::AccessFlagBits2::eTransferWrite,
		vk::AccessFlagBits2::eShaderSampledRead,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::PipelineStageFlagBits2::eFragmentShader,
		vk::ImageAspectFlagBits::eColor);
	viewportPreviewImageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	transition_image_layout(*fxaaImage,
		vk::ImageLayout::eTransferSrcOptimal,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlagBits2::eTransferRead,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);

	transition_image_layout(swapChainImages[imageIndex],
		vk::ImageLayout::eTransferDstOptimal,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::AccessFlagBits2::eTransferWrite,
		vk::AccessFlagBits2::eColorAttachmentWrite,
		vk::PipelineStageFlagBits2::eTransfer,
		vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		vk::ImageAspectFlagBits::eColor);
}

void Renderer::recordScenePass(vk::raii::CommandBuffer& commandBuffer)
{
	auto& registry = mEnttScene.getRegistry();
	std::array<glm::vec4, 4> pointLightPositions = {
		   glm::vec4(0.0f, -45.0f, 0.0f, 1.0f),
		   glm::vec4(-70.0f, -80.0f, 5.0f, 1.0f),
		   glm::vec4(10.0f, -50.0f, -75.0f, 1.0f),
		   glm::vec4(20.0f, 40.0f, -10.0f, 1.0f)
	};
	std::array<glm::vec4, 4> pointLightColors = {
		glm::vec4(1000.0f, 1000.0f, 1000.0f, 1.0f),
		glm::vec4(800.0f, 200.0f, 200.0f, 1.0f),
		glm::vec4(200.0f, 200.0f, 800.0f, 1.0f),
		glm::vec4(200.0f, 800.0f, 200.0f, 1.0f)
	};

	int lightIndex = 0;
	for (auto [lightEntity, light, lightTransform] : registry.view<PointLightComponent, TransformComponent>().each())
	{
		(void)lightEntity;
		if (lightIndex >= 4)
			break;
		if (!light.enabled)
			continue;

		pointLightPositions[lightIndex] = glm::vec4(lightTransform.GetPosition(), 1.0f);
		pointLightColors[lightIndex] = glm::vec4(light.color * light.intensity, 1.0f);
		++lightIndex;
	}

	for (; lightIndex < 4; ++lightIndex) {
		pointLightColors[lightIndex] = glm::vec4(0.0f);
	}

	for (auto [ecsEntity, renderable, transform] : registry.view<RenderableComponent, TransformComponent>().each())
	{
		(void)ecsEntity;

		UniformBuffer::updateUniformBuffer(frameIndex, &renderable, &transform, &camera, swapChainExtent, shadowSettings, pointLightPositions, pointLightColors);
		vk::Buffer     vertexBuffers[] = { renderable.vertexBuffer };
		vk::DeviceSize offsets[] = { 0 };
		commandBuffer.bindVertexBuffers(0, vertexBuffers, offsets);
		commandBuffer.bindIndexBuffer(*renderable.indexBuffer, 0, vk::IndexType::eUint32);

		for (const auto& mesh : renderable.meshes)
		{
			const Material& material = renderable.materials[mesh.materialIndex < renderable.materials.size() ? mesh.materialIndex : 0];
			MaterialPushConstants::push(commandBuffer, *pbrPipelineLayout, material);
			uint32_t descriptorMaterialIndex = mesh.materialIndex < renderable.materialDescriptorSets.size() ? mesh.materialIndex : 0;

			commandBuffer.bindDescriptorSets(
				vk::PipelineBindPoint::eGraphics,
				*pbrPipelineLayout,
				0,
				*renderable.materialDescriptorSets[descriptorMaterialIndex][frameIndex],
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
			defaultTextureImage, defaultTextureMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuf, defaultTextureImage, 1, 1);
		Image::transitionImageLayout(commandBuffer, defaultTextureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
		CommandBuffer::endSingleTimeCommands(std::move(commandBuffer), queue);

		defaultTextureView = ImageView::createImageView(device, *defaultTextureImage, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
	}

	// Flat normal 1x1
	{
		const uint32_t flatNormal = 0xFFFF7F7F;
		vk::DeviceSize imageSize = sizeof(uint32_t);

		vk::raii::Buffer       stagingBuf({});
		vk::raii::DeviceMemory stagingMem({});
		Buffer::createBuffer(device, physicalDevice, sizeof(uint32_t),
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			stagingBuf, stagingMem);
		void* d = stagingMem.mapMemory(0, sizeof(uint32_t));
		memcpy(d, &flatNormal, sizeof(uint32_t));
		stagingMem.unmapMemory();

		Image::createImage(device, physicalDevice, 1, 1, vk::Format::eR8G8B8A8Unorm,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			defaultNormalImage, defaultNormalMemory);

		vk::raii::CommandBuffer commandBuffer = CommandBuffer::beginSingleTimeCommands(device, commandPool);
		Image::transitionImageLayout(commandBuffer, defaultNormalImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Buffer::copyBufferToImage(commandBuffer, stagingBuf, defaultNormalImage, 1, 1);
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
	allocatorInfo.physicalDevice = *physicalDevice;
	allocatorInfo.device = *device;
	allocatorInfo.instance = *instance;

	if (vmaCreateAllocator(&allocatorInfo, &mRenderData.rdAllocator) != VK_SUCCESS)
		throw std::runtime_error("Failed to create VMA allocator for Assimp render data");

	ShaderStorageBuffer::init(mRenderData, mShaderNodeTransformBuffer);
	ShaderStorageBuffer::init(mRenderData, mShaderTRSMatrixBuffer);
	ShaderStorageBuffer::init(mRenderData, mShaderBoneMatrixBuffer);
	updateComputeDescriptorSets();
	initComputeSkinningResources();

	// Raw handles consumed by old VMA-based stack (VertexBuffer, IndexBuffer, Texture)
	mRenderData.rdDevice = *device;
	mRenderData.rdPhysicalDevice = *physicalDevice;
	mRenderData.rdInstance = *instance;
	mRenderData.rdGraphicsQueue = *queue;
	mRenderData.rdCommandPool = *commandPool;

	// ---- Skinning descriptor set layout ----
	// binding 0 = UBO          (vertex)
	// binding 1 = bone SSBO    (vertex)
	// binding 2 = diffuse tex  (fragment)
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings = { {
		{ 0, vk::DescriptorType::eUniformBuffer,        1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment },
		{ 1, vk::DescriptorType::eStorageBuffer,        1, vk::ShaderStageFlagBits::eVertex   },
		{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment }
	} };
	vk::DescriptorSetLayoutCreateInfo layoutInfo{
		.bindingCount = static_cast<uint32_t>(bindings.size()),
		.pBindings = bindings.data()
	};
	skinningDescriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
	mRenderData.rdAssimpTextureDescriptorLayout = *skinningDescriptorSetLayout;

	// ---- Descriptor pool ----
	std::array<vk::DescriptorPoolSize, 3> poolSizes = { {
		{ vk::DescriptorType::eUniformBuffer,        512 },
		{ vk::DescriptorType::eStorageBuffer,        512 },
		{ vk::DescriptorType::eCombinedImageSampler, 512 }
	} };
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 512,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};
	skinningDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);
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
		.magFilter = vk::Filter::eLinear,
		.minFilter = vk::Filter::eLinear,
		.mipmapMode = vk::SamplerMipmapMode::eLinear,
		.addressModeU = vk::SamplerAddressMode::eRepeat,
		.addressModeV = vk::SamplerAddressMode::eRepeat,
		.addressModeW = vk::SamplerAddressMode::eRepeat,
		.anisotropyEnable = vk::False,
		.maxAnisotropy = 1.0f,
		.compareEnable = vk::False,
		.compareOp = vk::CompareOp::eAlways,
		.borderColor = vk::BorderColor::eIntOpaqueBlack
	};
	skinningSampler = vk::raii::Sampler(device, samplerInfo);

	createSkinningPipeline();

	{
		Pipeline::PipelineConfig config{};
		config.shaderStages = {
			{ "shaders\\shadow_skinning.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" }
		};
		config.vertexBindings = { SkinnedVertex::getBindingDescription() };
		auto skinnedAttrib = SkinnedVertex::getAttributeDescriptions();
		config.vertexAttributes = { skinnedAttrib.begin(), skinnedAttrib.end() };
		config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);
		config.cullMode = vk::CullModeFlagBits::eNone;
		config.depthBiasEnable = true;
		config.depthBiasConstantFactor = 1.25f;
		config.depthBiasSlopeFactor = 1.75f;
		config.blendEnable = false;
		config.descriptorSetLayouts = { *skinningDescriptorSetLayout };
		config.pushConstantRanges = { vk::PushConstantRange{
			.stageFlags = vk::ShaderStageFlagBits::eVertex,
			.offset = 0,
			.size = sizeof(int)
		} };

		auto bundle = Pipeline::createPipeline(device, config);
		shadowSkinningPipelineLayout = std::move(bundle.layout);
		shadowSkinningPipeline = std::move(bundle.pipeline);
	}
}

void Renderer::initComputeSkinningResources()
{
	mComputeSkinningEnabled = false;

	std::array<vk::DescriptorSetLayoutBinding, 2> set0Bindings = { {
		{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
		{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute }
	} };
	vk::DescriptorSetLayoutCreateInfo set0LayoutInfo{
		.bindingCount = static_cast<uint32_t>(set0Bindings.size()),
		.pBindings = set0Bindings.data()
	};
	mComputeSetLayout0 = vk::raii::DescriptorSetLayout(device, set0LayoutInfo);

	std::array<vk::DescriptorSetLayoutBinding, 2> set1Bindings = { {
		{ 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
		{ 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute }
	} };
	vk::DescriptorSetLayoutCreateInfo set1LayoutInfo{
		.bindingCount = static_cast<uint32_t>(set1Bindings.size()),
		.pBindings = set1Bindings.data()
	};
	mComputeSetLayout1 = vk::raii::DescriptorSetLayout(device, set1LayoutInfo);

	std::array<vk::DescriptorPoolSize, 1> poolSizes = { {
		{ vk::DescriptorType::eStorageBuffer, 2048 }
	} };
	vk::DescriptorPoolCreateInfo poolInfo{
		.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
		.maxSets = 1024,
		.poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
		.pPoolSizes = poolSizes.data()
	};
	mComputeDescriptorPool = vk::raii::DescriptorPool(device, poolInfo);

	std::array<vk::DescriptorSetLayout, 2> set0Layouts = { *mComputeSetLayout0, *mComputeSetLayout0 };
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *mComputeDescriptorPool,
		.descriptorSetCount = static_cast<uint32_t>(set0Layouts.size()),
		.pSetLayouts = set0Layouts.data()
	};
	auto set0 = device.allocateDescriptorSets(allocInfo);
	mComputeTrsSet0 = std::move(set0[0]);
	mComputeBoneSet0 = std::move(set0[1]);

	const std::string shaderDir = "shaders\\";
	const std::string trsCompPath = shaderDir + "trs_matrix.comp.spv";
	const std::string boneCompPath = shaderDir + "bone_matrix.comp.spv";
	if (!std::filesystem::exists(trsCompPath) || !std::filesystem::exists(boneCompPath))
	{
		Logger::log(1, "%s warning: compute skinning shaders not found, skipping compute path\n", __FUNCTION__);
		return;
	}

	vk::PushConstantRange pushRange{
		.stageFlags = vk::ShaderStageFlagBits::eCompute,
		.offset = 0,
		.size = sizeof(uint32_t)
	};

	auto trsCode = readSpvU32(trsCompPath);
	vk::ShaderModuleCreateInfo trsModuleInfo{ .codeSize = trsCode.size() * sizeof(uint32_t), .pCode = trsCode.data() };
	vk::raii::ShaderModule trsModule(device, trsModuleInfo);
	vk::PipelineLayoutCreateInfo trsLayoutInfo{
		.setLayoutCount = 1,
		.pSetLayouts = &*mComputeSetLayout0,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushRange
	};
	mComputeTrsPipelineLayout = vk::raii::PipelineLayout(device, trsLayoutInfo);
	vk::ComputePipelineCreateInfo trsPipelineInfo{
		.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *trsModule, .pName = "main" },
		.layout = *mComputeTrsPipelineLayout
	};
	mComputeTrsPipeline = vk::raii::Pipeline(device, nullptr, trsPipelineInfo);

	std::array<vk::DescriptorSetLayout, 2> boneLayouts = { *mComputeSetLayout0, *mComputeSetLayout1 };
	auto boneCode = readSpvU32(boneCompPath);
	vk::ShaderModuleCreateInfo boneModuleInfo{ .codeSize = boneCode.size() * sizeof(uint32_t), .pCode = boneCode.data() };
	vk::raii::ShaderModule boneModule(device, boneModuleInfo);
	vk::PipelineLayoutCreateInfo boneLayoutInfo{
		.setLayoutCount = static_cast<uint32_t>(boneLayouts.size()),
		.pSetLayouts = boneLayouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushRange
	};
	mComputeBonePipelineLayout = vk::raii::PipelineLayout(device, boneLayoutInfo);
	vk::ComputePipelineCreateInfo bonePipelineInfo{
		.stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = *boneModule, .pName = "main" },
		.layout = *mComputeBonePipelineLayout
	};
	mComputeBonePipeline = vk::raii::Pipeline(device, nullptr, bonePipelineInfo);

	vk::CommandBufferAllocateInfo cmdAllocInfo{
		  .commandPool = *commandPool,
		  .level = vk::CommandBufferLevel::ePrimary,
		  .commandBufferCount = 1
	};
	auto computeCmds = device.allocateCommandBuffers(cmdAllocInfo);
	mComputeCommandBuffer = std::move(computeCmds[0]);
	mComputeSkinningEnabled = true;
}

void Renderer::ensureComputeModelResources(const std::shared_ptr<AssimpModel>& model)
{
	if (!model)
		return;

	const std::string key = model->getModelFileName();
	if (mComputeModelResources.count(key) > 0)
		return;

	ComputeModelResources resources{};
	resources.boneCount = static_cast<uint32_t>(model->getBoneList().size());
	if (resources.boneCount == 0)
		return;

	std::vector<int32_t> parentIndices(resources.boneCount, -1);
	std::vector<glm::mat4> boneOffsets(resources.boneCount, glm::mat4(1.0f));

	const auto& nodeList = model->getNodeList();
	std::unordered_map<std::string, int32_t> nodeToBone{};
	nodeToBone.reserve(resources.boneCount);
	for (const auto& bone : model->getBoneList())
		nodeToBone[bone->getBoneName()] = static_cast<int32_t>(bone->getBoneId());

	for (const auto& node : nodeList)
	{
		auto it = nodeToBone.find(node->getNodeName());
		if (it == nodeToBone.end())
			continue;

		const int32_t nodeBoneId = it->second;
		if (nodeBoneId < 0 || static_cast<uint32_t>(nodeBoneId) >= resources.boneCount)
			continue;

		int32_t parentBoneId = -1;
		auto parentNode = node->getParentNode();
		while (parentNode)
		{
			auto pb = nodeToBone.find(parentNode->getNodeName());
			if (pb != nodeToBone.end()) {
				parentBoneId = pb->second;
				break;
			}
			parentNode = parentNode->getParentNode();
		}

		parentIndices[nodeBoneId] = parentBoneId;
	}

	for (const auto& bone : model->getBoneList())
	{
		const uint32_t id = bone->getBoneId();
		if (id < resources.boneCount)
			boneOffsets[id] = bone->getOffsetMatrix();
	}

	ShaderStorageBuffer::init(mRenderData, resources.parentIndexBuffer, parentIndices.size() * sizeof(int32_t));
	ShaderStorageBuffer::init(mRenderData, resources.boneOffsetBuffer, boneOffsets.size() * sizeof(glm::mat4));
	ShaderStorageBuffer::uploadSsboData(mRenderData, resources.parentIndexBuffer, parentIndices);
	ShaderStorageBuffer::uploadSsboData(mRenderData, resources.boneOffsetBuffer, boneOffsets);

	std::array<vk::DescriptorSetLayout, 1> set1Layouts = { *mComputeSetLayout1 };
	vk::DescriptorSetAllocateInfo allocInfo{
		.descriptorPool = *mComputeDescriptorPool,
		.descriptorSetCount = 1,
		.pSetLayouts = set1Layouts.data()
	};
	auto sets = device.allocateDescriptorSets(allocInfo);
	resources.set1Descriptor = std::move(sets[0]);

	vk::DescriptorBufferInfo parentInfo{ .buffer = resources.parentIndexBuffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
	vk::DescriptorBufferInfo offsetInfo{ .buffer = resources.boneOffsetBuffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
	std::array<vk::WriteDescriptorSet, 2> writes = { {
	  {.dstSet = *resources.set1Descriptor, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &parentInfo },
		{.dstSet = *resources.set1Descriptor, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &offsetInfo },
	} };
	device.updateDescriptorSets(writes, nullptr);

	mComputeModelResources.insert({ key, std::move(resources) });
}

void Renderer::createAssimpInstanceGPUData(std::shared_ptr<AssimpInstance> instance)
{
	if (!instance)
		return;

	auto existingIt = std::find_if(mAssimpGPUData.begin(), mAssimpGPUData.end(),
		[&instance](const AssimpInstanceGPUData& d) { return d.instance == instance; });
	if (existingIt != mAssimpGPUData.end())
		return;

	AssimpInstanceGPUData gpuData;
	gpuData.instance = instance;

	auto& model = *instance->getModel();
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
			.descriptorPool = *skinningDescriptorPool,
			.descriptorSetCount = meshCount,
			.pSetLayouts = layouts.data()
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
			vk::ImageView diffuseView = *skinningWhiteView;
			if (m < meshes.size())
			{
				const VkTextureData* tex = model.getDiffuseTexture(m);
				if (tex && tex->imageView != VK_NULL_HANDLE)
					diffuseView = tex->imageView;
			}

			vk::DescriptorImageInfo imgInfo{
				.sampler = *skinningSampler,
				.imageView = diffuseView,
				.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
			};
			std::array<vk::WriteDescriptorSet, 3> writes = { {
				{
					.dstSet = *gpuData.descriptorSets[f][m],
					.dstBinding = 0,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eUniformBuffer,
					.pBufferInfo = &uboInfo
				},
				{
					.dstSet = *gpuData.descriptorSets[f][m],
					.dstBinding = 1,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eStorageBuffer,
					.pBufferInfo = &boneInfo
				},
				{
					.dstSet = *gpuData.descriptorSets[f][m],
					.dstBinding = 2,
					.descriptorCount = 1,
					.descriptorType = vk::DescriptorType::eCombinedImageSampler,
					.pImageInfo = &imgInfo
				}
			} };
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

// Helper triggered from Entt on_destroy to free GPU data when an AssimpInstanceComponent is removed.
void Renderer::onAssimpInstanceDestroyed(AssimpInstance* rawPtr)
{
	auto instance = AssimpSystems::FindInstance(mModelInstData, rawPtr);
	if (!instance)
		return;

	AssimpSystems::UnregisterInstance(mModelInstData, instance,
		[this](const std::shared_ptr<AssimpInstance>& unregisteringInstance) {
			deleteAssimpInstanceGPUData(unregisteringInstance);
		});

	if (mModelInstData.miAssimpInstances.empty()) {
		mModelInstData.miSelectedInstance = 0;
	}
	else if (mModelInstData.miSelectedInstance >= static_cast<int>(mModelInstData.miAssimpInstances.size())) {
		mModelInstData.miSelectedInstance = static_cast<int>(mModelInstData.miAssimpInstances.size()) - 1;
	}

	if (mModelInstData.miModelList.empty()) {
		mModelInstData.miSelectedModel = 0;
	}
	else if (mModelInstData.miSelectedModel >= static_cast<int>(mModelInstData.miModelList.size())) {
		mModelInstData.miSelectedModel = static_cast<int>(mModelInstData.miModelList.size()) - 1;
	}

	auto& registry = mEnttScene.getRegistry();
	entt::entity entity = AssimpSystems::FindEntityForInstance(registry, rawPtr);
	if (entity != entt::null && registry.valid(entity) && registry.any_of<AnimationComponent>(entity)) {
		registry.remove<AnimationComponent>(entity);
	}

	updateTriangleCount();
}

void Renderer::updateAssimpAnimations(float deltaTime)
{
	if (mModelInstData.miAssimpInstances.empty())
		return;

	auto& registry = mEnttScene.getRegistry();
	AssimpSystems::SyncTransformsFromEntt(registry);
	auto modelBatches = AssimpSystems::CollectValidModelInstanceBatches(mModelInstData.miAssimpInstancesPerModel);

	/* calculate the size of the node matrix buffer over all animated instances */
	size_t boneMatrixBufferSize = 0;
	for (const auto& modelType : modelBatches) {
		size_t numberOfInstances = modelType.instances->size();
		std::shared_ptr<AssimpModel> model = modelType.model;
		if (numberOfInstances > 0 && model->getTriangleCount() > 0) {

			/* animated models */
			if (model->hasAnimations() && !model->getBoneList().empty()) {
				size_t numberOfBones = model->getBoneList().size();

				/* buffer size must always be a multiple of "local_size_y" instances to avoid undefined behavior */
				boneMatrixBufferSize += numberOfBones * ((numberOfInstances - 1) / 32 + 1) * 32;
			}
		}
	}

	/* clear and resize world pos matrices */
	mWorldPosMatrices.clear();
	mWorldPosMatrices.resize(mModelInstData.miAssimpInstances.size());
	mNodeTransFormData.clear();
	mNodeTransFormData.resize(boneMatrixBufferSize);

	/* we need to track the presence of animated models */
	bool animatedModelLoaded = false;

	mInstanceBoneOffsets.clear();

	size_t instanceToStore = 0;
	size_t animatedInstancesToStore = 0;
	for (const auto& modelType : modelBatches) {
		size_t numberOfInstances = modelType.instances->size();
		if (numberOfInstances > 0) {
			std::shared_ptr<AssimpModel> model = modelType.model;

			/* animated models */
			if (model->hasAnimations() && !model->getBoneList().empty()) {
				size_t numberOfBones = model->getBoneList().size();
				animatedModelLoaded = true;

				for (unsigned int i = 0; i < numberOfInstances; ++i) {
					modelType.instances->at(i)->updateAnimation(deltaTime);
					std::vector<NodeTransformData> instanceNodeTransform = modelType.instances->at(i)->getNodeTransformData();
					std::copy(instanceNodeTransform.begin(), instanceNodeTransform.end(), mNodeTransFormData.begin() + animatedInstancesToStore + i * numberOfBones);
					mWorldPosMatrices.at(instanceToStore + i) = modelType.instances->at(i)->getWorldTransformMatrix();
					mInstanceBoneOffsets[modelType.instances->at(i).get()] = static_cast<uint32_t>(animatedInstancesToStore + i * numberOfBones);
				}

				size_t trsMatrixSize = numberOfBones * numberOfInstances * sizeof(glm::mat4);
				mRenderData.rdMatricesSize += trsMatrixSize;

				instanceToStore += numberOfInstances;
				animatedInstancesToStore += numberOfInstances * numberOfBones;
			}
			else {
				/* non-animated models */
				for (unsigned int i = 0; i < numberOfInstances; ++i) {
					mWorldPosMatrices.at(instanceToStore + i) = modelType.instances->at(i)->getWorldTransformMatrix();
				}

				mRenderData.rdMatricesSize += numberOfInstances * sizeof(glm::mat4);
				instanceToStore += numberOfInstances;
			}
		}
	}

	bool bufferResized = false;
	bufferResized = ShaderStorageBuffer::uploadSsboData(mRenderData, mShaderNodeTransformBuffer, mNodeTransFormData);

	/* resize SSBO if needed */
	bufferResized |= ShaderStorageBuffer::checkForResize(mRenderData, mShaderTRSMatrixBuffer, boneMatrixBufferSize * sizeof(glm::mat4));
	bufferResized |= ShaderStorageBuffer::checkForResize(mRenderData, mShaderBoneMatrixBuffer, boneMatrixBufferSize * sizeof(glm::mat4));

	if (bufferResized) {
		updateDescriptorSets();
		updateComputeDescriptorSets();
	}

	if (animatedModelLoaded && mComputeSkinningEnabled) {
		uint32_t computeShaderModelOffset = 0;
		for (const auto& modelType : modelBatches) {
			size_t numberOfInstances = modelType.instances->size();
			if (numberOfInstances == 0)
				continue;
			std::shared_ptr<AssimpModel> modelRef = modelType.model;
			if (modelRef && modelRef->hasAnimations() && !modelRef->getBoneList().empty()) {
				runComputeShaders(modelRef, numberOfInstances, computeShaderModelOffset);
				computeShaderModelOffset += static_cast<uint32_t>(numberOfInstances * modelRef->getBoneList().size());
			}
		}
		updateDescriptorSets();
	}
}

void Renderer::updateDescriptorSets()
{
	for (auto& gpuData : mAssimpGPUData)
	{
		auto& model = *gpuData.instance->getModel();
		const auto& meshes = model.getModelMeshes();
		const uint32_t meshCount = static_cast<uint32_t>(meshes.empty() ? 1 : meshes.size());

		const size_t boneCount = std::max<size_t>(1, model.getBoneList().size());
		vk::DeviceSize boneRange = static_cast<vk::DeviceSize>(sizeof(glm::mat4) * boneCount);
		vk::DeviceSize boneOffset = 0;
		if (mComputeSkinningEnabled && model.hasAnimations())
		{
			auto offsetIt = mInstanceBoneOffsets.find(gpuData.instance.get());
			if (offsetIt != mInstanceBoneOffsets.end())
				boneOffset = static_cast<vk::DeviceSize>(offsetIt->second) * sizeof(glm::mat4);
		}
		for (uint32_t f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f)
		{
			vk::DescriptorBufferInfo boneInfo{};
			if (mComputeSkinningEnabled && model.hasAnimations()) {
				boneInfo = { .buffer = mShaderBoneMatrixBuffer.buffer, .offset = boneOffset, .range = boneRange };
			}
			else {
				boneInfo = { .buffer = *gpuData.boneBuffer, .offset = 0, .range = boneRange };
			}
			vk::DescriptorBufferInfo uboInfo{ .buffer = *gpuData.uboBuffers[f], .offset = 0, .range = sizeof(UniformBufferObject) };

			for (uint32_t m = 0; m < meshCount; ++m)
			{
				vk::ImageView diffuseView = *skinningWhiteView;
				if (m < meshes.size())
				{
					const VkTextureData* tex = model.getDiffuseTexture(m);
					if (tex && tex->imageView != VK_NULL_HANDLE)
						diffuseView = tex->imageView;
				}

				vk::DescriptorImageInfo imgInfo{
					.sampler = *skinningSampler,
					.imageView = diffuseView,
					.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
				};

				std::array<vk::WriteDescriptorSet, 3> writes = { {
					{.dstSet = *gpuData.descriptorSets[f][m], .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eUniformBuffer, .pBufferInfo = &uboInfo },
					{.dstSet = *gpuData.descriptorSets[f][m], .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &boneInfo },
					{.dstSet = *gpuData.descriptorSets[f][m], .dstBinding = 2, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &imgInfo }
				} };

				device.updateDescriptorSets(writes, nullptr);
			}
		}
	}
}

void Renderer::updateComputeDescriptorSets()
{
	if (!mComputeSkinningEnabled)
		return;

	vk::DescriptorBufferInfo nodeInfo{
		 .buffer = mShaderNodeTransformBuffer.buffer,
		 .offset = 0,
		 .range = VK_WHOLE_SIZE
	};
	vk::DescriptorBufferInfo trsInfo{
		.buffer = mShaderTRSMatrixBuffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};
	vk::DescriptorBufferInfo boneInfo{
		.buffer = mShaderBoneMatrixBuffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE
	};

	std::array<vk::WriteDescriptorSet, 4> writes = { {
		{.dstSet = *mComputeTrsSet0, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &nodeInfo },
		{.dstSet = *mComputeTrsSet0, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &trsInfo },
		{.dstSet = *mComputeBoneSet0, .dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &trsInfo },
		{.dstSet = *mComputeBoneSet0, .dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &boneInfo }
	} };
	device.updateDescriptorSets(writes, nullptr);
}

void Renderer::runComputeShaders(const std::shared_ptr<AssimpModel>& model, size_t numberOfInstances, uint32_t modelOffset)
{
	if (!mComputeSkinningEnabled || !model || !model->hasAnimations() || model->getBoneList().empty() || numberOfInstances == 0)
		return;

	ensureComputeModelResources(model);
	auto modelIt = mComputeModelResources.find(model->getModelFileName());
	if (modelIt == mComputeModelResources.end())
		return;

	const uint32_t numberOfBones = static_cast<uint32_t>(model->getBoneList().size());
	vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	mComputeCommandBuffer.reset();
	mComputeCommandBuffer.begin(beginInfo);

	mComputeCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *mComputeTrsPipeline);
	mComputeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *mComputeTrsPipelineLayout, 0, *mComputeTrsSet0, nullptr);
	mComputeCommandBuffer.pushConstants<uint32_t>(*mComputeTrsPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, modelOffset);
	mComputeCommandBuffer.dispatch(numberOfBones, static_cast<uint32_t>((numberOfInstances - 1) / 32 + 1), 1);

	vk::MemoryBarrier2 memBarrier{
		.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
		.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
		.dstAccessMask = vk::AccessFlagBits2::eShaderRead
	};
	vk::DependencyInfo depInfo{ .memoryBarrierCount = 1, .pMemoryBarriers = &memBarrier };
	mComputeCommandBuffer.pipelineBarrier2(depInfo);

	mComputeCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *mComputeBonePipeline);
	std::array<vk::DescriptorSet, 2> sets = { *mComputeBoneSet0, *modelIt->second.set1Descriptor };
	mComputeCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *mComputeBonePipelineLayout, 0, sets, nullptr);
	mComputeCommandBuffer.pushConstants<uint32_t>(*mComputeBonePipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, modelOffset);
	mComputeCommandBuffer.dispatch(numberOfBones, static_cast<uint32_t>((numberOfInstances - 1) / 32 + 1), 1);

	mComputeCommandBuffer.end();

	vk::CommandBuffer rawCmd = *mComputeCommandBuffer;
	vk::SubmitInfo submitInfo{ .commandBufferCount = 1, .pCommandBuffers = &rawCmd };
	queue.submit(submitInfo, nullptr);
	queue.waitIdle();
}

void Renderer::recordAssimpSkinnedPass(vk::raii::CommandBuffer& commandBuffer)
{
	if (mAssimpGPUData.empty() || mModelInstData.miAssimpInstances.empty() || *skinningPipeline == VK_NULL_HANDLE)
		return;

	const UniformBufferObject* shadowTemplateUbo = nullptr;
	auto& registry = mEnttScene.getRegistry();
	for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
		(void)entity;
		if (renderable.uniformBuffersMapped.size() <= frameIndex)
			continue;
		if (!renderable.uniformBuffersMapped[frameIndex])
			continue;

		shadowTemplateUbo = reinterpret_cast<const UniformBufferObject*>(renderable.uniformBuffersMapped[frameIndex]);
		break;
	}

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
		auto& model = *gpuData.instance->getModel();
		const auto& meshes = model.getModelMeshes();
		const auto& vbos = model.getVertexBuffers();
		const auto& ibos = model.getIndexBuffers();

		if (meshes.empty()) continue;

		// Update the persistent per-frame UBO
		UniformBufferObject ubo{};
		if (shadowTemplateUbo)
			ubo = *shadowTemplateUbo;
		ubo.model = gpuData.instance->getWorldTransformMatrix();
		ubo.view = camera.getViewMatrix();
		ubo.proj = camera.getProjectionMatrix(aspect, 0.1f, 3000.0f);
		ubo.directionalLightDirection = glm::vec4(glm::normalize(shadowSettings.lightDirection), 0.0f);
		ubo.directionalLightColor = glm::vec4(1.0f);
		memcpy(gpuData.uboMapped[frameIndex], &ubo, sizeof(UniformBufferObject));

		// Draw each mesh with its own descriptor set (binding 0=UBO, 1=bones, 2=diffuse)
		for (size_t i = 0; i < meshes.size(); ++i)
		{
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
				*skinningPipelineLayout, 0,
				*gpuData.descriptorSets[frameIndex][i], nullptr);

			vk::Buffer     vb = vbos[i].buffer;
			vk::DeviceSize off = 0;
			commandBuffer.bindVertexBuffers(0, vb, off);
			commandBuffer.bindIndexBuffer(ibos[i].buffer, 0, vk::IndexType::eUint32);
			commandBuffer.drawIndexed(
				static_cast<uint32_t>(meshes[i].indices.size()), 1, 0, 0, 0);
		}
	}
}

