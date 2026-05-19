#include "renderer/renderer.h"
#include <ktx.h>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <unordered_set>
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
#include "../core/scene_serialization.h"
#include "../../lib/ImGuiFileDialog.h"
#include "../../include/imgui_internal.h"

namespace {
	using json = nlohmann::json;

	bool hasLoadedTextureViews(const RenderableComponent::PBRTextures& textures)
	{
		return (*textures.baseColorView != VK_NULL_HANDLE) ||
			(*textures.metallicRoughnessView != VK_NULL_HANDLE) ||
			(*textures.normalView != VK_NULL_HANDLE) ||
			(*textures.occlusionView != VK_NULL_HANDLE) ||
			(*textures.emissiveView != VK_NULL_HANDLE);
	}

	bool hasAnyLoadedTextureViews(const std::vector<RenderableComponent::PBRTextures>& textures)
	{
		return std::any_of(textures.begin(), textures.end(),
			[](const RenderableComponent::PBRTextures& textureSet) {
				return hasLoadedTextureViews(textureSet);
			});
	}

	bool materialHasTexturePaths(const Material& material)
	{
		return !material.albedoTexturePath.empty() ||
			!material.metallicRoughnessTexturePath.empty() ||
			!material.normalTexturePath.empty() ||
			!material.occlusionTexturePath.empty() ||
			!material.emissiveTexturePath.empty();
	}

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
				auto model = getOrLoadModelGltfAsset(modelPath);
				if (model)
				{
					auto& renderable = registry.emplace_or_replace<RenderableComponent>(ecsEntity);
					renderable.vertices = model->vertices;
					renderable.indices = model->indices;
					renderable.meshes = model->meshes;
					renderable.materials = model->materials;
					renderable.sourceModelFile = model->sourceModelFile;
					renderable.materialTextures.clear();
					renderable.materialTextures.resize(renderable.materials.size());
					renderable.materialDescriptorSets.clear();

					tryReuseCachedGltfTextures(modelPath, renderable);
					for (size_t i = 0; i < renderable.materials.size(); ++i)
						loadPBRTextures(renderable.materials[i], renderable.materialTextures[i]);
				}
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
		const glm::vec3& linearVelocity,
		bool alignBottomToEntity = false,
		const glm::vec3& centerOffset = glm::vec3(0.0f)) {
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
			collider.centerOffset = centerOffset;
			collider.alignBottomToEntity = alignBottomToEntity;

			physicsSystem.registerEntity(entity, registry);
		};

	makeEntity("Sponza",
		{ 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f },
		"models\\Sponza.gltf");

	const entt::entity floor = makeEntity(
		"Physics Arena Floor",
		{ 0.0f, 2.5f, 0.0f },
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f },
		"");
	addPhysics(floor, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.9f, 0.15f, false,
		{ 860.0f, 18.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f }, true);

	const entt::entity wallPosX = makeEntity("Physics Arena Wall +X", { 860.0f, -260.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallPosX, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 18.0f, 300.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f }, false);

	const entt::entity wallNegX = makeEntity("Physics Arena Wall -X", { -860.0f, -260.0f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallNegX, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 18.0f, 300.0f, 860.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f }, false);

	const entt::entity wallPosZ = makeEntity("Physics Arena Wall +Z", { 0.0f, -260.0f, 860.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallPosZ, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 860.0f, 300.0f, 18.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f }, false);

	const entt::entity wallNegZ = makeEntity("Physics Arena Wall -Z", { 0.0f, -260.0f, -860.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, "");
	addPhysics(wallNegZ, RigidBodyType::Static, ColliderShapeType::Box,
		1.0f, 0.8f, 0.1f, false,
		{ 860.0f, 300.0f, 18.0f }, 1.0f, 1.0f, { 0.0f, 0.0f, 0.0f }, false);

	const int spawnCount = std::max(2, physicsSpawnCount);
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
		const std::string modelPath = damagedHelmet ? "models\\DamagedHelmet.gltf" : "models\\FlightHelmet.gltf";
		const glm::vec3 scale = damagedHelmet ? glm::vec3(35.0f) : glm::vec3(100.0f);
		const glm::vec3 rotation = damagedHelmet ? glm::vec3(0.0f, angle, 0.0f) : glm::vec3(0.0f, angle + glm::radians(90.0f), 0.0f);
		const bool alignBottom = damagedHelmet;


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
			startVelocity,
			alignBottom);
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

		if (renderable.materialTextures.size() != renderable.materials.size()) {
			renderable.materialTextures.clear();
			renderable.materialTextures.resize(renderable.materials.size());
		}

		if (!renderable.sourceModelFile.empty()) {
			if (!hasAnyLoadedTextureViews(renderable.materialTextures)) {
				tryReuseCachedGltfTextures(renderable.sourceModelFile, renderable);
			}
		}

		createVertexBuffer(renderable);
		createIndexBuffer(renderable);
		for (size_t i = 0; i < renderable.materials.size(); ++i) {
			if (hasLoadedTextureViews(renderable.materialTextures[i]))
				continue;

			if (!materialHasTexturePaths(renderable.materials[i]))
				continue;

			{
				loadPBRTextures(renderable.materials[i], renderable.materialTextures[i]);
			}
		}
	}

	UniformBuffer::createUniformBuffers(registry, device, physicalDevice, MAX_FRAMES_IN_FLIGHT);

	DescriptorPool::createDescriptorPool(device, registry, descriptorPool, MAX_FRAMES_IN_FLIGHT);
	std::array<vk::ImageView, SHADOW_CASCADE_COUNT> shadowViews = {
		   *shadowImageViews[0], *shadowImageViews[1], *shadowImageViews[2], *shadowImageViews[3], *shadowImageViews[4]
	};
	DescriptorSet::createDescriptorSets(device, registry, descriptorPool, descriptorSetLayout, defaultTextureView, defaultNormalView, textureSampler, shadowViews, shadowSampler, MAX_FRAMES_IN_FLIGHT);
}

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

	createColliderDebugPipeline();

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

	mSceneRuntimeService.clearHistory();
	pushUndoSnapshot();
}

std::string Renderer::normalizeModelAssetKey(const std::string& modelFileName) const
{
	if (modelFileName.empty())
		return {};

	std::filesystem::path path(modelFileName);
	std::string key = path.lexically_normal().generic_string();
	std::transform(key.begin(), key.end(), key.begin(),
		[](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return key;
}

std::shared_ptr<AssimpModel> Renderer::getOrLoadModelAssimpAsset(const std::string& modelFileName)
{
	const std::string key = normalizeModelAssetKey(modelFileName);
	if (key.empty())
		return nullptr;

	if (auto it = mModelAssetCache.find(key); it != mModelAssetCache.end() && it->second)
	{
		ensureModelInSceneList(it->second);
		return it->second;
	}

	for (const auto& existing : mModelInstData.miModelList)
	{
		if (!existing)
			continue;

		const std::string fullPathKey = normalizeModelAssetKey(existing->getModelFileNamePath());
		const std::string fileNameKey = normalizeModelAssetKey(existing->getModelFileName());
		if (fullPathKey == key || fileNameKey == key)
		{
			mModelAssetCache[key] = existing;
			ensureModelInSceneList(existing);
			return existing;
		}
	}

	auto model = std::make_shared<AssimpModel>();
	if (!model->loadModel(mRenderData, modelFileName))
		return nullptr;

	mModelAssetCache[key] = model;
	ensureModelInSceneList(model);
	return model;
}

void Renderer::ensureModelInSceneList(const std::shared_ptr<AssimpModel>& model)
{
	if (!model)
		return;

	const std::string fullPathKey = normalizeModelAssetKey(model->getModelFileNamePath());
	const std::string fileNameKey = normalizeModelAssetKey(model->getModelFileName());

	auto it = std::find_if(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(),
		[&](const std::shared_ptr<AssimpModel>& existing) {
			if (!existing)
				return false;
			return normalizeModelAssetKey(existing->getModelFileNamePath()) == fullPathKey ||
				normalizeModelAssetKey(existing->getModelFileName()) == fileNameKey;
		});

	if (it == mModelInstData.miModelList.end())
		mModelInstData.miModelList.emplace_back(model);
}

std::shared_ptr<RenderableComponent> Renderer::getOrLoadModelGltfAsset(const std::string& modelFileName)
{
	const std::string key = normalizeModelAssetKey(modelFileName);
	if (key.empty())
		return nullptr;

	if (auto it = mGltfModelAssetCache.find(key); it != mGltfModelAssetCache.end() && it->second)
	{
		ensureGltfModelInSceneList(it->second);
		return it->second;
	}

	if (auto existing = getRenderableModel(modelFileName))
	{
		mGltfModelAssetCache[key] = existing;
		ensureGltfModelInSceneList(existing);
		return existing;
	}

	auto model = std::make_shared<RenderableComponent>();
	try
	{
		Model::loadModel(modelFileName, *model);
	}
	catch (...)
	{
		return nullptr;
	}

	model->sourceModelFile = std::filesystem::path(modelFileName).lexically_normal().generic_string();
	mGltfModelAssetCache[key] = model;
	ensureGltfModelInSceneList(model);
	return model;
}

bool Renderer::tryReuseCachedGltfTextures(const std::string& modelFileName, RenderableComponent& renderable)
{
	if (renderable.materialTextures.size() != renderable.materials.size()) {
		renderable.materialTextures.clear();
		renderable.materialTextures.resize(renderable.materials.size());
	}

	bool reusedAny = false;
	for (size_t i = 0; i < renderable.materials.size(); ++i)
	{
		auto& material = renderable.materials[i];
		auto& textures = renderable.materialTextures[i];

		if (*textures.baseColorView == VK_NULL_HANDLE &&
			tryAssignCachedTextureResource(material.albedoTexturePath, textures.baseColorImage, textures.baseColorMemory, textures.baseColorView))
		{
			reusedAny = true;
		}

		if (*textures.metallicRoughnessView == VK_NULL_HANDLE &&
			tryAssignCachedTextureResource(material.metallicRoughnessTexturePath, textures.metallicRoughnessImage, textures.metallicRoughnessMemory, textures.metallicRoughnessView))
		{
			reusedAny = true;
		}

		if (*textures.normalView == VK_NULL_HANDLE &&
			tryAssignCachedTextureResource(material.normalTexturePath, textures.normalImage, textures.normalMemory, textures.normalView))
		{
			reusedAny = true;
		}

		if (*textures.occlusionView == VK_NULL_HANDLE &&
			tryAssignCachedTextureResource(material.occlusionTexturePath, textures.occlusionImage, textures.occlusionMemory, textures.occlusionView))
		{
			reusedAny = true;
		}

		if (*textures.emissiveView == VK_NULL_HANDLE &&
			tryAssignCachedTextureResource(material.emissiveTexturePath, textures.emissiveImage, textures.emissiveMemory, textures.emissiveView))
		{
			reusedAny = true;
		}
	}

	return reusedAny;
}

void Renderer::cacheTextureResource(const std::string& texturePath, vk::raii::Image& image, vk::raii::DeviceMemory& memory, vk::raii::ImageView& view)
{
	if (texturePath.empty())
		return;

	if (*view == VK_NULL_HANDLE)
		return;

	const std::string key = normalizeModelAssetKey(texturePath);
	if (key.empty())
		return;

	CachedTextureResource resource{};
	resource.image = std::move(image);
	resource.memory = std::move(memory);
	resource.view = std::move(view);
	mTextureAssetCache[key].emplace_back(std::move(resource));
}

bool Renderer::tryAssignCachedTextureResource(const std::string& texturePath, vk::raii::Image& image, vk::raii::DeviceMemory& memory, vk::raii::ImageView& view)
{
	if (texturePath.empty())
		return false;

	const std::string key = normalizeModelAssetKey(texturePath);
	if (key.empty())
		return false;

	auto it = mTextureAssetCache.find(key);
	if (it == mTextureAssetCache.end() || it->second.empty())
		return false;

	auto& entries = it->second;
	while (!entries.empty())
	{
		auto cached = std::move(entries.front());
		entries.pop_front();
		if (*cached.view == VK_NULL_HANDLE)
			continue;

		image = std::move(cached.image);
		memory = std::move(cached.memory);
		view = std::move(cached.view);
		return true;
	}

	return false;
}

void Renderer::captureGltfTexturesFromScene(entt::registry& registry)
{
	mGltfModelTextureCache.clear();
	mTextureAssetCache.clear();

	for (auto [entity, renderable] : registry.view<RenderableComponent>().each())
	{
		(void)entity;
		if (renderable.sourceModelFile.empty() || renderable.materialTextures.empty())
			continue;

		const std::string key = normalizeModelAssetKey(renderable.sourceModelFile);
		if (key.empty())
			continue;

		if (auto it = mGltfModelAssetCache.find(key); it == mGltfModelAssetCache.end() || !it->second)
		{
			auto modelAsset = std::make_shared<RenderableComponent>();
			modelAsset->vertices = renderable.vertices;
			modelAsset->indices = renderable.indices;
			modelAsset->meshes = renderable.meshes;
			modelAsset->materials = renderable.materials;
			modelAsset->sourceModelFile = std::filesystem::path(renderable.sourceModelFile).lexically_normal().generic_string();
			mGltfModelAssetCache[key] = modelAsset;
		}

		if (renderable.materialTextures.size() != renderable.materials.size())
			continue;

		if (!hasAnyLoadedTextureViews(renderable.materialTextures))
			continue;

		for (size_t materialIndex = 0; materialIndex < renderable.materialTextures.size(); ++materialIndex)
		{
			auto& material = renderable.materials[materialIndex];
			auto& materialTextures = renderable.materialTextures[materialIndex];

			cacheTextureResource(material.albedoTexturePath,
				materialTextures.baseColorImage, materialTextures.baseColorMemory, materialTextures.baseColorView);
			cacheTextureResource(material.metallicRoughnessTexturePath,
				materialTextures.metallicRoughnessImage, materialTextures.metallicRoughnessMemory, materialTextures.metallicRoughnessView);
			cacheTextureResource(material.normalTexturePath,
				materialTextures.normalImage, materialTextures.normalMemory, materialTextures.normalView);
			cacheTextureResource(material.occlusionTexturePath,
				materialTextures.occlusionImage, materialTextures.occlusionMemory, materialTextures.occlusionView);
			cacheTextureResource(material.emissiveTexturePath,
				materialTextures.emissiveImage, materialTextures.emissiveMemory, materialTextures.emissiveView);
		}
	}
}

void Renderer::ensureGltfModelInSceneList(const std::shared_ptr<RenderableComponent>& model)
{
	if (!model)
		return;

	const std::string key = normalizeModelAssetKey(model->sourceModelFile);
	auto it = std::find_if(mSceneRenderableModels.begin(), mSceneRenderableModels.end(),
		[&](const std::shared_ptr<RenderableComponent>& existing) {
			if (!existing)
				return false;
			return normalizeModelAssetKey(existing->sourceModelFile) == key;
		});

	if (it == mSceneRenderableModels.end())
		mSceneRenderableModels.emplace_back(model);
}

std::shared_ptr<RenderableComponent> Renderer::getRenderableModel(std::string modelFileName)
{
	const std::string key = normalizeModelAssetKey(modelFileName);
	if (key.empty())
		return nullptr;

	auto it = std::find_if(mSceneRenderableModels.begin(), mSceneRenderableModels.end(),
		[&](const std::shared_ptr<RenderableComponent>& model) {
			if (!model)
				return false;

			const std::string fullPathKey = normalizeModelAssetKey(model->sourceModelFile);
			const std::string fileNameKey = normalizeModelAssetKey(std::filesystem::path(model->sourceModelFile).filename().generic_string());
			return fullPathKey == key || fileNameKey == key;
		});

	return (it != mSceneRenderableModels.end()) ? *it : nullptr;
}

bool Renderer::hasModel(std::string modelFileName) {
	const std::string key = normalizeModelAssetKey(modelFileName);
	if (key.empty())
		return false;

	auto modelIter = std::find_if(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(),
		[&](const auto& model) {
			if (!model)
				return false;
			return normalizeModelAssetKey(model->getModelFileNamePath()) == key ||
				normalizeModelAssetKey(model->getModelFileName()) == key;
		});
	return modelIter != mModelInstData.miModelList.end();
}

std::shared_ptr<AssimpModel> Renderer::getModel(std::string modelFileName) {
	const std::string key = normalizeModelAssetKey(modelFileName);
	if (key.empty())
		return nullptr;

	auto modelIter = std::find_if(mModelInstData.miModelList.begin(), mModelInstData.miModelList.end(),
		[&](const auto& model) {
			if (!model)
				return false;
			return normalizeModelAssetKey(model->getModelFileNamePath()) == key ||
				normalizeModelAssetKey(model->getModelFileName()) == key;
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

	std::shared_ptr<AssimpModel> model = getOrLoadModelAssimpAsset(modelFileName);
	if (!model) {
		return false;
	}

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

	constexpr float fixedStep = 1.0f / 60.0f;
	constexpr float maxFrameDelta = 0.25f;
	constexpr int maxSubsteps = 8;

	float accumulator = 0.0f;
	auto startTime = std::chrono::high_resolution_clock::now();

	while (!glfwWindowShouldClose(window))
	{
		auto        currentTime = std::chrono::high_resolution_clock::now();
		float       time = std::chrono::duration<float>(currentTime - startTime).count();
		float       deltaTime = time - lastFrameTime;
		lastFrameTime = time;

		deltaTime = std::min(deltaTime, maxFrameDelta);

		InputSystem::Update(deltaTime);

		const bool allowEditorCameraInput = !(sceneState == SceneState::PLAY && mGameplayRuntime.hasActiveLayer());
		if (allowEditorCameraInput)
			camera.processInput(window, camera, deltaTime);
		auto& reg = mEnttScene.getRegistry();

		if (sceneState == SceneState::PLAY)
		{
			accumulator += deltaTime;
			auto gameplayContext = buildGameplayRuntimeContext();

			int substeps = 0;

			while (accumulator >= fixedStep && substeps < maxSubsteps)
			{
				mGameplayRuntime.fixedUpdate(fixedStep, window, gameplayContext);
				physicsSystem.step(fixedStep, reg);
				updateAssimpAnimations(fixedStep);
				accumulator -= fixedStep;
				substeps++;
			}
		}
		else
		{
			accumulator = 0.0f;
			updateAssimpAnimations(0.0f);
		}

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

	colliderDebugPipeline = nullptr;
	colliderDebugPipelineLayout = nullptr;
	colliderDebugVertexBuffer = nullptr;
	colliderDebugVertexBufferMemory = nullptr;
	colliderDebugVertexCapacity = 0;
	colliderDebugVertices.clear();

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
		const std::string imguiIniPath = "imgui.ini";
		if (std::filesystem::exists(imguiIniPath)) {
			std::error_code ec;
			std::filesystem::remove(imguiIniPath, ec);
		}

		ImGui::LoadIniSettingsFromMemory("");

		mEditorDockLayoutInitialized = true;

		ImGui::DockBuilderRemoveNode(dockspaceId);
		ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
		ImGui::DockBuilderSetNodeSize(dockspaceId, mainViewport->Size);

		ImGuiID dockMain = dockspaceId;
		ImGuiID dockLeft = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Left, 0.20f, nullptr, &dockMain);
		ImGuiID dockRight = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Right, 0.28f, nullptr, &dockMain);
		ImGuiID dockBottom = ImGui::DockBuilderSplitNode(dockMain, ImGuiDir_Down, 0.30f, nullptr, &dockMain);
		ImGuiID dockLeftBottom = ImGui::DockBuilderSplitNode(dockLeft, ImGuiDir_Down, 0.45f, nullptr, &dockLeft);
		ImGuiID dockRightSecondary = ImGui::DockBuilderSplitNode(dockRight, ImGuiDir_Down, 0.48f, nullptr, &dockRight);

		ImGui::DockBuilderDockWindow("Viewport", dockMain);
		ImGui::DockBuilderDockWindow("ECS Scene", dockLeft);
		ImGui::DockBuilderDockWindow("ECS Lights", dockLeftBottom);
		ImGui::DockBuilderDockWindow("ECS Inspector", dockRight);
		ImGui::DockBuilderDockWindow("Camera Controls", dockRightSecondary);
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
	renderPlayModePanel(isEditMode);

	renderCameraControlsPanel();
	renderPrefabPanel(isEditMode, registry);
	renderShadowTuningPanel(isEditMode);
	renderAnimationControlsPanel(isEditMode);
	renderPlayHudPanel(isEditMode, registry);

	if (isEditMode)
	{
		renderEnttEditor(camera.getViewMatrix(), camera.getProjectionMatrix(static_cast<float>(WIDTH) / HEIGHT, 0.1f, 3000.0f));
	}

	renderPostProcessingAndPhysicsPanels(isEditMode, registry);

	// End the frame
	ImGui::EndFrame();

	// Render to generate draw data
	ImGui::Render();

	imGui->updateBuffers();
}

void Renderer::enterPlayMode()
{
	mSceneRuntimeService.saveEditorScene([this]() { return serializeEnttScene(); });

	mLogPlayToEditCacheStats = false;

	sceneState = SceneState::PLAY;
	currentFrameIndex = 0;
	physicsSystem.setPaused(false);

	auto gameplayContext = buildGameplayRuntimeContext();
	mGameplayRuntime.enterPlay(gameplayContext);
}

void Renderer::exitPlayMode()
{
	mLogPlayToEditCacheStats = true;

	if (mSceneRuntimeService.loadEditorScene([this](const std::string& jsonContent) {
		pushUndoSnapshot();
		return deserializeEnttScene(jsonContent);
		})) {
	}

	mLogPlayToEditCacheStats = false;

	auto gameplayContext = buildGameplayRuntimeContext();
	mGameplayRuntime.exitPlay(gameplayContext);

	sceneState = SceneState::EDIT;
	currentFrameIndex = 0;
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
		Core::SceneSerialization::serializeCommonComponents(registry, entity, node);

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

		root["entities"].push_back(node);
	}

	return root.dump(2);
}

std::shared_ptr<AssimpModel> Renderer::ensureModelLoadedForScene(const std::string& modelFileName)
{
	if (modelFileName.empty())
		return nullptr;

	return getOrLoadModelAssimpAsset(modelFileName);
}

bool Renderer::deserializeEnttScene(const std::string& sceneJson)
{
	json root = json::parse(sceneJson, nullptr, false);
	if (root.is_discarded() || !root.contains("entities") || !root["entities"].is_array()) {
		return false;
	}

	struct CacheReloadStats {
		uint32_t gltfModelCacheHits = 0;
		uint32_t gltfModelReloads = 0;
		uint32_t gltfTextureCacheHits = 0;
		uint32_t gltfTextureReloads = 0;
	};
	CacheReloadStats cacheStats{};

	auto& registry = mEnttScene.getRegistry();
	captureGltfTexturesFromScene(registry);

	std::vector<entt::entity> existingEntities;
	registry.view<EnttTagComponent>().each([&](entt::entity entity, EnttTagComponent&) {
		existingEntities.push_back(entity);
		});

	for (auto entity : existingEntities) {
		if (auto* renderable = registry.try_get<RenderableComponent>(entity)) {
			renderable->materialDescriptorSets.clear();
		}
		mEnttScene.destroyEntity(entity);
	}

	mEnttSelectedEntity = entt::null;
	mEnttMultiSelection.clear();

	std::unordered_map<uint32_t, entt::entity> entityMap;
	std::vector<Core::SceneSerialization::PendingHierarchy> pendingHierarchy;
	bool addedRenderable = false;

	for (const auto& node : root["entities"]) {
		const uint32_t sourceId = node.value("id", 0u);
		const std::string name = node.value("name", std::string("Entity"));
		entt::entity entity = mEnttScene.createEntity(name);
		entityMap[sourceId] = entity;

		Core::SceneSerialization::deserializeCommonComponents(registry, entity, node);

		if (node.contains("renderable") && node["renderable"].is_object()) {
			const auto& renderableNode = node["renderable"];
			const std::string modelPath = renderableNode.value("modelPath", std::string());
			if (!modelPath.empty()) {
				auto model = getOrLoadModelGltfAsset(modelPath);
				if (model) {
					addedRenderable = true;
					auto& renderableComp = registry.emplace_or_replace<RenderableComponent>(entity);
					renderableComp.vertices = model->vertices;
					renderableComp.indices = model->indices;
					renderableComp.meshes = model->meshes;
					renderableComp.materials = model->materials;
					renderableComp.sourceModelFile = model->sourceModelFile;
					renderableComp.materialTextures.clear();
					renderableComp.materialDescriptorSets.clear();

					if (!tryReuseCachedGltfTextures(modelPath, renderableComp)) {
						renderableComp.materialTextures.resize(renderableComp.materials.size());
						for (size_t i = 0; i < renderableComp.materials.size(); ++i)
							loadPBRTextures(renderableComp.materials[i], renderableComp.materialTextures[i]);
					}
				}
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
					if (s.contains("position") && s["position"].is_array() && s["position"].size() == 3)
						settings.isWorldPosition = glm::vec3(s["position"][0].get<float>(), s["position"][1].get<float>(), s["position"][2].get<float>());
					if (s.contains("rotation") && s["rotation"].is_array() && s["rotation"].size() == 3)
						settings.isWorldRotation = glm::vec3(s["rotation"][0].get<float>(), s["rotation"][1].get<float>(), s["rotation"][2].get<float>());
					settings.isScale = s.value("scale", settings.isScale);
					settings.isSwapYZAxis = s.value("swapYZ", settings.isSwapYZAxis);
					settings.isAnimClipNr = s.value("clipIndex", settings.isAnimClipNr);
					settings.isAnimPlayTimePos = s.value("playTime", settings.isAnimPlayTimePos);
					settings.isAnimSpeedFactor = s.value("speed", settings.isAnimSpeedFactor);
					assimpComp->instance->setInstanceSettings(settings);
				}
			}
		}

		pendingHierarchy.push_back(Core::SceneSerialization::buildPendingHierarchy(registry, entity, node));
	}

	std::vector<entt::entity> createdEntities;
	for (const auto& [id, entity] : entityMap) {
		if (!registry.valid(entity))
			continue;
		createdEntities.push_back(entity);
	}

	Core::SceneSerialization::applyPendingHierarchy(registry, pendingHierarchy, entityMap);

	for (entt::entity entity : createdEntities) {
		if (registry.any_of<RigidBodyComponent, ColliderComponent, TransformComponent>(entity)) {
			entt::entity e = entity;
			physicsSystem.registerEntity(e, registry);
		}
	}

	if (addedRenderable)
		rebuildRenderableRuntimeResources();

	const uint32_t selectedId = root.value("selected", 0u);
	auto selectedIt = entityMap.find(selectedId);
	entt::entity selectedEntity = (selectedIt != entityMap.end()) ? selectedIt->second : createdEntities.front();

	mEnttSelectedEntity = selectedEntity;
	mEnttMultiSelection.clear();
	mEnttMultiSelection.push_back(selectedEntity);
	return true;
}


bool Renderer::saveSelectedAsPrefab(const std::string& prefabPath)
{
	if (!mEnttScene.isValid(mEnttSelectedEntity))
		return false;

	auto& registry = mEnttScene.getRegistry();
	const auto* selectedTag = registry.try_get<EnttTagComponent>(mEnttSelectedEntity);
	if (!selectedTag)
		return false;

	using json = nlohmann::json;
	json root;
	root["entities"] = json::array();

	std::vector<entt::entity> subtree;
	std::unordered_set<entt::entity> subtreeSet;
	subtree.push_back(mEnttSelectedEntity);
	subtreeSet.insert(mEnttSelectedEntity);

	bool added = true;
	while (added) {
		added = false;
		for (auto [entity, hierarchy] : registry.view<HierarchyComponent>().each()) {
			if (!registry.valid(entity) || subtreeSet.contains(entity))
				continue;
			if (hierarchy.parent == entt::null)
				continue;
			if (!subtreeSet.contains(hierarchy.parent))
				continue;

			subtree.push_back(entity);
			subtreeSet.insert(entity);
			added = true;
		}
	}

	if (subtree.empty())
		return false;

	std::unordered_map<uint32_t, uint32_t> prefabIdMap;
	prefabIdMap.reserve(subtree.size());
	uint32_t nextPrefabId = 1u;
	for (entt::entity entity : subtree) {
		prefabIdMap.emplace(static_cast<uint32_t>(entt::to_integral(entity)), nextPrefabId++);
	}

	const std::string prefabEntityName = std::filesystem::path(prefabPath).stem().string();

	for (entt::entity entity : subtree) {
		json node;
		const uint32_t entityKey = static_cast<uint32_t>(entt::to_integral(entity));
		node["id"] = prefabIdMap[entityKey];

		const auto* tag = registry.try_get<EnttTagComponent>(entity);
		if (entity == mEnttSelectedEntity) {
			const std::string outlinerName = prefabEntityName.empty() ? selectedTag->name : prefabEntityName;
			node["name"] = outlinerName;
		}
		else {
			node["name"] = tag ? tag->name : "Entity";
		}
		Core::SceneSerialization::serializeCommonComponents(registry, entity, node);

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

		int64_t parentPrefabId = -1;
		glm::vec3 localPosition(0.0f);
		glm::quat localRotation(1.0f, 0.0f, 0.0f, 0.0f);
		glm::vec3 localScale(1.0f);

		if (const auto* hierarchy = registry.try_get<HierarchyComponent>(entity)) {
			if (hierarchy->parent != entt::null) {
				auto parentIt = prefabIdMap.find(static_cast<uint32_t>(entt::to_integral(hierarchy->parent)));
				if (parentIt != prefabIdMap.end()) {
					parentPrefabId = static_cast<int64_t>(parentIt->second);
				}
			}
			localPosition = hierarchy->localPosition;
			localRotation = hierarchy->localRotation;
			localScale = hierarchy->localScale;
		}
		else if (const auto* transform = registry.try_get<TransformComponent>(entity)) {
			localPosition = transform->GetPosition();
			localRotation = transform->GetRotation();
			localScale = transform->GetScale();
		}

		node["hierarchy"] = {
			{ "parent", parentPrefabId },
			{ "localPosition", { localPosition.x, localPosition.y, localPosition.z } },
			{ "localRotation", { localRotation.x, localRotation.y, localRotation.z, localRotation.w } },
			{ "localScale", { localScale.x, localScale.y, localScale.z } }
		};

		root["entities"].push_back(node);
	}

	root["selected"] = 1u;

	std::filesystem::path outPath(prefabPath);
	if (outPath.has_parent_path())
		std::filesystem::create_directories(outPath.parent_path());

	std::ofstream outFile(prefabPath, std::ios::out | std::ios::trunc);
	if (!outFile.is_open())
		return false;

	outFile << root.dump(2);
	mSceneRuntimeService.state().markPrefabAssetsDirty();
	return true;
}

void Renderer::refreshPrefabAssetList()
{
	mSceneRuntimeService.state().refreshPrefabAssetList();
}

bool Renderer::instantiatePrefab(const std::string& prefabPath)
{
	std::ifstream inFile(prefabPath);
	if (!inFile.is_open())
		return false;

	std::string jsonContent((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
	if (jsonContent.empty())
		return false;

	using json = nlohmann::json;
	json root = json::parse(jsonContent, nullptr, false);
	if (root.is_discarded() || !root.contains("entities") || !root["entities"].is_array() || root["entities"].empty())
		return false;

	auto& registry = mEnttScene.getRegistry();
	std::unordered_map<uint32_t, entt::entity> entityMap;
	std::vector<entt::entity> createdEntities;
	createdEntities.reserve(root["entities"].size());

	for (const auto& node : root["entities"]) {
		const uint32_t prefabId = node.value("id", 0u);
		const std::string baseName = node.value("name", std::string("Prefab Entity"));
		entt::entity entity = mEnttScene.createEntity(makeUniqueEntityName(registry, baseName));
		entityMap[prefabId] = entity;
		createdEntities.push_back(entity);
	}

	std::vector<Core::SceneSerialization::PendingHierarchy> pendingHierarchy;
	pendingHierarchy.reserve(root["entities"].size());

	bool addedRenderable = false;

	size_t nodeIndex = 0;
	for (const auto& node : root["entities"]) {
		const uint32_t prefabId = node.value("id", 0u);
		auto entityIt = entityMap.find(prefabId);
		if (entityIt == entityMap.end()) {
			++nodeIndex;
			continue;
		}
		entt::entity entity = entityIt->second;

		Core::SceneSerialization::deserializeCommonComponents(registry, entity, node);

		if (node.contains("renderable") && node["renderable"].is_object()) {
			const auto& renderableNode = node["renderable"];
			const std::string modelPath = renderableNode.value("modelPath", std::string());
			if (!modelPath.empty()) {
				auto model = getOrLoadModelGltfAsset(modelPath);
				if (model) {
					addedRenderable = true;
					auto& renderableComp = registry.emplace_or_replace<RenderableComponent>(entity);
					renderableComp.vertices = model->vertices;
					renderableComp.indices = model->indices;
					renderableComp.meshes = model->meshes;
					renderableComp.materials = model->materials;
					renderableComp.sourceModelFile = model->sourceModelFile;
					renderableComp.materialTextures.clear();
					renderableComp.materialDescriptorSets.clear();

					if (!tryReuseCachedGltfTextures(modelPath, renderableComp)) {
						renderableComp.materialTextures.resize(renderableComp.materials.size());
						for (size_t i = 0; i < renderableComp.materials.size(); ++i)
							loadPBRTextures(renderableComp.materials[i], renderableComp.materialTextures[i]);
					}
				}
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
				}
			}
		}

		pendingHierarchy.push_back(Core::SceneSerialization::buildPendingHierarchy(registry, entity, node, nodeIndex == 0));
		++nodeIndex;
	}

	Core::SceneSerialization::applyPendingHierarchy(registry, pendingHierarchy, entityMap);

	for (entt::entity entity : createdEntities) {
		if (registry.any_of<RigidBodyComponent, ColliderComponent, TransformComponent>(entity)) {
			entt::entity e = entity;
			physicsSystem.registerEntity(e, registry);
		}
	}

	if (addedRenderable)
		rebuildRenderableRuntimeResources();

	const uint32_t selectedId = root.value("selected", 0u);
	auto selectedIt = entityMap.find(selectedId);
	entt::entity selectedEntity = (selectedIt != entityMap.end()) ? selectedIt->second : createdEntities.front();

	mEnttSelectedEntity = selectedEntity;
	mEnttMultiSelection.clear();
	mEnttMultiSelection.push_back(selectedEntity);
	return true;
}

void Renderer::pushUndoSnapshot()
{
	mSceneRuntimeService.pushUndoSnapshot([this]() { return serializeEnttScene(); });
}

void Renderer::performUndo()
{
	mSceneRuntimeService.undo(
		[this]() { return serializeEnttScene(); },
		[this](const std::string& snapshot) { return deserializeEnttScene(snapshot); });
}

void Renderer::performRedo()
{
	mSceneRuntimeService.redo(
		[this]() { return serializeEnttScene(); },
		[this](const std::string& snapshot) { return deserializeEnttScene(snapshot); });
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

void Renderer::createColliderDebugPipeline()
{
	vk::PushConstantRange pushRange{
		.stageFlags = vk::ShaderStageFlagBits::eVertex,
		.offset = 0,
		.size = sizeof(DebugLinePushConstants)
	};

	auto binding = DebugLineVertex::getBindingDescription();
	auto attribs = DebugLineVertex::getAttributeDescriptions();

	Pipeline::PipelineConfig config{};
	config.shaderStages = {
		{ "shaders\\debug_lines.spv", vk::ShaderStageFlagBits::eVertex, "vertMain" },
		{ "shaders\\debug_lines.spv", vk::ShaderStageFlagBits::eFragment, "fragMain" }
	};
	config.vertexBindings = { binding };
	config.vertexAttributes = { attribs.begin(), attribs.end() };
	config.pushConstantRanges = { pushRange };
	config.colorAttachmentFormats = { vk::Format::eR16G16B16A16Sfloat };
	config.depthAttachmentFormat = DepthTarget::findDepthFormat(physicalDevice);
	config.topology = vk::PrimitiveTopology::eLineList;
	config.cullMode = vk::CullModeFlagBits::eNone;
	config.depthTestEnable = true;
	config.depthWriteEnable = false;
	config.depthCompareOp = vk::CompareOp::eLessOrEqual;
	config.blendEnable = false;

	auto bundle = Pipeline::createPipeline(device, config);
	colliderDebugPipelineLayout = std::move(bundle.layout);
	colliderDebugPipeline = std::move(bundle.pipeline);
}

void Renderer::ensureColliderDebugVertexCapacity(size_t vertexCount)
{
	if (vertexCount <= colliderDebugVertexCapacity)
		return;

	size_t newCapacity = std::max<size_t>(1024, colliderDebugVertexCapacity);
	while (newCapacity < vertexCount)
		newCapacity *= 2;

	vk::raii::Buffer newBuffer({ });
	vk::raii::DeviceMemory newMemory({ });
	Buffer::createBuffer(device, physicalDevice,
		static_cast<vk::DeviceSize>(newCapacity * sizeof(DebugLineVertex)),
		vk::BufferUsageFlagBits::eVertexBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		newBuffer, newMemory);

	colliderDebugVertexBuffer = std::move(newBuffer);
	colliderDebugVertexBufferMemory = std::move(newMemory);
	colliderDebugVertexCapacity = newCapacity;
}

void Renderer::rebuildColliderDebugLines()
{
	colliderDebugVertices.clear();
	auto& registry = mEnttScene.getRegistry();

	constexpr std::array<std::pair<int, int>, 12> boxEdges = { {
		{0,1},{1,2},{2,3},{3,0},
		{4,5},{5,6},{6,7},{7,4},
		{0,4},{1,5},{2,6},{3,7}
	} };

	for (auto [entity, tr, col] : registry.view<TransformComponent, ColliderComponent>().each())
	{
		(void)entity;
		const glm::vec3 color = glm::vec3(0.15f, 1.0f, 0.15f);
		const glm::mat4 M = glm::translate(glm::mat4(1.0f), tr.GetPosition()) * glm::mat4_cast(tr.GetRotation());
		auto transformPoint = [&M](const glm::vec3& p) {
			return glm::vec3(M * glm::vec4(p, 1.0f));
			};

		if (col.shapeType == ColliderShapeType::Box)
		{
			const glm::vec3 e = glm::max(col.halfExtents, glm::vec3(0.001f));
			glm::vec3 localCenter = col.centerOffset;
			if (col.alignBottomToEntity)
				localCenter.y += e.y;
			std::array<glm::vec3, 8> p = {
				localCenter + glm::vec3(-e.x,-e.y,-e.z), localCenter + glm::vec3(e.x,-e.y,-e.z),
				localCenter + glm::vec3(e.x, e.y,-e.z), localCenter + glm::vec3(-e.x, e.y,-e.z),
				localCenter + glm::vec3(-e.x,-e.y, e.z), localCenter + glm::vec3(e.x,-e.y, e.z),
				localCenter + glm::vec3(e.x, e.y, e.z), localCenter + glm::vec3(-e.x, e.y, e.z)
			};
			for (auto& v : p)
				v = transformPoint(v);

			for (auto [a, b] : boxEdges)
				appendLine(colliderDebugVertices, p[a], p[b], color);
		}
		else if (col.shapeType == ColliderShapeType::Sphere)
		{
			const float r = std::max(col.radius, 0.001f);
			glm::vec3 localCenter = col.centerOffset;
			if (col.alignBottomToEntity)
				localCenter.y += r;
			constexpr int seg = 24;
			constexpr float twoPi = 6.28318530718f;
			for (int i = 0; i < seg; ++i)
			{
				const float a0 = (static_cast<float>(i) / static_cast<float>(seg)) * twoPi;
				const float a1 = (static_cast<float>(i + 1) / static_cast<float>(seg)) * twoPi;

				appendLine(colliderDebugVertices,
					transformPoint(localCenter + glm::vec3(std::cos(a0) * r, std::sin(a0) * r, 0.0f)),
					transformPoint(localCenter + glm::vec3(std::cos(a1) * r, std::sin(a1) * r, 0.0f)),
					color);

				appendLine(colliderDebugVertices,
					transformPoint(localCenter + glm::vec3(0.0f, std::cos(a0) * r, std::sin(a0) * r)),
					transformPoint(localCenter + glm::vec3(0.0f, std::cos(a1) * r, std::sin(a1) * r)),
					color);

				appendLine(colliderDebugVertices,
					transformPoint(localCenter + glm::vec3(std::cos(a0) * r, 0.0f, std::sin(a0) * r)),
					transformPoint(localCenter + glm::vec3(std::cos(a1) * r, 0.0f, std::sin(a1) * r)),
					color);
			}
		}
		else
		{
			// Capsule aligned to local Y (matching Jolt's CapsuleShape axis).
			const float r = std::max(col.radius, 0.001f);
			const float h = std::max(col.halfHeight, 0.001f);
			glm::vec3 localCenter = col.centerOffset;
			if (col.alignBottomToEntity)
				localCenter.y += (h + r);
			constexpr int seg = 24;
			constexpr int hemiStacks = 6;
			constexpr float twoPi = 6.28318530718f;
			constexpr float halfPi = 1.57079632679f;

			auto drawRing = [&](float y, float ringRadius) {
				for (int i = 0; i < seg; ++i)
				{
					const float a0 = (static_cast<float>(i) / static_cast<float>(seg)) * twoPi;
					const float a1 = (static_cast<float>(i + 1) / static_cast<float>(seg)) * twoPi;
					appendLine(colliderDebugVertices,
						transformPoint(localCenter + glm::vec3(std::cos(a0) * ringRadius, y, std::sin(a0) * ringRadius)),
						transformPoint(localCenter + glm::vec3(std::cos(a1) * ringRadius, y, std::sin(a1) * ringRadius)),
						color);
				}
				};

			drawRing(+h, r);
			drawRing(-h, r);

			for (int i = 0; i < 4; ++i)
			{
				const float a = static_cast<float>(i) * (halfPi);
				appendLine(colliderDebugVertices,
					transformPoint(localCenter + glm::vec3(std::cos(a) * r, +h, std::sin(a) * r)),
					transformPoint(localCenter + glm::vec3(std::cos(a) * r, -h, std::sin(a) * r)),
					color);
			}

			for (int stack = 0; stack < hemiStacks; ++stack)
			{
				const float t0 = (static_cast<float>(stack) / static_cast<float>(hemiStacks)) * halfPi;
				const float t1 = (static_cast<float>(stack + 1) / static_cast<float>(hemiStacks)) * halfPi;

				const float ringR0 = r * std::cos(t0);
				const float ringR1 = r * std::cos(t1);
				const float topY0 = h + r * std::sin(t0);
				const float topY1 = h + r * std::sin(t1);
				const float botY0 = -h - r * std::sin(t0);
				const float botY1 = -h - r * std::sin(t1);

				drawRing(topY0, ringR0);
				drawRing(botY0, ringR0);

				for (int i = 0; i < seg; i += 4)
				{
					const float a = (static_cast<float>(i) / static_cast<float>(seg)) * twoPi;

					appendLine(colliderDebugVertices,
						transformPoint(localCenter + glm::vec3(std::cos(a) * ringR0, topY0, std::sin(a) * ringR0)),
						transformPoint(localCenter + glm::vec3(std::cos(a) * ringR1, topY1, std::sin(a) * ringR1)),
						color);

					appendLine(colliderDebugVertices,
						transformPoint(localCenter + glm::vec3(std::cos(a) * ringR0, botY0, std::sin(a) * ringR0)),
						transformPoint(localCenter + glm::vec3(std::cos(a) * ringR1, botY1, std::sin(a) * ringR1)),
						color);
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

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
	bool loadedAnyFromDisk = false;

	auto assignOrLoad = [&](const std::string& texturePath,
		vk::raii::Image& image,
		vk::raii::DeviceMemory& memory,
		vk::raii::ImageView& view,
		bool isSRGB) {
			if (texturePath.empty())
				return;

			if (*view != VK_NULL_HANDLE)
				return;

			if (tryAssignCachedTextureResource(texturePath, image, memory, view))
				return;

			if (!loadedAnyFromDisk) {
				std::cout << "Loading PBR textures for material: " << material.GetName() << std::endl;
				loadedAnyFromDisk = true;
			}

			Texture::loadTextureFromFile(device, physicalDevice, queue, commandPool, texturePath, image, memory, view, isSRGB);
		};

	assignOrLoad(material.albedoTexturePath,
		textures.baseColorImage,
		textures.baseColorMemory,
		textures.baseColorView,
		true);

	assignOrLoad(material.metallicRoughnessTexturePath,
		textures.metallicRoughnessImage,
		textures.metallicRoughnessMemory,
		textures.metallicRoughnessView,
		false);

	assignOrLoad(material.normalTexturePath,
		textures.normalImage,
		textures.normalMemory,
		textures.normalView,
		false);

	assignOrLoad(material.occlusionTexturePath,
		textures.occlusionImage,
		textures.occlusionMemory,
		textures.occlusionView,
		false);

	assignOrLoad(material.emissiveTexturePath,
		textures.emissiveImage,
		textures.emissiveMemory,
		textures.emissiveView,
		true);
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

	// ---- 1+·1 white fallback texture ----
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


