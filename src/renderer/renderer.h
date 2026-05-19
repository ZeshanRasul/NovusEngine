#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#	include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "ECS/components/camera_component.h"
#include "ECS/components/animation_component.h"
#include "ECS/components/hierarchy_component.h"
#include "ECS/components/light_component.h"
#include "ECS/components/player_controller_component.h"
#include "ECS/components/transform_component.h"
#include "renderer/renderer_types.h"
#include "ECS/components/renderable_component.h"
#include "ECS/components/physics_component.h"
#include "renderer/passes/shadow_pass.h"
#include "../imgui_vulkan_util.h"
#include "../input/input_system.h"
#include "VkRenderData.h"
#include "../model/ModelAndInstanceData.h"
#include "vulkan/uniform_buffer.h"
#include "vulkan/shader_storage_buffer.h"
#include "../model/AssimpInstance.h"
#include "physics/physics_system.h"
#include "ECS/entt/scene.h"
#include "ECS/entt/assimp_instance_component.h"
#include "ECS/entt/assimp_systems.h"
#include "ECS/entt/transform_systems.h"
#include "../gameplay/gameplay_layer.h"
#include "../gameplay/gameplay_runtime.h"
#include "../core/scene_runtime_service.h"

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyActivationListener.h>
#include <ECS/jolt_physics/physics_world.h>

enum SceneState {
	PLAY,
	EDIT
};

namespace Layers
{
	static constexpr JPH::ObjectLayer NON_MOVING = 0;
	static constexpr JPH::ObjectLayer MOVING = 1;
	static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter
{
public:
	virtual bool					ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
	{
		switch (inObject1)
		{
		case Layers::NON_MOVING:
			return inObject2 == Layers::MOVING; // Non moving only collides with moving
		case Layers::MOVING:
			return true; // Moving collides with everything
		default:
			JPH_ASSERT(false);
			return false;
		}
	}
};

namespace BroadPhaseLayers
{
	static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
	static constexpr JPH::BroadPhaseLayer MOVING(1);
	static constexpr JPH::uint NUM_LAYERS(2);
};

class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
public:
	BPLayerInterfaceImpl()
	{
		// Create a mapping table from object to broad phase layer
		mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
		mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
	}

	virtual JPH::uint					GetNumBroadPhaseLayers() const override
	{
		return BroadPhaseLayers::NUM_LAYERS;
	}

	virtual JPH::BroadPhaseLayer			GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
	{
		JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
		return mObjectToBroadPhase[inLayer];
	}

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
	virtual const char* GetBroadPhaseLayerName(BroadPhaseLayer inLayer) const override
	{
		switch ((BroadPhaseLayer::Type)inLayer)
		{
		case (BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:	return "NON_MOVING";
		case (BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:		return "MOVING";
		default:													JPH_ASSERT(false); return "INVALID";
		}
	}
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
	JPH::BroadPhaseLayer					mObjectToBroadPhase[Layers::NUM_LAYERS];
};

class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
	virtual bool				ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
	{
		switch (inLayer1)
		{
		case Layers::NON_MOVING:
			return inLayer2 == BroadPhaseLayers::MOVING;
		case Layers::MOVING:
			return true;
		default:
			JPH_ASSERT(false);
			return false;
		}
	}
};

constexpr uint32_t WIDTH = 3840;
constexpr uint32_t HEIGHT = 2160;
constexpr int      MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class Renderer
{
public:
	Renderer() = default;
	~Renderer() = default;

	void run();

private:
	// -------------------------------------------------------------------------
	// Initialisation / shutdown
	// -------------------------------------------------------------------------
	void initWindow();
	void initVulkan();
	void mainLoop();
	void cleanup();

	// -------------------------------------------------------------------------
	// Vulkan setup
	// -------------------------------------------------------------------------
	void                     createInstance();
	void                     setupDebugMessenger();
	void                     createSurface();
	bool                     isDeviceSuitable(vk::raii::PhysicalDevice const& pd);
	void                     pickPhysicalDevice();
	void                     createLogicalDevice();
	std::vector<const char*> getRequiredInstanceExtensions();

	bool deviceInit();

	void recreateSwapChain();

	// Shaders / pipelines
	void createGraphicsPipeline();
	bool createPBRPipeline();

	// Commands
	void                    recordCommandBuffer(uint32_t imageIndex);
	void                    beginMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
	void                    recordShadowPass(vk::raii::CommandBuffer& commandBuffer, uint32_t cascadeIndex);
	void                    recordScenePass(vk::raii::CommandBuffer& commandBuffer);
	void                    recordImguiPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
	void                    endMainPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

	// Buffers
	void     copyBuffer(vk::raii::Buffer& src, vk::raii::Buffer& dst, vk::DeviceSize size);
	void     createVertexBuffer(RenderableComponent& gameObj);
	void     createIndexBuffer(RenderableComponent& gameObj);

	// Images
	void transition_image_layout(vk::Image image,
		vk::ImageLayout old_layout, vk::ImageLayout new_layout,
		vk::AccessFlags2 src_access_mask, vk::AccessFlags2 dst_access_mask,
		vk::PipelineStageFlags2 src_stage_mask, vk::PipelineStageFlags2 dst_stage_mask,
		vk::ImageAspectFlags image_aspect_flags);

	// Textures / samplers
	void loadPBRTextures(const Material& material, RenderableComponent::PBRTextures& textures);
	void createDefaultTextures();
	void createTextureSampler();

	// Frame
	void drawFrame();
	void rebuildRenderableRuntimeResources();

	// Scene
	void setupGameObjects();
	void setupPhysicsShowcase();

	// Callbacks
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
	static void mousePosCallback(GLFWwindow* window, double xpos, double ypos);
	static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
	static VKAPI_ATTR vk::Bool32 VKAPI_CALL
		debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
			vk::DebugUtilsMessageTypeFlagsEXT type,
			const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*);

	// ImGUI
	void renderImgui();
	void buildEditorDockspace();
	void renderViewportPanel();
	void renderCameraControlsPanel();
	void renderPrefabPanel(bool isEditMode, entt::registry& registry);
	void renderPlayModePanel(bool isEditMode);
	void renderShadowTuningPanel(bool isEditMode);
	void renderAnimationControlsPanel(bool isEditMode);
	void renderPlayHudPanel(bool isEditMode, entt::registry& registry);
	void renderPostProcessingAndPhysicsPanels(bool isEditMode, entt::registry& registry);
	void initEnttDemoScene();
	void renderEnttEditor(glm::mat4 view, glm::mat4 projection);
	std::string serializeEnttScene() const;
	bool deserializeEnttScene(const std::string& sceneJson);
	bool saveSelectedAsPrefab(const std::string& prefabPath);
	bool instantiatePrefab(const std::string& prefabPath);
	void refreshPrefabAssetList();
	std::shared_ptr<AssimpModel> ensureModelLoadedForScene(const std::string& modelFileName);
	void pushUndoSnapshot();
	void performUndo();
	void performRedo();
	void enterPlayMode();
	void exitPlayMode();
	Gameplay::RuntimeContext buildGameplayRuntimeContext();

	// -------------------------------------------------------------------------
	// Members
	// -------------------------------------------------------------------------
	GLFWwindow* window = nullptr;

	vk::raii::Context                context;
	vk::raii::Instance               instance = nullptr;
	vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
	vk::raii::PhysicalDevice         physicalDevice = nullptr;
	vk::raii::Device                 device = nullptr;
	vk::raii::Queue                  queue = nullptr;
	uint32_t                         queueIndex = ~0u;

	vk::raii::SurfaceKHR             surface = nullptr;
	vk::raii::SwapchainKHR           swapChain = nullptr;
	std::vector<vk::Image>           swapChainImages;
	vk::Extent2D                     swapChainExtent;
	vk::SurfaceFormatKHR             swapChainSurfaceFormat;
	std::vector<vk::raii::ImageView> swapChainImageViews;
	bool                             framebufferResized = false;

	vk::raii::DescriptorPool             descriptorPool = nullptr;
	vk::raii::DescriptorSetLayout        descriptorSetLayout = nullptr;
	std::vector<vk::raii::DescriptorSet> descriptorSets;

	vk::raii::PipelineLayout pipelineLayout = nullptr;
	vk::raii::Pipeline       graphicsPipeline = nullptr;

	vk::raii::PipelineLayout pbrPipelineLayout = nullptr;
	vk::raii::Pipeline       pbrPipeline = nullptr;
	vk::raii::PipelineLayout shadowPipelineLayout = nullptr;
	vk::raii::Pipeline       shadowPipeline = nullptr;
	vk::raii::PipelineLayout shadowSkinningPipelineLayout = nullptr;
	vk::raii::Pipeline       shadowSkinningPipeline = nullptr;

	vk::raii::CommandPool                commandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> commandBuffers;

	std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
	std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
	std::vector<vk::raii::Fence>     inFlightFences;
	uint32_t                         frameIndex = 0;

	vk::raii::Buffer       vertexBuffer = nullptr;
	vk::raii::DeviceMemory vertexBufferMemory = nullptr;
	vk::raii::Buffer       indexBuffer = nullptr;
	vk::raii::DeviceMemory indexBufferMemory = nullptr;

	std::vector<vk::raii::Buffer>       uniformBuffers;
	std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
	std::vector<void*>                  uniformBuffersMapped;

	vk::raii::Image        textureImage = nullptr;
	vk::raii::DeviceMemory textureImageMemory = nullptr;
	vk::raii::ImageView    textureImageView = nullptr;
	vk::raii::Sampler      textureSampler = nullptr;

	vk::raii::Image        depthImage = nullptr;
	vk::raii::DeviceMemory depthImageMemory = nullptr;
	vk::raii::ImageView    depthImageView = nullptr;

	vk::raii::Image        defaultTextureImage = nullptr;
	vk::raii::DeviceMemory defaultTextureMemory = nullptr;
	vk::raii::ImageView    defaultTextureView = nullptr;

	vk::raii::Image        defaultNormalImage = nullptr;
	vk::raii::DeviceMemory defaultNormalMemory = nullptr;
	vk::raii::ImageView    defaultNormalView = nullptr;

	std::array<vk::raii::Image, SHADOW_CASCADE_COUNT> shadowImages = { nullptr, nullptr, nullptr, nullptr, nullptr };
	std::array<vk::raii::DeviceMemory, SHADOW_CASCADE_COUNT> shadowImageMemories = { nullptr, nullptr, nullptr, nullptr, nullptr };
	std::array<vk::raii::ImageView, SHADOW_CASCADE_COUNT> shadowImageViews = { nullptr, nullptr, nullptr, nullptr, nullptr };
	std::array<vk::ImageLayout, SHADOW_CASCADE_COUNT> shadowImageLayouts = { vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined, vk::ImageLayout::eUndefined };
	vk::raii::Sampler                    shadowSampler = nullptr;
	vk::raii::DescriptorSetLayout        shadowDescriptorSetLayout = nullptr;

	std::vector<const char*> requiredDeviceExtension = { vk::KHRSwapchainExtensionName };

	Camera camera;
	ShadowSettings shadowSettings;
	PhysicsSystem physicsSystem;
	bool physicsPaused = false;
	int physicsSpawnCount = 2;
	float physicsSpawnHeight = -400.0f;
	float  lastFrameTime = 0.0f;

	ImGuiVulkanUtil* imGui;
	ImTextureID mViewportTextureId = 0;
	bool mEditorDockLayoutInitialized = false;
	bool mViewportFocused = false;
	bool mViewportHovered = false;

	VkRenderData mRenderData{};
	ModelAndInstanceData mModelInstData{};
	std::vector<AssimpInstanceGPUData> mAssimpGPUData{};
	EnttScene mEnttScene{};
	entt::entity mEnttSelectedEntity = entt::null;
	std::vector<entt::entity> mEnttMultiSelection{};
	SceneRuntimeService mSceneRuntimeService{};
	bool mLogPlayToEditCacheStats = false;

	std::string normalizeModelAssetKey(const std::string& modelFileName) const;
	std::shared_ptr<AssimpModel> getOrLoadModelAssimpAsset(const std::string& modelFileName);
	void ensureModelInSceneList(const std::shared_ptr<AssimpModel>& model);

	std::unordered_map<std::string, std::shared_ptr<AssimpModel>> mModelAssetCache;

	std::shared_ptr<RenderableComponent> getOrLoadModelGltfAsset(const std::string& modelFileName);
	void ensureGltfModelInSceneList(const std::shared_ptr<RenderableComponent>& model);
	bool tryReuseCachedGltfTextures(const std::string& modelFileName, RenderableComponent& renderable);
	void captureGltfTexturesFromScene(entt::registry& registry);
	struct CachedTextureResource {
		vk::raii::Image image = nullptr;
		vk::raii::DeviceMemory memory = nullptr;
		vk::raii::ImageView view = nullptr;
	};
	void cacheTextureResource(const std::string& texturePath, vk::raii::Image& image, vk::raii::DeviceMemory& memory, vk::raii::ImageView& view);
	bool tryAssignCachedTextureResource(const std::string& texturePath, vk::raii::Image& image, vk::raii::DeviceMemory& memory, vk::raii::ImageView& view);

	std::unordered_map<std::string, std::shared_ptr<RenderableComponent>> mGltfModelAssetCache;
	std::vector<std::shared_ptr<RenderableComponent>> mSceneRenderableModels;
	std::unordered_map<std::string, std::deque<std::vector<RenderableComponent::PBRTextures>>> mGltfModelTextureCache;
	std::unordered_map<std::string, std::deque<CachedTextureResource>> mTextureAssetCache;
	std::shared_ptr<RenderableComponent> getRenderableModel(std::string modelFileName);

	bool hasModel(std::string modelFileName);
	std::shared_ptr<AssimpModel> getModel(std::string modelFileName);
	bool addModel(std::string modelFileName);
	void deleteModel(std::string modelFileName);

	std::shared_ptr<AssimpInstance> addInstance(std::shared_ptr<AssimpModel> model);
	void addInstances(std::shared_ptr<AssimpModel> model, int numInstances);
	void deleteInstance(std::shared_ptr<AssimpInstance> instance);
	void cloneInstance(std::shared_ptr<AssimpInstance> instance);
	void updateTriangleCount();
	entt::entity createAssimpEnttEntity(const std::shared_ptr<AssimpInstance>& instance, const std::string& namePrefix = "Assimp: ");
	void destroyAssimpEnttEntity(const std::shared_ptr<AssimpInstance>& instance);

	// Assimp / skinning pipeline
	void initAssimpRenderData();
	void createSkinningPipeline();
	void createAssimpInstanceGPUData(std::shared_ptr<AssimpInstance> instance);
	void deleteAssimpInstanceGPUData(std::shared_ptr<AssimpInstance> instance);
	void createAssimpInstanceForEntity(std::shared_ptr<AssimpModel> model, entt::entity entity);
	void removeInstanceFromEntity(entt::entity entity);
	void onAssimpInstanceDestroyed(AssimpInstance* rawPtr);
	void onAssimpInstanceComponentDestroyed(entt::registry& reg, entt::entity e);
	void updateDescriptorSets();
	void updateComputeDescriptorSets();
	void initComputeSkinningResources();
	void ensureComputeModelResources(const std::shared_ptr<AssimpModel>& model);
	void runComputeShaders(const std::shared_ptr<AssimpModel>& model, size_t numberOfInstances, uint32_t modelOffset);
	void updateAssimpAnimations(float deltaTime);
	void recordAssimpSkinnedPass(vk::raii::CommandBuffer& commandBuffer);
	void recordAssimpShadowPass(vk::raii::CommandBuffer& commandBuffer, uint32_t cascadeIndex);
	void recordSceneCopyPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);

	struct ComputeModelResources {
		VkShaderStorageBufferData parentIndexBuffer{};
		VkShaderStorageBufferData boneOffsetBuffer{};
		vk::raii::DescriptorSet set1Descriptor = nullptr;
		uint32_t boneCount = 0;
	};

	vk::raii::DescriptorSetLayout skinningDescriptorSetLayout = nullptr;
	vk::raii::PipelineLayout      skinningPipelineLayout = nullptr;
	vk::raii::Pipeline            skinningPipeline = nullptr;
	vk::raii::DescriptorPool      skinningDescriptorPool = nullptr;
	vk::raii::Sampler             skinningSampler = nullptr;
	vk::raii::Image               skinningWhiteImage = nullptr;
	vk::raii::DeviceMemory        skinningWhiteMemory = nullptr;
	vk::raii::ImageView           skinningWhiteView = nullptr;
	bool                          mCleanupDone = false;
	int mManyInstanceCreateNum = 1;
	std::vector<glm::mat4> mWorldPosMatrices{};
	std::vector<NodeTransformData> mNodeTransFormData{};
	VkShaderStorageBufferData mShaderNodeTransformBuffer{};
	VkShaderStorageBufferData mShaderTRSMatrixBuffer{};
	VkShaderStorageBufferData mShaderBoneMatrixBuffer{};
	vk::raii::DescriptorSetLayout mComputeSetLayout0 = nullptr;
	vk::raii::DescriptorSetLayout mComputeSetLayout1 = nullptr;
	vk::raii::DescriptorPool mComputeDescriptorPool = nullptr;
	vk::raii::DescriptorSet mComputeTrsSet0 = nullptr;
	vk::raii::DescriptorSet mComputeBoneSet0 = nullptr;
	vk::raii::PipelineLayout mComputeTrsPipelineLayout = nullptr;
	vk::raii::Pipeline mComputeTrsPipeline = nullptr;
	vk::raii::PipelineLayout mComputeBonePipelineLayout = nullptr;
	vk::raii::Pipeline mComputeBonePipeline = nullptr;
	vk::raii::CommandBuffer mComputeCommandBuffer = nullptr;
	bool mComputeSkinningEnabled = false;
	std::unordered_map<std::string, ComputeModelResources> mComputeModelResources{};
	std::unordered_map<AssimpInstance*, uint32_t> mInstanceBoneOffsets{};


	// FXAA Pass
	vk::raii::Image 	fxaaImage = nullptr;
	vk::raii::DeviceMemory fxaaImageMemory = nullptr;
	vk::raii::ImageView fxaaImageView = nullptr;
	vk::raii::Image viewportPreviewImage = nullptr;
	vk::raii::DeviceMemory viewportPreviewImageMemory = nullptr;
	vk::raii::ImageView viewportPreviewImageView = nullptr;
	vk::ImageLayout viewportPreviewImageLayout = vk::ImageLayout::eUndefined;
	vk::raii::DescriptorSetLayout fxaaDescriptorSetLayout = nullptr;
	std::vector<vk::raii::DescriptorSet> fxaaDescriptorSets;
	vk::raii::DescriptorPool fxaaDescriptorPool = nullptr;
	vk::raii::PipelineLayout fxaaPipelineLayout = nullptr;
	vk::raii::Pipeline fxaaPipeline = nullptr;
	vk::raii::Sampler fxaaSampler = nullptr;
	float fxaaExposure = 1.0f;
	float fxaaGamma = 2.2f;

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

	void createFxaaSampler();
	void createBloomResources();
	void createBloomPipelines();
	void createBloomDescriptorSets();
	void recordBloomPasses(vk::raii::CommandBuffer& commandBuffer);
	void recordFxaaPass(vk::raii::CommandBuffer& commandBuffer, uint32_t imageIndex);
	void createFxaaPipeline();

	vk::Format bloomFormat = vk::Format::eR16G16B16A16Sfloat;
	vk::Extent2D bloomExtent{};
	vk::raii::Image bloomImageA = nullptr;
	vk::raii::DeviceMemory bloomImageAMemory = nullptr;
	vk::raii::ImageView bloomImageAView = nullptr;
	vk::raii::Image bloomImageB = nullptr;
	vk::raii::DeviceMemory bloomImageBMemory = nullptr;
	vk::raii::ImageView bloomImageBView = nullptr;
	vk::ImageLayout bloomImageALayout = vk::ImageLayout::eUndefined;
	vk::ImageLayout bloomImageBLayout = vk::ImageLayout::eUndefined;

	vk::raii::DescriptorSetLayout bloomExtractDescriptorSetLayout = nullptr;
	vk::raii::DescriptorSetLayout bloomBlurDescriptorSetLayout = nullptr;
	vk::raii::DescriptorPool bloomDescriptorPool = nullptr;
	std::vector<vk::raii::DescriptorSet> bloomExtractDescriptorSets;
	std::vector<vk::raii::DescriptorSet> bloomBlurFromADescriptorSets;
	std::vector<vk::raii::DescriptorSet> bloomBlurFromBDescriptorSets;

	vk::raii::PipelineLayout bloomExtractPipelineLayout = nullptr;
	vk::raii::Pipeline bloomExtractPipeline = nullptr;
	vk::raii::PipelineLayout bloomBlurPipelineLayout = nullptr;
	vk::raii::Pipeline bloomBlurPipeline = nullptr;

	bool bloomEnabled = true;
	float bloomThreshold = 0.13f;
	float bloomSoftKnee = 0.5f;
	float bloomPrefilterScale = 2.0f;
	float bloomIntensity = 0.7f;
	float bloomBlurScale = 1.0f;
	int bloomBlurPasses = 2;
	int postProcessDebugMode = 0;



	//IMGUIZMO

	ImGuizmo::OPERATION currentOperation = ImGuizmo::OPERATION::TRANSLATE;
	ImGuizmo::MODE currentMode = ImGuizmo::MODE::LOCAL;

	// JOLT
	PhysicsWorld* physicsWorld;
	JPH::BodyID sphereBodyID;
	JPH::uint step = 0;

	JPH::JobSystemThreadPool jobSystem;
	BPLayerInterfaceImpl broad_phase_layer_interface;
	ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
	ObjectLayerPairFilterImpl object_vs_object_layer_filter;

	void createColliderDebugPipeline();
	void recordColliderDebugPass(vk::raii::CommandBuffer& commandBuffer);
	void rebuildColliderDebugLines();
	void ensureColliderDebugVertexCapacity(size_t vertexCount);

	struct DebugLineVertex
	{
		glm::vec3 position;
		glm::vec3 color;

		static vk::VertexInputBindingDescription getBindingDescription()
		{
			return { .binding = 0, .stride = sizeof(DebugLineVertex), .inputRate = vk::VertexInputRate::eVertex };
		}

		static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
		{
			return {
				vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(DebugLineVertex, position)),
				vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(DebugLineVertex, color))
			};
		}
	};


	struct DebugLinePushConstants
	{
		glm::mat4 viewProj{ 1.0f };
	};

	static void appendLine(std::vector<Renderer::DebugLineVertex>& out,
		const glm::vec3& a, const glm::vec3& b, const glm::vec3& color)
	{
		out.push_back({ a, color });
		out.push_back({ b, color });
	};

	vk::raii::PipelineLayout colliderDebugPipelineLayout = nullptr;
	vk::raii::Pipeline colliderDebugPipeline = nullptr;
	vk::raii::Buffer colliderDebugVertexBuffer = nullptr;
	vk::raii::DeviceMemory colliderDebugVertexBufferMemory = nullptr;
	size_t colliderDebugVertexCapacity = 0;
	std::vector<DebugLineVertex> colliderDebugVertices{};
	bool physicsDrawColliderDebug = true;

	SceneState sceneState = SceneState::EDIT;
	int currentFrameIndex = 0;
	bool playShowDebugUI = false;

	bool uiShowViewport = true;
	bool uiShowCameraControls = true;
	bool uiShowPlayHud = true;
	bool uiShowPostProcessingWindow = true;
	bool uiShowShadowTuningWindow = true;
	bool uiShowPhysicsWindow = true;
	bool uiShowPrefabWindow = true;
	bool uiShowGpuTimingsWindow = true;

	bool renderEnableShadows = true;
	bool renderEnablePostProcessing = true;
	bool renderEnableFxaa = true;
	bool renderEnableBloom = true;

	void saveRenderPreset();
	void loadRenderPreset();

	// GPU timestamp queries
	void createTimestampQueryPool();
	void readTimestamps();
	void renderGpuTimingsPanel(bool isEditMode);

	vk::raii::QueryPool mTimestampQueryPool = nullptr;
	float               mTimestampPeriod    = 1.0f;
	bool                mTimestampsSupported = false;
	GpuTimings          mGpuTimings{};

	Gameplay::GameplayRuntime mGameplayRuntime{};
};
