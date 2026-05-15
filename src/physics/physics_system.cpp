#include "physics/physics_system.h"

#include "ECS/components/physics_component.h"
#include "ECS/components/transform_component.h"
#include <Jolt/RegisterTypes.h>
#include <thread>

bool PhysicsSystem::ObjectLayerPairFilterImpl::ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const
{
    constexpr JPH::ObjectLayer nonMoving = 0;
    constexpr JPH::ObjectLayer moving = 1;

    if (inObject1 == nonMoving)
        return inObject2 == moving;
    if (inObject1 == moving)
        return true;
    return false;
}

PhysicsSystem::BPLayerInterfaceImpl::BPLayerInterfaceImpl()
{
    mObjectToBroadPhase[0] = JPH::BroadPhaseLayer(0);
    mObjectToBroadPhase[1] = JPH::BroadPhaseLayer(1);
}

JPH::uint PhysicsSystem::BPLayerInterfaceImpl::GetNumBroadPhaseLayers() const
{
    return 2;
}

JPH::BroadPhaseLayer PhysicsSystem::BPLayerInterfaceImpl::GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const
{
    JPH_ASSERT(inLayer < 2);
    return mObjectToBroadPhase[inLayer];
}

bool PhysicsSystem::ObjectVsBroadPhaseLayerFilterImpl::ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const
{
    constexpr JPH::ObjectLayer nonMoving = 0;
    constexpr JPH::BroadPhaseLayer moving(1);

    if (inLayer1 == nonMoving)
        return inLayer2 == moving;
    return true;
}

void PhysicsSystem::applyContextSettings()
{
    context.physicsSystem = &physicsSystem;
    context.bodyInterface = bodyInterface;
    context.tempAllocator = tempAllocator;
    context.jobSystem = jobSystem;
    context.layers = layerConfig;
    context.paused = paused;
    context.gravity = gravity;
}

void PhysicsSystem::initialize()
{
    if (initialized)
        return;

    JPH::RegisterDefaultAllocator();
    if (JPH::Factory::sInstance == nullptr)
    {
        JPH::Factory::sInstance = new JPH::Factory();
        ownsJoltFactory = true;
        JPH::RegisterTypes();
    }

    constexpr uint32_t maxBodies = 8192;
    constexpr uint32_t numBodyMutexes = 0;
    constexpr uint32_t maxBodyPairs = 8192;
    constexpr uint32_t maxContactConstraints = 8192;

    tempAllocator = new JPH::TempAllocatorImpl(10 * 1024 * 1024);
    jobSystem = new JPH::JobSystemThreadPool(
        JPH::cMaxPhysicsJobs,
        JPH::cMaxPhysicsBarriers,
        std::max(1u, std::thread::hardware_concurrency() - 1));

    physicsSystem.Init(
        maxBodies,
        numBodyMutexes,
        maxBodyPairs,
        maxContactConstraints,
        broadPhaseLayerInterface,
        objectVsBroadPhaseLayerFilter,
        objectLayerPairFilter);

    bodyInterface = &physicsSystem.GetBodyInterface();
    layerConfig.nonMoving = 0;
    layerConfig.moving = 1;

    paused = false;
    applyContextSettings();
    PhysicsSystems::SetGravity(context, gravity);
    initialized = true;
}

void PhysicsSystem::shutdown()
{
    if (!initialized)
        return;

    clear();

    delete jobSystem;
    jobSystem = nullptr;

    delete tempAllocator;
    tempAllocator = nullptr;

    bodyInterface = nullptr;

    if (ownsJoltFactory)
    {
        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
        ownsJoltFactory = false;
    }

    initialized = false;
}

void PhysicsSystem::setPaused(bool value)
{
    paused = value;
    PhysicsSystems::SetPaused(context, value);
}

bool PhysicsSystem::isPaused() const
{
    return PhysicsSystems::IsPaused(context);
}

void PhysicsSystem::setGravity(glm::vec3 const& g)
{
    gravity = g;
    PhysicsSystems::SetGravity(context, g);
}

glm::vec3 const& PhysicsSystem::getGravity() const
{
    return gravity;
}

void PhysicsSystem::bindRegistry(entt::registry& registry)
{
    boundRegistry = &registry;
}

void PhysicsSystem::registerEntity(entt::entity& entity, entt::registry& registry)
{
    if (!initialized || entity == entt::null)
        return;

    auto* legacy = registry.try_get<PhysicsComponent>(entity);

    if (!registry.any_of<RigidBodyComponent>(entity))
        registry.emplace<RigidBodyComponent>(entity);
    if (!registry.any_of<ColliderComponent>(entity))
        registry.emplace<ColliderComponent>(entity);

    auto* rb = registry.try_get<RigidBodyComponent>(entity);
    auto* col = registry.try_get<ColliderComponent>(entity);
    if (!rb || !col)
        return;

    // Legacy compatibility: if PhysicsComponent alias data exists, mirror values.
    if (legacy)
    {
        rb->bodyType = legacy->bodyType;
        rb->mass = legacy->mass;
        rb->friction = legacy->friction;
        rb->restitution = legacy->restitution;
        rb->useGravity = legacy->useGravity;
        rb->linearVelocity = legacy->getLinearVelocity();

        col->shapeType = legacy->shapeType;
        col->halfExtents = legacy->halfExtents;
        col->radius = legacy->radius;
    }

    PhysicsSystems::RegisterEntity(context, registry, entity);

    auto bodyIt = context.entityToBody.find(entity);
    if (bodyIt != context.entityToBody.end())
    {
        rb->bodyId = static_cast<int>(bodyIt->second.GetIndex());
        rb->registeredInWorld = true;
    }

    boundRegistry = &registry;
}

void PhysicsSystem::unregisterEntity(entt::entity& entity, entt::registry& registry)
{
    if (!initialized || entity == entt::null)
        return;

    PhysicsSystems::UnregisterEntity(context, registry, entity);

    if (auto* rb = registry.try_get<RigidBodyComponent>(entity))
    {
        rb->bodyId = -1;
        rb->registeredInWorld = false;
    }
}

void PhysicsSystem::clear()
{
    if (!initialized)
        return;

    if (boundRegistry)
        PhysicsSystems::Clear(context, *boundRegistry);
    else
    {
        context.entityToBody.clear();
        context.bodyToEntity.clear();
    }
}

void PhysicsSystem::reset()
{
    if (initialized && boundRegistry)
        PhysicsSystems::Reset(context, *boundRegistry);
}

void PhysicsSystem::step(float deltaTime)
{
    if (boundRegistry)
        step(deltaTime, *boundRegistry);
}

void PhysicsSystem::step(float deltaTime, entt::registry& registry)
{
    if (!initialized)
        return;

    boundRegistry = &registry;
    applyContextSettings();
    PhysicsSystems::Update(context, registry, deltaTime);
}
