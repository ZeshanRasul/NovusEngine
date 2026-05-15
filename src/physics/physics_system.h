#pragma once

#include <glm/glm.hpp>

#include <entt/entt.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include "ECS/entt/physics_systems.h"

class PhysicsSystem
{
public:
    void initialize();
    void shutdown();

    void setPaused(bool paused);
    bool isPaused() const;

    void setGravity(glm::vec3 const& gravity);
    glm::vec3 const& getGravity() const;

    void registerEntity(entt::entity& entity, entt::registry& registry);
    void unregisterEntity(entt::entity& entity, entt::registry& registry);
    void clear();

    // Resets all dynamic bodies to their original spawn pose and zero velocity.
    void reset();

    void step(float deltaTime);
    void step(float deltaTime, entt::registry& registry);
    void bindRegistry(entt::registry& registry);

private:
    struct ObjectLayerPairFilterImpl final : public JPH::ObjectLayerPairFilter
    {
        virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override;
    };

    struct BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
    {
        BPLayerInterfaceImpl();
        virtual JPH::uint GetNumBroadPhaseLayers() const override;
        virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override;

    private:
        JPH::BroadPhaseLayer mObjectToBroadPhase[2];
    };

    struct ObjectVsBroadPhaseLayerFilterImpl final : public JPH::ObjectVsBroadPhaseLayerFilter
    {
        virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override;
    };

    bool initialized = false;
    bool ownsJoltFactory = false;
    glm::vec3 gravity = glm::vec3(0.0f, 9.81f, 0.0f);
    bool paused = false;
    entt::registry* boundRegistry = nullptr;

    JPH::TempAllocatorImpl* tempAllocator = nullptr;
    JPH::JobSystemThreadPool* jobSystem = nullptr;
    JPH::PhysicsSystem physicsSystem;
    JPH::BodyInterface* bodyInterface = nullptr;

    PhysicsSystems::Context context{};
    PhysicsSystems::LayerConfig layerConfig{};

    BPLayerInterfaceImpl broadPhaseLayerInterface;
    ObjectVsBroadPhaseLayerFilterImpl objectVsBroadPhaseLayerFilter;
    ObjectLayerPairFilterImpl objectLayerPairFilter;

    void applyContextSettings();
};
