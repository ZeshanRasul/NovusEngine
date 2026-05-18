#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <entt/entt.hpp>
#include <glm/glm.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/SphereShape.h>
#include <Jolt/Physics/PhysicsSystem.h>

#include "../components/transform_component.h"
#include "physics_components.h"

namespace PhysicsSystems
{
    struct LayerConfig
    {
        JPH::ObjectLayer nonMoving = 0;
        JPH::ObjectLayer moving = 1;
    };

    struct Context
    {
        struct BodyIDHash
        {
            std::size_t operator()(const JPH::BodyID& bodyId) const noexcept
            {
                return static_cast<std::size_t>(bodyId.GetIndexAndSequenceNumber());
            }
        };

        JPH::PhysicsSystem* physicsSystem = nullptr;
        JPH::BodyInterface* bodyInterface = nullptr;
        JPH::TempAllocator* tempAllocator = nullptr;
        JPH::JobSystem* jobSystem = nullptr;

        LayerConfig layers{};

        bool paused = false;
        glm::vec3 gravity = glm::vec3(0.0f, 9.0f, 0.0f);

        std::unordered_map<entt::entity, JPH::BodyID> entityToBody{};
        std::unordered_map<JPH::BodyID, entt::entity, BodyIDHash> bodyToEntity{};
    };

    inline JPH::EMotionType ToMotionType(RigidBodyType bodyType)
    {
        switch (bodyType)
        {
        case RigidBodyType::Static: return JPH::EMotionType::Static;
        case RigidBodyType::Kinematic: return JPH::EMotionType::Kinematic;
        case RigidBodyType::Dynamic:
        default:
            return JPH::EMotionType::Dynamic;
        }
    }

    inline float GetBottomOffsetY(const ColliderComponent& collider)
    {
        switch (collider.shapeType)
        {
        case ColliderShapeType::Sphere:
            return std::max(collider.radius, 0.001f);
        case ColliderShapeType::Capsule:
            return std::max(collider.halfHeight, 0.001f) + std::max(collider.radius, 0.001f);
        case ColliderShapeType::Box:
        default:
            return std::max(collider.halfExtents.y, 0.001f);
        }
    }

    inline glm::vec3 GetColliderCenterOffset(const ColliderComponent& collider)
    {
        glm::vec3 offset = collider.centerOffset;
        if (collider.alignBottomToEntity)
            offset.y += GetBottomOffsetY(collider);
        return offset;
    }

    inline JPH::ObjectLayer ToObjectLayer(const LayerConfig& layers, RigidBodyType bodyType)
    {
        return bodyType == RigidBodyType::Static ? layers.nonMoving : layers.moving;
    }

    inline JPH::ShapeRefC CreateShape(const ColliderComponent& collider)
    {
        switch (collider.shapeType)
        {
        case ColliderShapeType::Sphere:
            return new JPH::SphereShape(std::max(collider.radius, 0.001f));
        case ColliderShapeType::Capsule:
            return new JPH::CapsuleShape(std::max(collider.halfHeight, 0.001f), std::max(collider.radius, 0.001f));
        case ColliderShapeType::Box:
        default:
            return new JPH::BoxShape(JPH::Vec3(
                std::max(collider.halfExtents.x, 0.001f),
                std::max(collider.halfExtents.y, 0.001f),
                std::max(collider.halfExtents.z, 0.001f)));
        }
    }

    inline JPH::Vec3 ToJoltVec3(const glm::vec3& value)
    {
        return JPH::Vec3(value.x, value.y, value.z);
    }

    inline JPH::RVec3 ToJoltRVec3(const glm::vec3& value)
    {
        return JPH::RVec3(value.x, value.y, value.z);
    }

    inline JPH::Quat ToJoltQuat(const glm::quat& value)
    {
        return JPH::Quat(value.x, value.y, value.z, value.w);
    }

    inline glm::vec3 ToGlmFromRVec3(const JPH::RVec3& value)
    {
        return glm::vec3(
            static_cast<float>(value.GetX()),
            static_cast<float>(value.GetY()),
            static_cast<float>(value.GetZ()));
    }

    inline glm::vec3 ToGlmVec3(const JPH::Vec3& value)
    {
        return glm::vec3(value.GetX(), value.GetY(), value.GetZ());
    }

    inline glm::quat ToGlmQuat(const JPH::Quat& value)
    {
        return glm::quat(value.GetW(), value.GetX(), value.GetY(), value.GetZ());
    }

    inline void SetPaused(Context& context, bool paused)
    {
        context.paused = paused;
    }

    inline bool IsPaused(const Context& context)
    {
        return context.paused;
    }

    inline void SetGravity(Context& context, const glm::vec3& gravity)
    {
        context.gravity = gravity;
        if (context.physicsSystem)
            context.physicsSystem->SetGravity(ToJoltVec3(gravity));
    }

    inline glm::vec3 GetGravity(const Context& context)
    {
        return context.gravity;
    }

    inline void RegisterEntity(Context& context, entt::registry& registry, entt::entity entity)
    {
        if (!context.bodyInterface || !registry.valid(entity))
            return;

        auto [transform, rigidBody, collider] = registry.try_get<TransformComponent, RigidBodyComponent, ColliderComponent>(entity);
        if (!transform || !rigidBody || !collider || rigidBody->registeredInWorld)
            return;

        JPH::ShapeRefC shape = CreateShape(*collider);
        const glm::vec3 centerOffsetWorld = transform->GetRotation() * GetColliderCenterOffset(*collider);
        JPH::BodyCreationSettings settings(
            shape,
            ToJoltRVec3(transform->GetPosition() + centerOffsetWorld),
            ToJoltQuat(transform->GetRotation()),
            ToMotionType(rigidBody->bodyType),
            ToObjectLayer(context.layers, rigidBody->bodyType));

        settings.mFriction = rigidBody->friction;
        settings.mRestitution = rigidBody->restitution;
        settings.mAllowDynamicOrKinematic = rigidBody->bodyType != RigidBodyType::Static;

        if (rigidBody->bodyType == RigidBodyType::Dynamic)
            settings.mMassPropertiesOverride.mMass = std::max(rigidBody->mass, 0.001f);

        const JPH::EActivation activation = rigidBody->bodyType == RigidBodyType::Static ?
            JPH::EActivation::DontActivate :
            JPH::EActivation::Activate;

        const JPH::BodyID bodyId = context.bodyInterface->CreateAndAddBody(settings, activation);

        rigidBody->registeredInWorld = true;
        context.entityToBody[entity] = bodyId;
        context.bodyToEntity[bodyId] = entity;

        context.bodyInterface->SetLinearVelocity(bodyId, ToJoltVec3(rigidBody->linearVelocity));
        if (!rigidBody->useGravity)
            context.bodyInterface->SetGravityFactor(bodyId, 0.0f);
    }

    inline void UnregisterEntity(Context& context, entt::registry& registry, entt::entity entity)
    {
        (void)registry;
        auto bodyIt = context.entityToBody.find(entity);
        if (bodyIt == context.entityToBody.end() || !context.bodyInterface)
            return;

        const JPH::BodyID bodyId = bodyIt->second;
        context.bodyInterface->RemoveBody(bodyId);
        context.bodyInterface->DestroyBody(bodyId);

        context.bodyToEntity.erase(bodyId);
        context.entityToBody.erase(bodyIt);

        auto* rigidBody = registry.try_get<RigidBodyComponent>(entity);
        if (rigidBody)
            rigidBody->registeredInWorld = false;
    }

    inline void SyncNewBodies(Context& context, entt::registry& registry)
    {
        for (auto [entity, rigidBody, collider, transform] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each())
        {
            (void)collider;
            (void)transform;
            if (!rigidBody.registeredInWorld)
                RegisterEntity(context, registry, entity);
        }
    }

    inline void SyncKinematicToPhysics(Context& context, entt::registry& registry)
    {
        if (!context.bodyInterface)
            return;

        for (auto [entity, rigidBody, collider, transform] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each())
        {
            if (rigidBody.bodyType != RigidBodyType::Kinematic)
                continue;

            auto bodyIt = context.entityToBody.find(entity);
            if (bodyIt == context.entityToBody.end())
                continue;

            context.bodyInterface->SetPositionAndRotation(
                bodyIt->second,
                ToJoltRVec3(transform.GetPosition() + (transform.GetRotation() * GetColliderCenterOffset(collider))),
                ToJoltQuat(transform.GetRotation()),
                JPH::EActivation::Activate);

            context.bodyInterface->SetLinearVelocity(bodyIt->second, ToJoltVec3(rigidBody.linearVelocity));
            context.bodyInterface->SetGravityFactor(bodyIt->second, 0.0f);
        }
    }

    inline void Step(Context& context, float deltaTime)
    {
        if (context.paused || deltaTime <= 0.0f || !context.physicsSystem || !context.tempAllocator || !context.jobSystem)
            return;

        const float dt = std::min(deltaTime, 0.05f);
        const int collisionSteps = std::max(1, static_cast<int>(std::ceil(dt * 60.0f)));
        context.physicsSystem->Update(dt, collisionSteps, context.tempAllocator, context.jobSystem);
    }

    inline void SyncPhysicsToTransforms(Context& context, entt::registry& registry)
    {
        if (!context.bodyInterface)
            return;

        std::vector<entt::entity> staleEntities{};
        staleEntities.reserve(context.entityToBody.size());

        for (const auto& [entity, bodyId] : context.entityToBody)
        {
            if (!registry.valid(entity))
            {
                staleEntities.push_back(entity);
                continue;
            }

            auto [transform, rigidBody, collider] = registry.try_get<TransformComponent, RigidBodyComponent, ColliderComponent>(entity);
            if (!transform || !rigidBody || !collider)
            {
                staleEntities.push_back(entity);
                continue;
            }

            const glm::vec3 bodyPos = ToGlmFromRVec3(context.bodyInterface->GetCenterOfMassPosition(bodyId));
            const glm::quat bodyRot = ToGlmQuat(context.bodyInterface->GetRotation(bodyId));
            transform->SetPosition(bodyPos - (bodyRot * GetColliderCenterOffset(*collider)));
            transform->SetRotation(ToGlmQuat(context.bodyInterface->GetRotation(bodyId)));
            rigidBody->linearVelocity = ToGlmVec3(context.bodyInterface->GetLinearVelocity(bodyId));
        }

        for (const entt::entity staleEntity : staleEntities)
            UnregisterEntity(context, registry, staleEntity);
    }

    inline void Update(Context& context, entt::registry& registry, float deltaTime)
    {
        SyncNewBodies(context, registry);
        SyncKinematicToPhysics(context, registry);
        Step(context, deltaTime);
        SyncPhysicsToTransforms(context, registry);
    }

    inline void Reset(Context& context, entt::registry& registry)
    {
        if (!context.bodyInterface)
            return;

        for (auto [entity, rigidBody, collider, transform] : registry.view<RigidBodyComponent, ColliderComponent, TransformComponent>().each())
        {
            if (rigidBody.bodyType != RigidBodyType::Dynamic)
                continue;

            auto bodyIt = context.entityToBody.find(entity);
            if (bodyIt == context.entityToBody.end())
                continue;

            const JPH::BodyID bodyId = bodyIt->second;
            context.bodyInterface->SetLinearVelocity(bodyId, JPH::Vec3::sZero());
            context.bodyInterface->SetAngularVelocity(bodyId, JPH::Vec3::sZero());
            rigidBody.linearVelocity = glm::vec3(0.0f);

            context.bodyInterface->SetPositionAndRotation(
                bodyId,
                ToJoltRVec3(transform.GetPosition() + (transform.GetRotation() * GetColliderCenterOffset(collider))),
                ToJoltQuat(transform.GetRotation()),
                JPH::EActivation::Activate);
        }
    }

    inline void Clear(Context& context, entt::registry& registry)
    {
        if (context.bodyInterface)
        {
            for (const auto& [entity, bodyId] : context.entityToBody)
            {
                context.bodyInterface->RemoveBody(bodyId);
                context.bodyInterface->DestroyBody(bodyId);
                if (registry.valid(entity))
                {
                    if (auto* rigidBody = registry.try_get<RigidBodyComponent>(entity))
                        rigidBody->registeredInWorld = false;
                }
            }
        }

        context.entityToBody.clear();
        context.bodyToEntity.clear();
    }

    inline bool SetLinearVelocity(Context& context, entt::registry& registry, entt::entity entity, const glm::vec3& velocity)
    {
        if (!context.bodyInterface || !registry.valid(entity))
            return false;

        auto bodyIt = context.entityToBody.find(entity);
        if (bodyIt == context.entityToBody.end())
            return false;

        context.bodyInterface->SetLinearVelocity(bodyIt->second, ToJoltVec3(velocity));

        if (auto* rb = registry.try_get<RigidBodyComponent>(entity))
            rb->linearVelocity = velocity;

        return true;
    }
}
