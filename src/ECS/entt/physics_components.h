#pragma once

#include <glm/glm.hpp>

enum class RigidBodyType
{
    Static,
    Dynamic,
    Kinematic
};

enum class ColliderShapeType
{
    Box,
    Sphere,
    Capsule
};

struct RigidBodyComponent
{
    RigidBodyType bodyType = RigidBodyType::Dynamic;
    float mass = 1.0f;
    float friction = 0.5f;
    float restitution = 0.2f;
    bool useGravity = true;

    glm::vec3 linearVelocity = glm::vec3(0.0f);
    bool registeredInWorld = false;
    int bodyId = -1;

    void setLinearVelocity(const glm::vec3& velocity) { linearVelocity = velocity; }
    const glm::vec3& getLinearVelocity() const { return linearVelocity; }
};

struct ColliderComponent
{
    ColliderShapeType shapeType = ColliderShapeType::Box;
    glm::vec3 halfExtents = glm::vec3(0.5f);
    float radius = 0.5f;
    float halfHeight = 0.5f;
    glm::vec3 centerOffset = glm::vec3(0.0f);
    bool alignBottomToEntity = false;
};

using PhysicsBodyType = RigidBodyType;
using PhysicsShapeType = ColliderShapeType;

struct PhysicsComponent : public RigidBodyComponent, public ColliderComponent
{
};
