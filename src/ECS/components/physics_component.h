#pragma once

#include "../component.h"
#include <glm/glm.hpp>

enum class PhysicsBodyType
{
    Static,
    Dynamic,
    Kinematic
};

enum class PhysicsShapeType
{
    Box,
    Sphere,
    Capsule
};

class PhysicsComponent : public Component
{
public:
    PhysicsBodyType bodyType = PhysicsBodyType::Dynamic;
    PhysicsShapeType shapeType = PhysicsShapeType::Box;
    glm::vec3 halfExtents = glm::vec3(0.5f);
    float radius = 0.5f;
    float mass = 1.0f;
    float friction = 0.5f;
    float restitution = 0.2f;
    bool useGravity = true;

    int bodyId = -1;
    bool registeredInWorld = false;

    void setLinearVelocity(glm::vec3 const& v) { linearVelocity = v; }
    glm::vec3 const& getLinearVelocity() const { return linearVelocity; }

private:
    glm::vec3 linearVelocity = glm::vec3(0.0f);
};
