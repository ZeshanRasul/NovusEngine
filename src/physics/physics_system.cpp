#include "physics/physics_system.h"

#include "ECS/entity.h"
#include "ECS/components/physics_component.h"
#include "ECS/components/transform_component.h"
#include <algorithm>
#include <glm/gtx/norm.hpp>

void PhysicsSystem::initialize()
{
    paused = false;
}

void PhysicsSystem::shutdown()
{
    clear();
}

void PhysicsSystem::setPaused(bool value)
{
    paused = value;
}

bool PhysicsSystem::isPaused() const
{
    return paused;
}

void PhysicsSystem::setGravity(glm::vec3 const& g)
{
    gravity = g;
}

glm::vec3 const& PhysicsSystem::getGravity() const
{
    return gravity;
}

void PhysicsSystem::registerEntity(Entity* entity)
{
    if (!entity)
        return;

    auto* transform = entity->GetComponent<TransformComponent>();
    auto* physics = entity->GetComponent<PhysicsComponent>();
    if (!transform || !physics)
        return;

    if (physics->registeredInWorld)
        return;

    physics->bodyId = nextBodyId++;
    physics->registeredInWorld = true;

    BodyState state{};
    state.entity = entity;
    state.transform = transform;
    state.physics = physics;
    state.velocity = physics->getLinearVelocity();
    state.spawn.position = transform->GetPosition();
    state.spawn.velocity = state.velocity;
    bodies[physics->bodyId] = state;
}

void PhysicsSystem::unregisterEntity(Entity* entity)
{
    if (!entity)
        return;

    auto* physics = entity->GetComponent<PhysicsComponent>();
    if (!physics || !physics->registeredInWorld)
        return;

    bodies.erase(physics->bodyId);
    physics->bodyId = -1;
    physics->registeredInWorld = false;
}

void PhysicsSystem::clear()
{
    for (auto& [id, body] : bodies)
    {
        if (body.physics)
        {
            body.physics->bodyId = -1;
            body.physics->registeredInWorld = false;
            body.physics->setLinearVelocity(glm::vec3(0.0f));
        }
    }
    bodies.clear();
}

void PhysicsSystem::reset()
{
    for (auto& [id, body] : bodies)
    {
        if (!body.physics || !body.transform)
            continue;

        if (body.physics->bodyType == PhysicsBodyType::Dynamic)
        {
            body.transform->SetPosition(body.spawn.position);
            body.velocity = body.spawn.velocity;
            body.physics->setLinearVelocity(body.velocity);
        }
    }
}

void PhysicsSystem::step(float deltaTime)
{
    if (paused || deltaTime <= 0.0f)
        return;

    // Clamp to avoid spiral-of-death on a stall frame
    const float dt = std::min(deltaTime, 0.05f);

    for (auto& [id, body] : bodies)
    {
        if (!body.physics || !body.transform)
            continue;

        switch (body.physics->bodyType)
        {
        case PhysicsBodyType::Dynamic:
            stepDynamicBody(body, dt);
            break;
        case PhysicsBodyType::Kinematic:
            body.velocity = body.physics->getLinearVelocity();
            body.transform->SetPosition(body.transform->GetPosition() + body.velocity * dt);
            break;
        case PhysicsBodyType::Static:
        default:
            break;
        }
    }

    resolveBodyCollisions();
}

void PhysicsSystem::stepDynamicBody(BodyState& state, float dt)
{
    auto& physics = *state.physics;
    auto& transform = *state.transform;

    if (physics.useGravity)
        state.velocity += gravity * dt;

    glm::vec3 pos = transform.GetPosition();
    pos += state.velocity * dt;

    // --- Floor collision ---
    const float floorY = 0.0f;
    const float shapeRadius = physics.shapeType == PhysicsShapeType::Sphere
        ? physics.radius
        : physics.halfExtents.y;
    const float effectiveRadius = std::max(0.05f, shapeRadius);

    if (pos.y - effectiveRadius < floorY)
    {
        pos.y = floorY + effectiveRadius;
        if (state.velocity.y < 0.0f)
        {
            state.velocity.y = -state.velocity.y * glm::clamp(physics.restitution, 0.0f, 1.0f);

            // Kill tiny bounces so objects come fully to rest
            if (std::abs(state.velocity.y) < 0.5f)
                state.velocity.y = 0.0f;
        }
        const float frictionFactor = glm::clamp(1.0f - physics.friction * dt * 6.0f, 0.0f, 1.0f);
        state.velocity.x *= frictionFactor;
        state.velocity.z *= frictionFactor;
    }

    transform.SetPosition(pos);
    physics.setLinearVelocity(state.velocity);
}

void PhysicsSystem::resolveBodyCollisions()
{
    // Collect all dynamic bodies into a list for O(n^2) broad + narrow phase.
    // For a demo with <=128 objects this is fast enough.
    std::vector<BodyState*> dynamics;
    dynamics.reserve(bodies.size());
    for (auto& [id, body] : bodies)
    {
        if (body.physics && body.transform &&
            body.physics->bodyType == PhysicsBodyType::Dynamic)
            dynamics.push_back(&body);
    }

    for (size_t i = 0; i < dynamics.size(); ++i)
    {
        for (size_t j = i + 1; j < dynamics.size(); ++j)
        {
            BodyState& a = *dynamics[i];
            BodyState& b = *dynamics[j];

            glm::vec3 posA = a.transform->GetPosition();
            glm::vec3 posB = b.transform->GetPosition();
            glm::vec3 delta = posB - posA;

            // Use the average of the two radii as a unified contact radius so
            // sphere vs. box pairs get a reasonable approximation.
            auto effectiveRadius = [](const PhysicsComponent& p) {
                return p.shapeType == PhysicsShapeType::Sphere
                    ? p.radius
                    : glm::length(p.halfExtents) * 0.57735f; // inscribed sphere of cube
            };

            const float rA = effectiveRadius(*a.physics);
            const float rB = effectiveRadius(*b.physics);
            const float minDist = rA + rB;

            const float dist2 = glm::length2(delta);
            if (dist2 >= minDist * minDist || dist2 < 1e-8f)
                continue;

            const float dist = std::sqrt(dist2);
            const glm::vec3 normal = delta / dist;           // from A toward B
            const float penetration = minDist - dist;

            // Positional correction — push bodies apart equally
            const float correction = penetration * 0.5f;
            posA -= normal * correction;
            posB += normal * correction;
            a.transform->SetPosition(posA);
            b.transform->SetPosition(posB);

            // Velocity response along the collision normal (1D impulse)
            const float restitution = std::min(a.physics->restitution, b.physics->restitution);
            const float vRel = glm::dot(b.velocity - a.velocity, normal);

            // Only resolve if bodies are approaching
            if (vRel < 0.0f)
            {
                // Equal-mass impulse (both mass=1 assumed for demo simplicity)
                const float impulse = -(1.0f + restitution) * vRel * 0.5f;
                a.velocity -= normal * impulse;
                b.velocity += normal * impulse;
                a.physics->setLinearVelocity(a.velocity);
                b.physics->setLinearVelocity(b.velocity);
            }
        }
    }
}
