#include "physics/physics_system.h"

#include "ECS/entity.h"
#include "ECS/components/physics_component.h"
#include "ECS/components/transform_component.h"
#include <algorithm>
#include <cmath>
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

void PhysicsSystem::registerEntity(entt::entity& entity, entt::registry& registry)
{
    if (entity == entt::null)
        return;

	auto [transform, physics] = registry.try_get<TransformComponent, PhysicsComponent>(entity);
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

void PhysicsSystem::unregisterEntity(entt::entity& entity, entt::registry& registry)
{
    if (entity == entt::null)
        return;

    auto* physics = registry.try_get<PhysicsComponent>(entity);
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

    transform.SetPosition(pos);
    physics.setLinearVelocity(state.velocity);
}

void PhysicsSystem::resolveBodyCollisions()
{
    std::vector<BodyState*> dynamics;
    std::vector<BodyState*> statics;
    dynamics.reserve(bodies.size());
    statics.reserve(bodies.size());

    for (auto& [id, body] : bodies)
    {
        if (!body.physics || !body.transform)
            continue;

        if (body.physics->bodyType == PhysicsBodyType::Dynamic)
            dynamics.push_back(&body);
        else if (body.physics->bodyType == PhysicsBodyType::Static)
            statics.push_back(&body);
    }

    auto scaledHalfExtents = [](BodyState const& b) {
        const glm::vec3 s = glm::abs(b.transform->GetScale());
        return b.physics->halfExtents * s;
    };

    // Dynamic vs Static (AABB, axis-aligned, no rotation)
    for (BodyState* d : dynamics)
    {
        if (d->physics->shapeType != PhysicsShapeType::Box)
            continue;

        glm::vec3 dPos = d->transform->GetPosition();
        const glm::vec3 dHalf = scaledHalfExtents(*d);

        for (BodyState* s : statics)
        {
            if (s->physics->shapeType != PhysicsShapeType::Box)
                continue;

            const glm::vec3 sPos = s->transform->GetPosition();
            const glm::vec3 sHalf = scaledHalfExtents(*s);

            const glm::vec3 delta = dPos - sPos;
            const glm::vec3 overlap = (dHalf + sHalf) - glm::abs(delta);
            if (overlap.x <= 0.0f || overlap.y <= 0.0f || overlap.z <= 0.0f)
                continue;

            // Resolve along minimum penetration axis
            glm::vec3 normal(0.0f);
            float minPen = overlap.x;
            normal.x = (delta.x >= 0.0f) ? 1.0f : -1.0f;

            if (overlap.y < minPen) {
                minPen = overlap.y;
                normal = glm::vec3(0.0f, (delta.y >= 0.0f) ? 1.0f : -1.0f, 0.0f);
            }
            if (overlap.z < minPen) {
                minPen = overlap.z;
                normal = glm::vec3(0.0f, 0.0f, (delta.z >= 0.0f) ? 1.0f : -1.0f);
            }

            dPos += normal * minPen;

            const float vRelN = glm::dot(d->velocity, normal);
            if (vRelN < 0.0f) {
                const float e = glm::clamp(d->physics->restitution, 0.0f, 1.0f);
                d->velocity -= (1.0f + e) * vRelN * normal;
            }

            // Basic ground friction when resting on top surface
            if (normal.y > 0.5f) {
                const float fr = glm::clamp(1.0f - d->physics->friction * 0.1f, 0.0f, 1.0f);
                d->velocity.x *= fr;
                d->velocity.z *= fr;
            }
        }

        d->transform->SetPosition(dPos);
        d->physics->setLinearVelocity(d->velocity);
    }
}
