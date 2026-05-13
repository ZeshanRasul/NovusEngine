#pragma once

#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>

class Entity;
class TransformComponent;
class PhysicsComponent;

class PhysicsSystem
{
public:
    void initialize();
    void shutdown();

    void setPaused(bool paused);
    bool isPaused() const;

    void setGravity(glm::vec3 const& gravity);
    glm::vec3 const& getGravity() const;

    void registerEntity(Entity* entity);
    void unregisterEntity(Entity* entity);
    void clear();

    // Resets all dynamic bodies to their original spawn pose and zero velocity.
    void reset();

    void step(float deltaTime);

private:
    struct SpawnState
    {
        glm::vec3 position;
        glm::vec3 velocity;
    };

    struct BodyState
    {
        Entity* entity = nullptr;
        TransformComponent* transform = nullptr;
        PhysicsComponent* physics = nullptr;
        glm::vec3 velocity = glm::vec3(0.0f);
        SpawnState spawn{};
    };

    int nextBodyId = 0;
    glm::vec3 gravity = glm::vec3(0.0f, 9.81f, 0.0f);
    bool paused = false;
    std::unordered_map<int, BodyState> bodies;

    void stepDynamicBody(BodyState& state, float dt);
    void resolveBodyCollisions();
};
