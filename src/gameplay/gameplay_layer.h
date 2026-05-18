#pragma once

#include <functional>
#include <string>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "../ECS/component.h"
#include "../ECS/components/transform_component.h"
#include "../ECS/components/physics_component.h"
#include "../ECS/components/player_controller_component.h"
#include "../physics/physics_system.h"

namespace Gameplay
{
    struct GameplayInputState
    {
        bool moveForward = false;
        bool moveBackward = false;
        bool moveLeft = false;
        bool moveRight = false;
        bool jump = false;
        bool sprint = false;
    };

    struct RuntimeContext
    {
        entt::registry* registry = nullptr;
        PhysicsSystem* physicsSystem = nullptr;
        std::function<entt::entity(const std::string&, const glm::vec3&)> spawnPrefab;
        std::function<void(entt::entity)> destroyEntity;
    };

    class IGameLayer
    {
    public:
        virtual ~IGameLayer() = default;

        virtual void onEnterPlay(RuntimeContext& context) { (void)context; }
        virtual void onExitPlay(RuntimeContext& context) { (void)context; }
        virtual void onFixedUpdate(float fixedDeltaTime, const GameplayInputState& input, RuntimeContext& context)
        {
            (void)fixedDeltaTime;
            (void)input;
            (void)context;
        }
    };

    class DefaultGameLayer final : public IGameLayer
    {
    public:
        void onEnterPlay(RuntimeContext& context) override
        {
            mPlayerEntity = entt::null;
            if (!context.registry)
                return;

            auto& registry = *context.registry;
         for (auto entity : registry.view<PlayerControllerComponent>())
            {
               mPlayerEntity = entity;
                break;
            }
        }

        void onFixedUpdate(float, const GameplayInputState& input, RuntimeContext& context) override
        {
            if (!context.registry || !context.physicsSystem)
                return;

            auto& registry = *context.registry;
          if (!registry.valid(mPlayerEntity) || !registry.all_of<RigidBodyComponent, TransformComponent, PlayerControllerComponent>(mPlayerEntity))
                return;

          auto& controller = registry.get<PlayerControllerComponent>(mPlayerEntity);
            if (!controller.enabled)
                return;

            auto& rigidBody = registry.get<RigidBodyComponent>(mPlayerEntity);
            glm::vec3 velocity = rigidBody.linearVelocity;

           const float speed = input.sprint ? controller.sprintSpeed : controller.moveSpeed;
            glm::vec3 moveDir(0.0f);
            if (input.moveForward) moveDir.z -= 1.0f;
            if (input.moveBackward) moveDir.z += 1.0f;
            if (input.moveLeft) moveDir.x -= 1.0f;
            if (input.moveRight) moveDir.x += 1.0f;

            if (glm::dot(moveDir, moveDir) > 0.0f)
                moveDir = glm::normalize(moveDir);

            velocity.x = moveDir.x * speed;
            velocity.z = moveDir.z * speed;
            if (input.jump)
              velocity.y = controller.jumpImpulse;

            context.physicsSystem->setLinearVelocity(mPlayerEntity, velocity);
        }

    private:
        entt::entity mPlayerEntity = entt::null;
    };
}
