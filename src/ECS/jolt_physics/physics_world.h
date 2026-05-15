#pragma once
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <glm/glm.hpp>

class PhysicsWorld
{
public:

	void Init();
	void Shutdown();

	void CreateScene();
	void Step(float dt);

	glm::vec3 GetSpherePosition() const;

	JPH::PhysicsSystem physicsSystem;
	JPH::TempAllocatorImpl* tempAllocator = nullptr;
	JPH::JobSystemThreadPool* jobSystem = nullptr;
	JPH::BodyInterface* bodies;
	JPH::BodyID sphereBodyID;
};