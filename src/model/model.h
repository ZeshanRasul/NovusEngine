#pragma once
#include <string>
#include "../ECS/components/renderable_component.h"

class Model
{
public:
	static void loadModel(std::string modelFilename, RenderableComponent& gameObj);
};