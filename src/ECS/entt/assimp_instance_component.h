#pragma once

#include <memory>

class AssimpInstance;

struct AssimpInstanceComponent
{
    std::shared_ptr<AssimpInstance> instance;
};