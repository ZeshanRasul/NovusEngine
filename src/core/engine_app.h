#pragma once

class Renderer;

class EngineApp
{
public:
    EngineApp();
    ~EngineApp();

    void run();

private:
    Renderer* mRenderer = nullptr;
};
