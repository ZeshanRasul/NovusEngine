#include "core/engine_app.h"

#include "renderer/renderer.h"

EngineApp::EngineApp()
    : mRenderer(new Renderer())
{
}

EngineApp::~EngineApp()
{
    delete mRenderer;
    mRenderer = nullptr;
}

void EngineApp::run()
{
    if (mRenderer)
        mRenderer->run();
}
