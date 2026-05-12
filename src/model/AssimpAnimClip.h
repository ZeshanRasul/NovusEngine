#pragma once

#include <string>
#include <vector>
#include <memory>
#include <assimp/anim.h>
#include "AssimpAnimChannel.h"

class AssimpAnimClip {
  public:
    void addChannels(aiAnimation* animation);

    // keep existing
    std::vector<std::shared_ptr<AssimpAnimChannel>> getChannels();
    // add mutable accessor for setup pass
    std::vector<std::shared_ptr<AssimpAnimChannel>>& getChannelsMutable();

    std::string getClipName();
    float getClipDuration();
    float getClipTicksPerSecond();
    void setClipName(std::string name);

  private:
    std::string mClipName;
    double mClipDuration = 0.0f;
    double mClipTicksPerSecond = 0.0f;
    std::vector<std::shared_ptr<AssimpAnimChannel>> mAnimChannels{};
};
