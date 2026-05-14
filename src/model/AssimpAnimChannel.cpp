#include "AssimpAnimChannel.h"

#include <cmath>

#include "Logger.h"

void AssimpAnimChannel::loadChannelData(aiNodeAnim* nodeAnim) {
  mNodeName = nodeAnim->mNodeName.C_Str();
  unsigned int numTranslations = nodeAnim->mNumPositionKeys;
  unsigned int numRotations = nodeAnim->mNumRotationKeys;
  unsigned int numScalings = nodeAnim->mNumScalingKeys;
  unsigned int preState = nodeAnim->mPreState;
  unsigned int postState = nodeAnim->mPostState;

  Logger::log(1, "%s: - loading animation channel for node '%s', with %i translation keys, %i rotation keys, %i scaling keys (preState %i, postState %i)\n",
              __FUNCTION__, mNodeName.c_str(), numTranslations, numRotations, numScalings, preState, postState);

  for (unsigned int i = 0; i < numTranslations; ++i) {
    mTranslationTiminngs.emplace_back(static_cast<float>(nodeAnim->mPositionKeys[i].mTime));
    mTranslations.emplace_back(glm::vec3(nodeAnim->mPositionKeys[i].mValue.x, nodeAnim->mPositionKeys[i].mValue.y, nodeAnim->mPositionKeys[i].mValue.z));
  }

  for (unsigned int i = 0; i < numRotations; ++i) {
    mRotationTiminigs.emplace_back(static_cast<float>(nodeAnim->mRotationKeys[i].mTime));
    mRotations.emplace_back(glm::quat(nodeAnim->mRotationKeys[i].mValue.w, nodeAnim->mRotationKeys[i].mValue.x, nodeAnim->mRotationKeys[i].mValue.y, nodeAnim->mRotationKeys[i].mValue.z));
  }

  for (unsigned int i = 0; i < numScalings; ++i) {
    mScaleTimings.emplace_back(static_cast<float>(nodeAnim->mScalingKeys[i].mTime));
    mScalings.emplace_back(glm::vec3(nodeAnim->mScalingKeys[i].mValue.x, nodeAnim->mScalingKeys[i].mValue.y, nodeAnim->mScalingKeys[i].mValue.z));
  }

  /* precalcuate the inverse offset to avoid divisions when scaling the section */
  for (unsigned int i = 0; i < mTranslationTiminngs.size() - 1; ++i) {
    mInverseTranslationTimeDiffs.emplace_back(1.0f / (mTranslationTiminngs.at(i + 1) - mTranslationTiminngs.at(i)));
  }
  for (unsigned int i = 0; i < mRotationTiminigs.size() - 1; ++i) {
    mInverseRotationTimeDiffs.emplace_back(1.0f / (mRotationTiminigs.at(i + 1) - mRotationTiminigs.at(i)));
  }
  for (unsigned int i = 0; i < mScaleTimings.size() - 1; ++i) {
    mInverseScaleTimeDiffs.emplace_back(1.0f / (mScaleTimings.at(i + 1) - mScaleTimings.at(i)));
  }

  mPreState = preState;
  mPostState = postState;
}

std::string AssimpAnimChannel::getTargetNodeName() {
  return mNodeName;
}

float AssimpAnimChannel::getMaxTime() {
  float maxTranslationTime = mTranslationTiminngs.at(mTranslationTiminngs.size() - 1);
  float maxRotationTime = mRotationTiminigs.at(mRotationTiminigs.size() - 1);
  float maxScaleTime = mScaleTimings.at(mScaleTimings.size() - 1);

  return std::max(std::max(maxRotationTime, maxTranslationTime), maxScaleTime);
}

/* precalculate TRS matrix */
glm::mat4 AssimpAnimChannel::getTRSMatrix(float time) {
  return glm::translate(glm::mat4_cast(getRotation(time)) * glm::scale(glm::mat4(1.0f), getScaling(time)), getTranslation(time));
}

glm::vec3 AssimpAnimChannel::getTranslation(float time) {
  if (mTranslations.empty()) {
    return glm::vec3(0.0f);
  }

  if (mTranslations.size() == 1 || mTranslationTiminngs.size() <= 1) {
    return mTranslations.front();
  }

  const float firstTime = mTranslationTiminngs.front();
  const float lastTime = mTranslationTiminngs.back();

  if (lastTime <= firstTime) {
    return mTranslations.front();
  }

  if (mPreState == aiAnimBehaviour_REPEAT || mPostState == aiAnimBehaviour_REPEAT) {
    const float duration = lastTime - firstTime;
    if (duration > 0.0f) {
      time = std::fmod(time - firstTime, duration);
      if (time < 0.0f) {
        time += duration;
      }
      time += firstTime;
    }
  }

  /* handle time before and after */
  if (time <= firstTime) {
    return mTranslations.front();
  }

  if (time >= lastTime) {
    return mTranslations.back();
  }

  auto upper = std::upper_bound(mTranslationTiminngs.begin(), mTranslationTiminngs.end(), time);
  size_t index1 = static_cast<size_t>(std::distance(mTranslationTiminngs.begin(), upper));
  size_t index0 = index1 - 1;   

  const float t0 = mTranslationTiminngs[index0];
  const float t1 = mTranslationTiminngs[index1];
  const float dt = t1 - t0;
  const float alpha = dt > 0.0f ? (time - t0) / dt : 0.0f;

  return glm::mix(mTranslations[index0], mTranslations[index1], alpha);
}

glm::vec3 AssimpAnimChannel::getScaling(float time) {
  if (mScalings.empty()) {
    return glm::vec3(1.0f);
  }

  if (mScalings.size() == 1 || mScaleTimings.size() <= 1) {
    return mScalings.front();
  }

  const float firstTime = mScaleTimings.front();
  const float lastTime = mScaleTimings.back();

  if (lastTime <= firstTime) {
    return mScalings.front();
  }

  if (mPreState == aiAnimBehaviour_REPEAT || mPostState == aiAnimBehaviour_REPEAT) {
    const float duration = lastTime - firstTime;
    if (duration > 0.0f) {
      time = std::fmod(time - firstTime, duration);
      if (time < 0.0f) {
        time += duration;
      }
      time += firstTime;
    }
  }

  /* handle time before and after */
  if (time <= firstTime) {
    return mScalings.front();
  }

  if (time >= lastTime) {
    return mScalings.back();
  }

  auto upper = std::upper_bound(mScaleTimings.begin(), mScaleTimings.end(), time);
  size_t index1 = static_cast<size_t>(std::distance(mScaleTimings.begin(), upper));
  size_t index0 = index1 - 1;

  const float t0 = mScaleTimings[index0];
  const float t1 = mScaleTimings[index1];
  const float dt = t1 - t0;
  const float alpha = dt > 0.0f ? (time - t0) / dt : 0.0f;

  return glm::mix(mScalings[index0], mScalings[index1], alpha);
}

glm::quat AssimpAnimChannel::getRotation(float time) {
  if (mRotations.empty()) {
    return glm::identity<glm::quat>();
  }

  if (mRotations.size() == 1 || mRotationTiminigs.size() <= 1) {
    return glm::normalize(mRotations.front());
  }

  const float firstTime = mRotationTiminigs.front();
  const float lastTime = mRotationTiminigs.back();

  if (lastTime <= firstTime) {
    return glm::normalize(mRotations.front());
  }

  if (mPreState == aiAnimBehaviour_REPEAT || mPostState == aiAnimBehaviour_REPEAT) {
    const float duration = lastTime - firstTime;
    if (duration > 0.0f) {
      time = std::fmod(time - firstTime, duration);
      if (time < 0.0f) {
        time += duration;
      }
      time += firstTime;
    }
  }

  /* handle time before and after */
  if (time <= firstTime) {
    return glm::normalize(mRotations.front());
  }

  if (time >= lastTime) {
    return glm::normalize(mRotations.back());
  }

  auto upper = std::upper_bound(mRotationTiminigs.begin(), mRotationTiminigs.end(), time);
  size_t index1 = static_cast<size_t>(std::distance(mRotationTiminigs.begin(), upper));
  size_t index0 = index1 - 1;

  const float t0 = mRotationTiminigs[index0];
  const float t1 = mRotationTiminigs[index1];
  const float dt = t1 - t0;
  const float alpha = dt > 0.0f ? (time - t0) / dt : 0.0f;

  /* rotations are interpolated via SLERP */
  return glm::normalize(glm::slerp(mRotations[index0], mRotations[index1], alpha));
}

int AssimpAnimChannel::getBoneId() {
    return mBoneId;
}

void AssimpAnimChannel::setBoneId(int id) {
    mBoneId = id;
}
