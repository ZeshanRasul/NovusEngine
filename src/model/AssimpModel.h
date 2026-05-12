/* Assimp model, ready to draw */
#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

#include <glm/glm.hpp>

#include "AssimpMesh.h"
#include "AssimpNode.h"
#include "AssimpAnimClip.h"
#include "../vulkan/VertexBuffer.h"
#include "../vulkan/IndexBuffer.h"

#include "../renderer/VkRenderData.h"

class AssimpModel {
  public:
    bool loadModel(VkRenderData &renderData, std::string modelFilename, unsigned int extraImportFlags = 0);
    glm::mat4 getRootTranformationMatrix();

    void draw(VkRenderData &renderData);
    void drawInstanced(VkRenderData &renderData, uint32_t instanceCount);
    unsigned int getTriangleCount();

    std::string getModelFileName();
    std::string getModelFileNamePath();

    bool hasAnimations();
    const std::vector<std::shared_ptr<AssimpAnimClip>>& getAnimClips();

    const std::vector<std::shared_ptr<AssimpNode>>& getNodeList();
    const std::unordered_map<std::string, std::shared_ptr<AssimpNode>>& getNodeMap();

    const std::vector<std::shared_ptr<AssimpBone>>& getBoneList();
    const std::unordered_map<std::string, glm::mat4>& getBoneOffsetMatrices();

    const std::shared_ptr<AssimpNode> getRootNode();

    // Access raw mesh/buffer data for the skinning render path
    const std::vector<VkMesh>& getModelMeshes() const { return mModelMeshes; }
    const std::vector<VkVertexBufferData>& getVertexBuffers() const { return mVertexBuffers; }
    const std::vector<VkIndexBufferData>& getIndexBuffers() const { return mIndexBuffers; }

    // Returns the diffuse VkTextureData for mesh i, or nullptr if none
    const VkTextureData* getDiffuseTexture(size_t meshIndex) const
    {
        if (meshIndex >= mModelMeshes.size()) return nullptr;
        const auto& texMap = mModelMeshes[meshIndex].textures;
        auto it = texMap.find(aiTextureType_DIFFUSE);
        if (it == texMap.end()) return nullptr;
        auto texIt = mTextures.find(it->second);
        if (texIt != mTextures.end()) return &texIt->second;
        return nullptr;
    }

    void cleanup(VkRenderData &renderData);

private:
    void processNode(VkRenderData &renderData, std::shared_ptr<AssimpNode> node, aiNode* aNode, const aiScene* scene, std::string assetDirectory);
    void createNodeList(std::shared_ptr<AssimpNode> node, std::shared_ptr<AssimpNode> newNode, std::vector<std::shared_ptr<AssimpNode>> &list);

    unsigned int mTriangleCount = 0;
    unsigned int mVertexCount = 0;

    /* store the root node for direct access */
    std::shared_ptr<AssimpNode> mRootNode = nullptr;
    /* a map to find the node by name */
    std::unordered_map<std::string, std::shared_ptr<AssimpNode>> mNodeMap{};
    /* and a 'flat' map to keep the order of insertation  */
    std::vector<std::shared_ptr<AssimpNode>> mNodeList{};

    std::vector<std::shared_ptr<AssimpBone>> mBoneList;
    std::unordered_map<std::string, glm::mat4> mBoneOffsetMatrices{};

    std::vector<std::shared_ptr<AssimpAnimClip>> mAnimClips{};

    std::vector<VkMesh> mModelMeshes{};
    std::vector<VkVertexBufferData> mVertexBuffers{};
    std::vector<VkIndexBufferData> mIndexBuffers{};

    // map textures to external or internal texture names
    std::unordered_map<std::string, VkTextureData> mTextures{};
    VkTextureData mPlaceholderTexture{};
    VkTextureData mWhiteTexture{};

    glm::mat4 mRootTransformMatrix = glm::mat4(1.0f);

    std::string mModelFilenamePath;
    std::string mModelFilename;
};
