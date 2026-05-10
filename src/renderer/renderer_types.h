#pragma once

#include <string>
#include <array>
#include <functional>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#   include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

// ---------------------------------------------------------------------------
// Vertex layout matching the PBR shader (pos / normal / texcoord / tangent)
// ---------------------------------------------------------------------------
struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec4 tangent;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return { .binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat,    offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat,    offsetof(Vertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat,       offsetof(Vertex, texCoord)),
            vk::VertexInputAttributeDescription(3, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent))
        };
    }

    bool operator==(const Vertex& other) const
    {
        return pos == other.pos && normal == other.normal &&
               texCoord == other.texCoord && tangent == other.tangent;
    }
};

namespace std
{
    template<> struct hash<Vertex>
    {
        size_t operator()(Vertex const& vertex) const
        {
            return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1)
                 ^ (hash<glm::vec2>()(vertex.texCoord) << 1)
                 ^ (hash<glm::vec4>()(vertex.tangent) << 1);
        }
    };
}

// ---------------------------------------------------------------------------
// PBR material (glTF Metallic-Roughness)
// ---------------------------------------------------------------------------
class Material {
public:
    explicit Material(std::string name) : name(std::move(name)) {}
    ~Material() = default;

    [[nodiscard]] const std::string& GetName() const { return name; }

    // PBR properties
    glm::vec3 albedo    = glm::vec3(1.0f);
    float metallic      = 0.0f;
    float roughness     = 0.6f;
    float ao            = 1.0f;
    glm::vec3 emissive  = glm::vec3(0.0f);
    float ior           = 1.5f;
    float emissiveStrength = 1.0f;
    float alpha         = 1.0f;
    float transmissionFactor = 0.0f;

    // Specular-Glossiness workflow
    bool useSpecularGlossiness = false;
    glm::vec3 specularFactor   = glm::vec3(0.04f);
    float glossinessFactor     = 1.0f;
    std::string specGlossTexturePath;

    // Alpha handling
    std::string alphaMode  = "OPAQUE";
    float alphaCutoff      = 0.5f;

    // Texture paths
    std::string albedoTexturePath;
    std::string normalTexturePath;
    std::string metallicRoughnessTexturePath;
    std::string occlusionTexturePath;
    std::string emissiveTexturePath;

    bool isGlass  = false;
    bool isLiquid = false;

    glm::vec4 baseColorFactor               = glm::vec4(1.0f);
    float metallicFactor                    = 0.0f;
    float roughnessFactor                   = 1.0f;
    float baseColorTextureIndex             = -1.0f;
    float metallicRoughnessTextureIndex     = -1.0f;
    float normalTextureIndex                = -1.0f;
    float occlusionTextureIndex             = -1.0f;
    float emissiveTextureIndex              = -1.0f;

private:
    std::string name;
};

// ---------------------------------------------------------------------------
// Submesh range (one glTF primitive)
// ---------------------------------------------------------------------------
struct Mesh {
    uint32_t firstIndex    = 0;
    uint32_t indexCount    = 0;
    uint32_t materialIndex = 0;
};
