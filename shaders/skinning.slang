// Skinned mesh vertex + fragment shader
// Vertex layout matches VkVertex: position, color, normal, uv, boneNumber(uvec4), boneWeight(vec4)

struct VSInput {
    float3 Position  : POSITION;
    float4 Color     : COLOR0;
    float3 Normal    : NORMAL;
    float2 UV        : TEXCOORD0;
    uint4  BoneIndex : BLENDINDICES;
    float4 BoneWeight: BLENDWEIGHT;
};

struct VSOutput {
    float4 Position  : SV_POSITION;
    float3 WorldPos  : POSITION;
    float3 Normal    : NORMAL;
    float2 UV        : TEXCOORD0;
    float4 Color     : COLOR0;
};

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrices[5];
    float4 directionalLightDirection;
    float4 directionalLightColor;
    float4 lightPositions[4];
    float4 lightColors[4];
    float4 camPos;
    float4 cascadeSplits;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float4 shadowTuning;
    float4 shadowDebug;
};

struct BoneMatrices {
    float4x4 bones[256];
};

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> boneBuffer;
[[vk::binding(2, 0)]] Texture2D diffuseMap;
[[vk::binding(2, 0)]] SamplerState diffuseSampler;

[[shader("vertex")]]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    // Accumulate skinned position and normal
    float4 skinnedPos    = float4(0.0, 0.0, 0.0, 0.0);
    float3 skinnedNormal = float3(0.0, 0.0, 0.0);

    float totalWeight = input.BoneWeight.x + input.BoneWeight.y
                      + input.BoneWeight.z + input.BoneWeight.w;

    // Guard: if no bones are influencing, fall back to identity transform
    if (totalWeight < 0.001) {
        skinnedPos    = float4(input.Position, 1.0);
        skinnedNormal = input.Normal;
    } else {
        [unroll]
        for (int i = 0; i < 4; ++i) {
            float w = input.BoneWeight[i];
            if (w > 0.0) {
                uint  boneIdx = input.BoneIndex[i];
                float4x4 boneMat = boneBuffer[boneIdx];
                skinnedPos    += w * mul(boneMat, float4(input.Position, 1.0));
                skinnedNormal += w * mul((float3x3)boneMat, input.Normal);
            }
        }
        // Re-normalise the accumulated normal
        skinnedNormal = normalize(skinnedNormal);
    }

    float4 worldPos   = mul(ubo.model, skinnedPos);
    output.Position   = mul(ubo.proj, mul(ubo.view, worldPos));
    output.WorldPos   = worldPos.xyz;
    output.Normal     = normalize(mul((float3x3)ubo.model, skinnedNormal));
    output.UV         = input.UV;
    output.Color      = input.Color;

    return output;
}

[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float3 lightDir = normalize(-ubo.directionalLightDirection.xyz);
    float3 N        = normalize(input.Normal);

    // Lambertian diffuse
    float diff = max(dot(N, lightDir), 0.0);

    float4 texColor = diffuseMap.Sample(diffuseSampler, input.UV);
    // Fallback to vertex color if texture is the default white (sampler will return 1,1,1,1)
    float3 baseColor = texColor.rgb * input.Color.rgb;

    float3 ambient  = 0.15 * ubo.directionalLightColor.rgb * baseColor;
    float3 diffuse  = diff * ubo.directionalLightColor.rgb * baseColor;

    return float4(ambient + diffuse, texColor.a * input.Color.a);
}
