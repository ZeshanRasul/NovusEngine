struct VSInput {
    float3 Position  : POSITION;
    float4 Color     : COLOR0;
    float3 Normal    : NORMAL;
    float2 UV        : TEXCOORD0;
    uint4  BoneIndex : BLENDINDICES;
    float4 BoneWeight: BLENDWEIGHT;
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

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> boneBuffer;

struct ShadowPushConstants { int cascadeIndex; };
[[vk::push_constant]] ShadowPushConstants pc;

[shader("vertex")]
float4 vertMain(VSInput input) : SV_POSITION
{
    float4 skinnedPos = float4(0.0, 0.0, 0.0, 0.0);

    float totalWeight = input.BoneWeight.x + input.BoneWeight.y
                      + input.BoneWeight.z + input.BoneWeight.w;

    if (totalWeight < 0.001)
    {
        skinnedPos = float4(input.Position, 1.0);
    }
    else
    {
        [unroll]
        for (int i = 0; i < 4; ++i)
        {
            float w = input.BoneWeight[i];
            if (w > 0.0)
            {
                uint boneIdx = input.BoneIndex[i];
                skinnedPos += w * mul(boneBuffer[boneIdx], float4(input.Position, 1.0));
            }
        }
    }

    return mul(ubo.lightSpaceMatrices[pc.cascadeIndex], mul(ubo.model, skinnedPos));
}
