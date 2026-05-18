struct VSInput {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float2 UV : TEXCOORD0;
    float4 Tangent : TANGENT;
};

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrices[5];
    float4 directionalLightDirection;
    float4 directionalLightColor;    
    float4 lightPositions[16];
    float4 lightColors[16];
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

struct ShadowPushConstants { int cascadeIndex; };
[[vk::push_constant]] ShadowPushConstants pc;

[shader("vertex")]
float4 vertMain(VSInput input) : SV_POSITION
{
    return mul(ubo.lightSpaceMatrices[pc.cascadeIndex], mul(ubo.model, float4(input.Position, 1.0)));
}