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
    float4 lightPositions[4];
    float4 lightColors[4];
    float4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
};

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

[shader("vertex")]
float4 vertMain(VSInput input) : SV_POSITION
{
    float4 worldPos = mul(ubo.model, float4(input.Position, 1.0));
    return mul(ubo.proj, mul(ubo.view, worldPos));
}