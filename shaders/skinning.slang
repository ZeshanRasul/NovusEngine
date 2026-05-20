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

static const float PI = 3.14159265359;

struct BoneMatrices {
    float4x4 bones[256];
};

[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] StructuredBuffer<float4x4> boneBuffer;
[[vk::binding(2, 0)]] Texture2D diffuseMap;
[[vk::binding(2, 0)]] SamplerState diffuseSampler;
[[vk::binding(3, 0)]] TextureCube irradianceMap;
[[vk::binding(3, 0)]] SamplerState irradianceSampler;
[[vk::binding(4, 0)]] TextureCube prefilteredMap;
[[vk::binding(4, 0)]] SamplerState prefilteredSampler;
[[vk::binding(5, 0)]] Texture2D brdfLUT;
[[vk::binding(5, 0)]] SamplerState brdfSampler;

float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
{
    float3 maxRefl = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxRefl - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

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
    float3 V        = normalize(ubo.camPos.xyz - input.WorldPos);
    float3 R        = reflect(-V, N);
    float NdotV     = max(dot(N, V), 0.0);

    // Simple direct Lambert term
    float diff = max(dot(N, lightDir), 0.0);

    float4 texColor = diffuseMap.Sample(diffuseSampler, input.UV);
    float3 baseColor = texColor.rgb * input.Color.rgb;

    const float metallic = 0.0;
    const float roughness = 0.6;
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);

    float3 diffuseDirect = diff * ubo.directionalLightColor.rgb * baseColor;

    // IBL ambient
    float3 F = FresnelSchlickRoughness(NdotV, F0, roughness);
    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 irradiance = irradianceMap.Sample(irradianceSampler, N).rgb;
    float3 diffuseIBL = irradiance * baseColor;

    float maxLod = max(ubo.prefilteredCubeMipLevels - 1.0, 0.0);
    float3 prefilteredColor = prefilteredMap.SampleLevel(prefilteredSampler, R, roughness * maxLod).rgb;
    float2 brdf = brdfLUT.Sample(brdfSampler, float2(NdotV, roughness)).rg;
    float3 specularIBL = prefilteredColor * (F * brdf.x + brdf.y);

    float3 ambient = (kD * diffuseIBL + specularIBL) * ubo.scaleIBLAmbient;
    float3 color = ambient + diffuseDirect;

    return float4(color, texColor.a * input.Color.a);
}
