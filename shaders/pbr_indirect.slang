// PBR shader for GPU-driven indirect rendering.
// Differences from pbr.slang:
//   - No per-object UBO; model matrix fetched from ObjectData SSBO via SV_StartInstanceLocation.
//   - No push constants; material properties fetched from MaterialData SSBO.
//   - All PBR textures accessed through a bindless Texture2D array (set 1).

// ---------------------------------------------------------------------------
// Vertex input
// ---------------------------------------------------------------------------
struct VSInput {
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
    float2 UV       : TEXCOORD0;
    float4 Tangent  : TANGENT;
};

struct VSOutput {
    float4 Position  : SV_POSITION;
    float3 WorldPos  : POSITION;
    float3 Normal    : NORMAL;
    float2 UV        : TEXCOORD0;
    float4 Tangent   : TANGENT;
    float  ViewDepth : TEXCOORD1;
    uint   ObjectIdx : TEXCOORD2;   // forwarded to fragment for material lookup
};

// ---------------------------------------------------------------------------
// Per-frame global data (set 0, binding 0)
// ---------------------------------------------------------------------------
struct FrameData {
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrices[5];
    float4   directionalLightDirection;
    float4   directionalLightColor;
    float4   lightPositions[16];
    float4   lightColors[16];
    float4   camPos;
    float4   cascadeSplits;
    float    exposure;
    float    gamma;
    float    prefilteredCubeMipLevels;
    float    scaleIBLAmbient;
    float4   shadowTuning;
    float4   shadowDebug;
};

// ---------------------------------------------------------------------------
// Per-object data (set 0, binding 1)
// ---------------------------------------------------------------------------
struct ObjectData {
    float4x4 modelMatrix;
    float4   boundingSphere;
    uint     materialIndex;
    uint     _pad0;
    uint     _pad1;
    uint     _pad2;
};

// ---------------------------------------------------------------------------
// Per-material data (set 0, binding 2)
// ---------------------------------------------------------------------------
struct MaterialData {
    float4 baseColorFactor;
    float  metallicFactor;
    float  roughnessFactor;
    float  alphaMask;
    float  alphaMaskCutoff;
    int    baseColorTexIdx;
    int    metallicRoughnessTexIdx;
    int    normalTexIdx;
    int    occlusionTexIdx;
    int    emissiveTexIdx;
    uint   _pad0;
    uint   _pad1;
    uint   _pad2;
};

// ---------------------------------------------------------------------------
// Resource bindings
// ---------------------------------------------------------------------------
// set 0 — global frame data
[[vk::binding(0, 0)]] ConstantBuffer<FrameData>                 frame;
[[vk::binding(1, 0)]] StructuredBuffer<ObjectData>              objectData;
[[vk::binding(2, 0)]] StructuredBuffer<MaterialData>            materialData;

// Shadow cascade maps (bindings 3–7)
[[vk::binding(3, 0)]] Texture2D      shadowMap0; [[vk::binding(3, 0)]] SamplerState shadowSampler0;
[[vk::binding(4, 0)]] Texture2D      shadowMap1; [[vk::binding(4, 0)]] SamplerState shadowSampler1;
[[vk::binding(5, 0)]] Texture2D      shadowMap2; [[vk::binding(5, 0)]] SamplerState shadowSampler2;
[[vk::binding(6, 0)]] Texture2D      shadowMap3; [[vk::binding(6, 0)]] SamplerState shadowSampler3;
[[vk::binding(7, 0)]] Texture2D      shadowMap4; [[vk::binding(7, 0)]] SamplerState shadowSampler4;

// IBL (bindings 8–10)
[[vk::binding(8,  0)]] TextureCube   irradianceMap;  [[vk::binding(8,  0)]] SamplerState irradianceSampler;
[[vk::binding(9,  0)]] TextureCube   prefilteredMap; [[vk::binding(9,  0)]] SamplerState prefilteredSampler;
[[vk::binding(10, 0)]] Texture2D     brdfLUT;        [[vk::binding(10, 0)]] SamplerState brdfSampler;

// set 1 — bindless PBR textures (UPDATE_AFTER_BIND, PARTIALLY_BOUND)
[[vk::binding(0, 1)]] Texture2D      allTextures[];
[[vk::binding(0, 1)]] SamplerState   allSamplers[];

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const float PI = 3.14159265359;

// ---------------------------------------------------------------------------
// BRDF helpers — identical to pbr.slang
// ---------------------------------------------------------------------------
float DistributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) / 8.0;
    float g1 = NdotV / (NdotV * (1.0 - k) + k);
    float g2 = NdotL / (NdotL * (1.0 - k) + k);
    return g1 * g2;
}

float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 FresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    float3 maxRefl = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0);
    return F0 + (maxRefl - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ---------------------------------------------------------------------------
// Shadow helpers — identical logic, same cascade maps
// ---------------------------------------------------------------------------
float SampleShadowMap(int ci, float2 uv) {
    if      (ci == 0) return shadowMap0.Sample(shadowSampler0, uv).r;
    else if (ci == 1) return shadowMap1.Sample(shadowSampler1, uv).r;
    else if (ci == 2) return shadowMap2.Sample(shadowSampler2, uv).r;
    else if (ci == 3) return shadowMap3.Sample(shadowSampler3, uv).r;
    else              return shadowMap4.Sample(shadowSampler4, uv).r;
}

float ComputeCascadeShadow(int ci, float3 worldPos, float3 normal, float3 lightDir) {
    float4x4 lsm = frame.lightSpaceMatrices[ci];
    float3   pc  = (mul(lsm, float4(worldPos, 1.0)).xyz / mul(lsm, float4(worldPos, 1.0)).w);
    pc.xy = pc.xy * 0.5 + 0.5;

    if (pc.z > 1.0 || pc.z < 0.0)              return 0.0;
    if (any(pc.xy < 0.0) || any(pc.xy > 1.0))  return 0.0;

    float bias = max(frame.shadowTuning.x * (1.0 - dot(normal, lightDir)), frame.shadowTuning.y);
    float2 ts  = float2(1.0 / 2048.0, 1.0 / 2048.0);
    float shadow = 0.0;
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x)
        shadow += (pc.z - bias) > SampleShadowMap(ci, pc.xy + float2(x, y) * ts) ? 1.0 : 0.0;
    return shadow / 9.0;
}

float ShadowCalculation(float viewDepth, float3 worldPos, float3 normal, float3 lightDir) {
    if (frame.shadowTuning.w < 0.5) return 0.0;
    int ci = 4;
    if      (viewDepth < frame.cascadeSplits.x) ci = 0;
    else if (viewDepth < frame.cascadeSplits.y) ci = 1;
    else if (viewDepth < frame.cascadeSplits.z) ci = 2;
    else if (viewDepth < frame.cascadeSplits.w) ci = 3;

    float shadow = ComputeCascadeShadow(ci, worldPos, normal, lightDir);

    if (ci < 4) {
        float splitDepth  = ci == 0 ? frame.cascadeSplits.x : ci == 1 ? frame.cascadeSplits.y : ci == 2 ? frame.cascadeSplits.z : frame.cascadeSplits.w;
        float prevSplit   = ci == 0 ? 0.0 : ci == 1 ? frame.cascadeSplits.x : ci == 2 ? frame.cascadeSplits.y : frame.cascadeSplits.z;
        float tr          = max((splitDepth - prevSplit) * frame.shadowTuning.z, 10.0);
        float blendStart  = splitDepth - tr;
        if (viewDepth > blendStart) {
            float nextShadow = ComputeCascadeShadow(ci + 1, worldPos, normal, lightDir);
            shadow = lerp(shadow, nextShadow, saturate((viewDepth - blendStart) / tr));
        }
    }
    return shadow;
}

// ---------------------------------------------------------------------------
// Vertex shader
// SV_StartInstanceLocation carries firstInstance, which we set to objectIndex
// when building the indirect draw commands.
// ---------------------------------------------------------------------------
[[shader("vertex")]]
VSOutput vertMain(VSInput input, uint objectIdx : SV_StartInstanceLocation)
{
    ObjectData obj = objectData[objectIdx];

    float4 worldPos  = mul(obj.modelMatrix, float4(input.Position, 1.0));
    float4 clipPos   = mul(frame.proj, mul(frame.view, worldPos));
    float4 viewPos   = mul(frame.view, worldPos);

    VSOutput o;
    o.Position  = clipPos;
    o.WorldPos  = worldPos.xyz;
    o.Normal    = normalize(mul((float3x3)obj.modelMatrix, input.Normal));
    o.UV        = input.UV;
    o.Tangent   = input.Tangent;
    o.ViewDepth = -viewPos.z;
    o.ObjectIdx = objectIdx;
    return o;
}

// ---------------------------------------------------------------------------
// Helper: sample a bindless texture, return default if index is negative.
// ---------------------------------------------------------------------------
float4 sampleBindless(int texIdx, float2 uv, float4 defaultVal) {
    if (texIdx < 0) return defaultVal;
    return allTextures[NonUniformResourceIndex(texIdx)].Sample(
               allSamplers[NonUniformResourceIndex(texIdx)], uv);
}

// ---------------------------------------------------------------------------
// Fragment shader
// ---------------------------------------------------------------------------
[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
    ObjectData   obj = objectData[input.ObjectIdx];
    MaterialData mat = materialData[obj.materialIndex];

    // Cascade index for debug view
    int ci = 4;
    if      (input.ViewDepth < frame.cascadeSplits.x) ci = 0;
    else if (input.ViewDepth < frame.cascadeSplits.y) ci = 1;
    else if (input.ViewDepth < frame.cascadeSplits.z) ci = 2;
    else if (input.ViewDepth < frame.cascadeSplits.w) ci = 3;

    // --- Material sampling ---
    float4 baseColor = sampleBindless(mat.baseColorTexIdx, input.UV, float4(1.0, 1.0, 1.0, 1.0));
    baseColor *= mat.baseColorFactor;

    // Alpha masking
    if (mat.alphaMask > 0.5 && baseColor.a < mat.alphaMaskCutoff) discard;

    float metallic  = saturate(mat.metallicFactor);
    float roughness = clamp(mat.roughnessFactor, 0.04, 1.0);

    float4 mrSample = sampleBindless(mat.metallicRoughnessTexIdx, input.UV, float4(0.0, 1.0, 0.0, 0.0));
    if (mat.metallicRoughnessTexIdx >= 0) {
        metallic  = saturate(mrSample.b * mat.metallicFactor);
        roughness = clamp(mrSample.g * mat.roughnessFactor, 0.04, 1.0);
    }

    float ao      = sampleBindless(mat.occlusionTexIdx, input.UV, float4(1.0, 1.0, 1.0, 1.0)).r;
    float3 emissive = sampleBindless(mat.emissiveTexIdx, input.UV, float4(0.0, 0.0, 0.0, 0.0)).rgb;

    // --- Normal mapping ---
    float3 N = normalize(input.Normal);
    if (mat.normalTexIdx >= 0) {
        float3 tn  = sampleBindless(mat.normalTexIdx, input.UV, float4(0.5, 0.5, 1.0, 0.0)).xyz * 2.0 - 1.0;
        float3 T   = normalize(input.Tangent.xyz);
        float3 B   = normalize(cross(N, T)) * input.Tangent.w;
        N = normalize(mul(tn, float3x3(T, B, N)));
    }

    float3 V  = normalize(frame.camPos.xyz - input.WorldPos);
    float3 R  = reflect(-V, N);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), baseColor.rgb, metallic);

    // --- Point lights ---
    float3 Lo = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < 16; ++i) {
        float3 toL  = frame.lightPositions[i].xyz - input.WorldPos;
        float  dist = length(toL);
        float3 L    = toL / max(dist, 0.0001);
        float  att  = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        float3 rad  = frame.lightColors[i].rgb * att;
        float3 H    = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);
        float  D    = DistributionGGX(NdotH, roughness);
        float  G    = GeometrySmith(NdotV, NdotL, roughness);
        float3 F    = FresnelSchlick(HdotV, F0);
        float3 spec = D * G * F / max(4.0 * NdotV * NdotL, 0.0001);
        float3 kD   = (1.0 - F) * (1.0 - metallic);
        Lo += (kD * baseColor.rgb / PI + spec) * rad * NdotL;
    }

    // --- Directional light + shadow ---
    float3 dirL  = normalize(-frame.directionalLightDirection.xyz);
    float3 Hdir  = normalize(V + dirL);
    float NdotL  = max(dot(N, dirL), 0.0);
    float NdotV  = max(dot(N, V), 0.0);
    float NdotH  = max(dot(N, Hdir), 0.0);
    float  D     = DistributionGGX(NdotH, roughness);
    float  G     = GeometrySmith(NdotV, NdotL, roughness);
    float  F     = FresnelSchlick(max(dot(Hdir, V), 0.0), F0).r;
    float3 spec  = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
    float shadow = ShadowCalculation(input.ViewDepth, input.WorldPos, N, dirL);

    if (frame.shadowDebug.x > 0.5) {
        float3 cc = ci == 0 ? float3(1, 0.2, 0.2) : ci == 1 ? float3(0.2, 1, 0.2)
                  : ci == 2 ? float3(0.2, 0.4, 1)  : ci == 3 ? float3(1, 1, 0.2)
                  : float3(1, 0.2, 1);
        return float4(cc, 1.0);
    }

    Lo += (1.0 - shadow) * (baseColor.rgb / PI * (1.0 - metallic) + spec)
        * frame.directionalLightColor.rgb * NdotL;

    // --- IBL ---
    float3 kS_ibl  = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 kD_ibl  = (1.0 - kS_ibl) * (1.0 - metallic);
    float3 irr     = irradianceMap.Sample(irradianceSampler, N).rgb;
    float3 prefiltered = prefilteredMap.SampleLevel(prefilteredSampler, R,
                            roughness * frame.prefilteredCubeMipLevels).rgb;
    float2 envBRDF = brdfLUT.Sample(brdfSampler, float2(max(dot(N, V), 0.0), roughness)).rg;
    float3 ambient = (kD_ibl * irr * baseColor.rgb +
                      prefiltered * (kS_ibl * envBRDF.x + envBRDF.y)) * ao * frame.scaleIBLAmbient;

    float3 color = ambient + Lo + emissive;
    return float4(color, baseColor.a);
}
