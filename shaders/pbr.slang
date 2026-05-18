// Combined vertex and fragment shader for PBR rendering

// Input from vertex buffer - Data sent per vertex from CPU
struct VSInput {
    float3 Position : POSITION;     // 3D position in model space
    float3 Normal : NORMAL;         // Surface normal for lighting calculations
    float2 UV : TEXCOORD0;          // Texture coordinates for material sampling
    float4 Tangent : TANGENT;       // Tangent vector for normal mapping (w component = handedness)
};

// Output from vertex shader / Input to fragment shader - Interpolated data
struct VSOutput {
    float4 Position : SV_POSITION; // Required clip space position for rasterization
    float3 WorldPos : POSITION;    // World space position for lighting calculations
    float3 Normal : NORMAL;        // World space normal (interpolated)
    float2 UV : TEXCOORD0;         // Texture coordinates (interpolated)
    float4 Tangent : TANGENT;      // World space tangent (interpolated)
    float  ViewDepth : TEXCOORD1;  // View-space depth for cascade selection
};

// Uniform buffer - Global data shared across all vertices/fragments
struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4x4 lightSpaceMatrices[5];     // One per shadow cascade
    float4 directionalLightDirection;
    float4 directionalLightColor;
    float4 lightPositions[16];
    float4 lightColors[16];
    float4 camPos;
    float4 cascadeSplits;               // xyz = split distances, w unused (float4 avoids std140 padding issues)
    float exposure;
    float gamma;
    float prefilteredCubeMipLevels;
    float scaleIBLAmbient;
    float4 shadowTuning;                // x = bias scale, y = min bias, z = blend factor
    float4 shadowDebug;                 // x = cascade debug view toggle
};

// Push constants - Fast, small data updated frequently per material/object
struct PushConstants {
    float4 baseColorFactor;             // Base color tint/multiplier
    float metallicFactor;               // Metallic property multiplier
    float roughnessFactor;              // Surface roughness multiplier
    int baseColorTextureSet;            // Texture binding index for base color (-1 = none)
    int physicalDescriptorTextureSet;   // Texture binding for metallic/roughness
    int normalTextureSet;               // Texture binding for normal maps
    int occlusionTextureSet;            // Texture binding for ambient occlusion
    int emissiveTextureSet;             // Texture binding for emissive maps
    float alphaMask;                    // Alpha masking enable flag
    float alphaMaskCutoff;              // Alpha cutoff threshold
};

// Mathematical constants
static const float PI = 3.14159265359;

// Resource bindings - Connect CPU resources to GPU shader registers
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;
[[vk::binding(1, 0)]] Texture2D baseColorMap;
[[vk::binding(1, 0)]] SamplerState baseColorSampler;
[[vk::binding(2, 0)]] Texture2D metallicRoughnessMap;
[[vk::binding(2, 0)]] SamplerState metallicRoughnessSampler;
[[vk::binding(3, 0)]] Texture2D normalMap;
[[vk::binding(3, 0)]] SamplerState normalSampler;
[[vk::binding(4, 0)]] Texture2D occlusionMap;
[[vk::binding(4, 0)]] SamplerState occlusionSampler;
[[vk::binding(5, 0)]] Texture2D emissiveMap;
[[vk::binding(5, 0)]] SamplerState emissiveSampler;
[[vk::binding(6, 0)]] Texture2D shadowMap0;
[[vk::binding(6, 0)]] SamplerState shadowSampler0;
[[vk::binding(7, 0)]] Texture2D shadowMap1;
[[vk::binding(7, 0)]] SamplerState shadowSampler1;
[[vk::binding(8, 0)]] Texture2D shadowMap2;
[[vk::binding(8, 0)]] SamplerState shadowSampler2;
[[vk::binding(9, 0)]] Texture2D shadowMap3;
[[vk::binding(9, 0)]] SamplerState shadowSampler3;
[[vk::binding(10, 0)]] Texture2D shadowMap4;
[[vk::binding(10, 0)]] SamplerState shadowSampler4;

[[vk::push_constant]] PushConstants material;

// Normal Distribution Function (D) - GGX/Trowbridge-Reitz Distribution
// Describes the statistical distribution of microfacet orientations
float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;        // Remapping for more perceptual linearity
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;                         // Numerator: concentration factor
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;             // Normalization factor

    return nom / denom;                     // Normalized distribution
}

// Geometry Function (G) - Smith's method with Schlick-GGX approximation
// Models self-shadowing and masking between microfacets
float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;               // Direct lighting remapping

    // Geometry obstruction from view direction (masking)
    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    // Geometry obstruction from light direction (shadowing)
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);

    return ggx1 * ggx2;                     // Combined masking-shadowing
}

// Fresnel Reflectance (F) - Schlick's approximation
// Models how reflectance changes with viewing angle
float3 FresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

float SampleShadowMap(int cascadeIndex, float2 uv)
{
   if      (cascadeIndex == 0) return shadowMap0.Sample(shadowSampler0, uv).r;
    else if (cascadeIndex == 1) return shadowMap1.Sample(shadowSampler1, uv).r;
    else if (cascadeIndex == 2) return shadowMap2.Sample(shadowSampler2, uv).r;
    else if (cascadeIndex == 3) return shadowMap3.Sample(shadowSampler3, uv).r;
    else                        return shadowMap4.Sample(shadowSampler4, uv).r;
}

float ComputeCascadeShadow(int cascadeIndex, float3 worldPos, float3 normal, float3 lightDir)
{
    float4x4 lightSpaceMat = ubo.lightSpaceMatrices[cascadeIndex];
    float4 fragPosLightSpace = mul(lightSpaceMat, float4(worldPos, 1.0));
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.xy = projCoords.xy * 0.5 + 0.5;

    if (projCoords.z > 1.0 || projCoords.z < 0.0) return 0.0;
    if (any(projCoords.xy < 0.0) || any(projCoords.xy > 1.0)) return 0.0;

    float bias = max(ubo.shadowTuning.x * (1.0 - dot(normal, lightDir)), ubo.shadowTuning.y);
    float currentDepth = projCoords.z;
    float2 texelSize = float2(1.0 / 2048.0, 1.0 / 2048.0);
    float shadow = 0.0;

    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            float2 offset = float2(x, y) * texelSize;
            float closestDepth = SampleShadowMap(cascadeIndex, projCoords.xy + offset);
            shadow += currentDepth - bias > closestDepth ? 1.0 : 0.0;
        }
    }

    return shadow / 9.0;
}

float ShadowCalculation(float viewDepth, float3 worldPos, float3 normal, float3 lightDir)
{
    if (ubo.shadowTuning.w < 0.5)
        return 0.0;

    int cascadeIndex = 4;
    if      (viewDepth < ubo.cascadeSplits.x) cascadeIndex = 0;
    else if (viewDepth < ubo.cascadeSplits.y) cascadeIndex = 1;
    else if (viewDepth < ubo.cascadeSplits.z) cascadeIndex = 2;
    else if (viewDepth < ubo.cascadeSplits.w) cascadeIndex = 3;

    float shadow = ComputeCascadeShadow(cascadeIndex, worldPos, normal, lightDir);

    if (cascadeIndex < 4) {
        float splitDepth = cascadeIndex == 0 ? ubo.cascadeSplits.x
            : cascadeIndex == 1 ? ubo.cascadeSplits.y
            : cascadeIndex == 2 ? ubo.cascadeSplits.z
            : ubo.cascadeSplits.w;
        float previousSplit = cascadeIndex == 0 ? 0.0
            : cascadeIndex == 1 ? ubo.cascadeSplits.x
            : cascadeIndex == 2 ? ubo.cascadeSplits.y
            : ubo.cascadeSplits.z;
        float transitionRange = max((splitDepth - previousSplit) * ubo.shadowTuning.z, 10.0);
        float blendStart = splitDepth - transitionRange;

        if (viewDepth > blendStart) {
            float nextShadow = ComputeCascadeShadow(cascadeIndex + 1, worldPos, normal, lightDir);
            float blend = saturate((viewDepth - blendStart) / transitionRange);
            shadow = lerp(shadow, nextShadow, blend);
        }
    }

  return shadow;
}


// Vertex shader entry point - Executes once per vertex
[[shader("vertex")]]
VSOutput vertMain(VSInput input)
{
    VSOutput output;

    // Transform vertex position through the rendering pipeline
    // Model -> World -> Camera -> Clip space transformation chain
    float4 worldPos = mul(ubo.model, float4(input.Position, 1.0));
    output.Position = mul(ubo.proj, mul(ubo.view, worldPos));

    // Pass world position for fragment lighting calculations
    output.WorldPos = worldPos.xyz;

    output.Normal = normalize(mul((float3x3)ubo.model, input.Normal));
    output.UV = input.UV;
    output.Tangent = input.Tangent;

    // Store view-space Z for cascade selection in the fragment shader
    float4 viewPos = mul(ubo.view, worldPos);
    output.ViewDepth = -viewPos.z; // Negate: view space Z is negative in front of camera

    return output;
}

// Fragment shader entry point - Executes once per pixel
[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
    int cascadeIndex = 4;
    if      (input.ViewDepth < ubo.cascadeSplits.x) cascadeIndex = 0;
    else if (input.ViewDepth < ubo.cascadeSplits.y) cascadeIndex = 1;
    else if (input.ViewDepth < ubo.cascadeSplits.z) cascadeIndex = 2;
    else if (input.ViewDepth < ubo.cascadeSplits.w) cascadeIndex = 3;

    // === MATERIAL PROPERTY SAMPLING ===
    // Start from glTF defaults and only sample textures when they exist.
    float4 baseColor = material.baseColorFactor;
    if (material.baseColorTextureSet >= 0) {
        baseColor *= baseColorMap.Sample(baseColorSampler, input.UV);
    }

    // glTF standard: metallic stored in blue, roughness in green
    float metallic = saturate(material.metallicFactor);
    float roughness = clamp(material.roughnessFactor, 0.04, 1.0);
    if (material.physicalDescriptorTextureSet >= 0) {
        float2 metallicRoughness = metallicRoughnessMap.Sample(metallicRoughnessSampler, input.UV).bg;
        metallic = saturate(metallicRoughness.x * material.metallicFactor);
        roughness = clamp(metallicRoughness.y * material.roughnessFactor, 0.04, 1.0);
    }

    float ao = 1.0;
    if (material.occlusionTextureSet >= 0) {
        ao = occlusionMap.Sample(occlusionSampler, input.UV).r;
    }

    float3 emissive = float3(0.0, 0.0, 0.0);
    if (material.emissiveTextureSet >= 0) {
        emissive = emissiveMap.Sample(emissiveSampler, input.UV).rgb;
    }

    // === NORMAL CALCULATION ===
    // Start with interpolated surface normal
    float3 N = normalize(input.Normal);

    // Apply normal mapping if texture is available
    if (material.normalTextureSet >= 0) {
        // Sample normal map and convert from [0,1] to [-1,1] range
        float3 tangentNormal = normalMap.Sample(normalSampler, input.UV).xyz * 2.0 - 1.0;

        // Construct tangent-space to world-space transformation matrix (TBN)
        float3 T = normalize(input.Tangent.xyz);              // Tangent
        float3 B = normalize(cross(N, T)) * input.Tangent.w;  // Bitangent (w = handedness)
        float3x3 TBN = float3x3(T, B, N);                     // Tangent-Bitangent-Normal matrix

        // Transform normal from tangent space to world space
        N = normalize(mul(tangentNormal, TBN));
    }

    // === LIGHTING SETUP ===
    // Calculate view direction (fragment to camera)
    float3 V = normalize(ubo.camPos.xyz - input.WorldPos);

    // Calculate reflection vector for environment mapping
    float3 R = reflect(-V, N);

    // === PBR MATERIAL SETUP ===
    // Calculate F0 (reflectance at normal incidence)
    // Non-metals: low reflectance (~0.04), Metals: colored reflectance from base color
    float3 F0 = float3(0.04, 0.04, 0.04);  // Dielectric default
    F0 = lerp(F0, baseColor.rgb, metallic); // Lerp to metallic behavior

    // Initialize outgoing radiance accumulator
    float3 Lo = float3(0.0, 0.0, 0.0);
    // === DIRECT LIGHTING LOOP ===
    // Calculate contribution from each light source
    for (int i = 0; i < 16; i++) {
        float3 lightPos = ubo.lightPositions[i].xyz;
        float3 lightColor = ubo.lightColors[i].rgb;

        // Calculate light direction and attenuation
        float3 toLight = lightPos - input.WorldPos;
        float distance = length(toLight);
        float3 L = toLight / max(distance, 0.0001);

        // Softer point-light attenuation than strict inverse-square.
        // This makes authored light colors/intensities much easier to read 
        // in a small test scene without requiring extremely large values.
        float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);
        float3 radiance = lightColor * attenuation;

        // Calculate half vector (between view and light directions)
        float3 H = normalize(V + L);

        // === BRDF EVALUATION ===
        // Calculate all necessary dot products for BRDF terms
        float NdotL = max(dot(N, L), 0.0);  // Lambertian falloff
        float NdotV = max(dot(N, V), 0.0);  // View angle
        float NdotH = max(dot(N, H), 0.0);  // Half vector for specular
        float HdotV = max(dot(H, V), 0.0);  // For Fresnel calculation

        // Evaluate Cook-Torrance BRDF components
        float D = DistributionGGX(NdotH, roughness);    // Normal distribution
        float G = GeometrySmith(NdotV, NdotL, roughness); // Geometry function
        float3 F = FresnelSchlick(HdotV, F0);           // Fresnel reflectance

        // Calculate specular BRDF
        float3 numerator = D * G * F;
        float denominator = 4.0 * NdotV * NdotL + 0.0001; // Prevent division by zero
        float3 specular = numerator / denominator;

        // === ENERGY CONSERVATION ===
        // Fresnel term represents specular reflection ratio
        float3 kS = F;                          // Specular contribution
        float3 kD = float3(1.0, 1.0, 1.0) - kS; // Diffuse contribution (energy conservation)
        kD *= 1.0 - metallic;                   // Metals have no diffuse reflection

        // === RADIANCE ACCUMULATION ===
        // Combine diffuse (Lambertian) and specular (Cook-Torrance) terms
        // Multiply by incident radiance and cosine foreshortening
        Lo += (kD * baseColor.rgb / PI + specular) * radiance * NdotL;
    }
    
    // === DIRECTIONAL LIGHTING ===
    float3 dirLightDir = normalize(ubo.directionalLightDirection.xyz);
    float3 L = normalize(-dirLightDir);
    float3 dirLightColor = ubo.directionalLightColor.rgb;
    float NdotL = max(dot(N, L), 0.0);
    float3 H = normalize(V + L);
    float3 diffuse = baseColor.rgb / PI; // Lambertian diffuse
    float D = DistributionGGX(max(dot(N, H), 0.0), roughness);
    float G = GeometrySmith(max(dot(N, V), 0.0), NdotL, roughness);
    float F = FresnelSchlick(max(dot(H, V), 0.0), F0).r; // Assuming grayscale F0 for simplicity
    float3 specular = (D * G * F) / max(4.0 * max(dot(N, V), 0.0) * NdotL, 0.001);
    float shadow = ShadowCalculation(input.ViewDepth, input.WorldPos, N, L);

    if (ubo.shadowDebug.x > 0.5) {
        float3 cascadeColor = cascadeIndex == 0 ? float3(1.0, 0.2, 0.2)
            : cascadeIndex == 1 ? float3(0.2, 1.0, 0.2)
            : cascadeIndex == 2 ? float3(0.2, 0.4, 1.0)
            : cascadeIndex == 3 ? float3(1.0, 1.0, 0.2)
            : float3(1.0, 0.2, 1.0);
        return float4(cascadeColor, 1.0);
    }

    Lo += (1.0 - shadow) * (diffuse * (1.0 - metallic) + specular) * dirLightColor * NdotL;

    // === AMBIENT AND EMISSIVE ===
    // Add simple ambient lighting (should be replaced with IBL in production)
    float3 ambient = ubo.scaleIBLAmbient * baseColor.rgb * ao;

    // Combine all lighting contributions
    float3 color = ambient + Lo + emissive;

    // Output final color with original alpha
    return float4(color, baseColor.a);
}