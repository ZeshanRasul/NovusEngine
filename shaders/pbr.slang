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
};

// Uniform buffer - Global data shared across all vertices/fragments
struct UniformBufferObject {
    float4x4 model;                     // Model-to-world transformation matrix
    float4x4 view;                      // World-to-camera transformation matrix
    float4x4 proj;                      // Camera-to-clip space projection matrix
    float4 lightPositions[4];           // Light positions in world space
    float4 lightColors[4];              // Light intensities and colors
    float4 camPos;                      // Camera position for view-dependent effects
    float exposure;                     // HDR exposure control
    float gamma;                        // Gamma correction value (typically 2.2)
    float prefilteredCubeMipLevels;     // IBL prefiltered environment map mip levels
    float scaleIBLAmbient;              // IBL ambient contribution scale
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
    // Fragment shader needs world space position to calculate light vectors
    output.WorldPos = worldPos.xyz;

    // Transform normal from model space to world space
    // Use only rotation/scale part of model matrix (upper-left 3x3)
    // Normalize to ensure unit length after transformation
    output.Normal = normalize(mul((float3x3)ubo.model, input.Normal));

    // Pass through texture coordinates unchanged
    // UV coordinates are typically in [0,1] range and don't need transformation
    output.UV = input.UV;

    // Pass tangent vector for normal mapping
    // Will be used in fragment shader to construct tangent-space basis
    output.Tangent = input.Tangent;

    return output;
}

// Fragment shader entry point - Executes once per pixel
[[shader("fragment")]]
float4 fragMain(VSOutput input) : SV_TARGET
{
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
    for (int i = 0; i < 4; i++) {
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

    // === AMBIENT AND EMISSIVE ===
    // Add simple ambient lighting (should be replaced with IBL in production)
    float3 ambient = ubo.scaleIBLAmbient * baseColor.rgb * ao;

    // Combine all lighting contributions
    float3 color = ambient + Lo + emissive;

    // Apply exposure before tone mapping
    color *= ubo.exposure;

    // === HDR TONE MAPPING AND GAMMA CORRECTION ===
    // Apply Reinhard tone mapping to compress HDR values to [0,1] range
    color = color / (color + float3(1.0, 1.0, 1.0));

    // Apply gamma correction for sRGB display (inverse gamma)
    color = pow(color, float3(1.0 / ubo.gamma, 1.0 / ubo.gamma, 1.0 / ubo.gamma));

    // Output final color with original alpha
    return float4(color, baseColor.a);
}