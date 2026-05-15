// FXAA post-processing pass
// VS: generates a fullscreen triangle from gl_VertexIndex — no vertex buffer needed
// FS: runs FXAA 3.11 quality on the offscreen scene color image

struct VSOutput {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

// Binding 0: the offscreen scene color image rendered by the main pass
[[vk::binding(0, 0)]] Texture2D    sceneColor;
[[vk::binding(0, 0)]] SamplerState sceneSampler;
[[vk::binding(1, 0)]] Texture2D    bloomColor;
[[vk::binding(1, 0)]] SamplerState bloomSampler;
[[vk::binding(2, 0)]] Texture2D    sceneDepth;
[[vk::binding(2, 0)]] SamplerState depthSampler;

struct FxaaPushConstants
{
    float2 rcpFrame;
    float exposure;
    float gamma;
    float bloomIntensity;
    int debugMode;
    float2 _pad;
};

[[vk::push_constant]] FxaaPushConstants pc;

// ---------------------------------------------------------------------------
// Vertex shader — fullscreen triangle (no vertex buffer)
// Vertex 0: (-1, -1) → UV (0, 0)
// Vertex 1: ( 3, -1) → UV (2, 0)
// Vertex 2: (-1,  3) → UV (0, 2)
// The oversized triangle covers the entire NDC screen in one draw call
// ---------------------------------------------------------------------------
[shader("vertex")]
VSOutput vertMain(uint vertexIndex : SV_VertexID)
{
    VSOutput output;
    float2 pos = float2(
        (vertexIndex == 1) ?  3.0 : -1.0,
        (vertexIndex == 2) ?  3.0 : -1.0
    );
    output.Position = float4(pos, 0.0, 1.0);
    output.UV       = pos * float2(0.5, 0.5) + float2(0.5, 0.5);
    return output;
}

// ---------------------------------------------------------------------------
// FXAA 3.11 quality implementation
// Algorithm:
//   1. Sample luma (perceived brightness) at current pixel and 4 neighbours
//   2. Measure local contrast — skip pixels in flat regions (no edge)
//   3. Find the dominant edge direction (horizontal vs vertical)
//   4. Step along the edge in both directions to find its endpoints
//   5. Blend the pixel toward its neighbour proportional to how near the
//      endpoints are — this sub-pixel shift removes the jagged appearance
// ---------------------------------------------------------------------------

// Luma converts RGB to perceived brightness using the green-weighted formula.
// Green contributes most to human brightness perception.
float luma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

float3 TonemapACESFilm(float3 x)
{
    // Narkowicz ACES approximation
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 LinearToSRGB(float3 linearColor, float gamma)
{
    return pow(saturate(linearColor), float3(1.0 / gamma));
}

float LinearizeDepth(float depth)
{
    static const float nearPlane = 0.1;
    static const float farPlane = 3000.0;
    float z = depth * 2.0 - 1.0;
    return (2.0 * nearPlane * farPlane) / (farPlane + nearPlane - z * (farPlane - nearPlane));
}

float3 SampleTonemapped(float2 uv, float exposure, float gamma)
{
    static const float3 fogColor = float3(0.62, 0.70, 0.82);
    static const float fogDensity = 0.15;
    static const float fogMax = 0.9;

    float3 hdr = sceneColor.Sample(sceneSampler, uv).rgb; // linear HDR

    if (pc.debugMode == 1)
    {
        float3 mappedScene = TonemapACESFilm(hdr * exposure);
        return LinearToSRGB(mappedScene, gamma);
    }

    float3 bloom = bloomColor.Sample(bloomSampler, uv).rgb;
    if (pc.debugMode == 2)
    {
        float3 mappedBloom = TonemapACESFilm(bloom * exposure);
        return LinearToSRGB(mappedBloom, gamma);
    }

    hdr += bloom * pc.bloomIntensity;

    float depth = sceneDepth.Sample(depthSampler, uv).r;
    float distanceToCamera = LinearizeDepth(depth);
    float fogAmount = 1.0 - exp(-distanceToCamera * fogDensity);
    fogAmount = saturate(fogAmount);
    fogAmount = min(fogAmount, fogMax);

    hdr = lerp(hdr, fogColor, fogAmount);

    float3 mapped = TonemapACESFilm(hdr * exposure);
    return LinearToSRGB(mapped, gamma);
}


[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float2 uv = input.UV;

    float exposure = pc.exposure;
    float gamma = pc.gamma;

    float2 rcpFrame = pc.rcpFrame;

    // --- 1. Sample luma at center and cardinal neighbours ---
    float lumaCenter = luma(SampleTonemapped(uv, exposure, gamma));
    float lumaUp     = luma(SampleTonemapped(uv + float2( 0.0,  rcpFrame.y), exposure, gamma));
    float lumaDown   = luma(SampleTonemapped(uv + float2( 0.0, -rcpFrame.y), exposure, gamma));
    float lumaLeft   = luma(SampleTonemapped(uv + float2(-rcpFrame.x,  0.0), exposure, gamma));
    float lumaRight  = luma(SampleTonemapped(uv + float2( rcpFrame.x,  0.0), exposure, gamma));

    float lumaMin = min(lumaCenter, min(min(lumaUp, lumaDown), min(lumaLeft, lumaRight)));
    float lumaMax = max(lumaCenter, max(max(lumaUp, lumaDown), max(lumaLeft, lumaRight)));

    // --- 2. Early exit for flat regions — no visible edge here ---
    float lumaRange = lumaMax - lumaMin;
    static const float EDGE_THRESHOLD_MIN = 0.0312; // Ignore very dark regions
    static const float EDGE_THRESHOLD     = 0.125;  // Minimum contrast to trigger AA
    if (lumaRange < max(EDGE_THRESHOLD_MIN, lumaMax * EDGE_THRESHOLD))
        {
            float3 finalColor = SampleTonemapped(uv, exposure, gamma);
            return float4(finalColor, 1.0);
        }

    // --- 3. Sample diagonal corners to determine edge direction ---
    float lumaUL = luma(SampleTonemapped(uv + float2(-rcpFrame.x,  rcpFrame.y), exposure, gamma));
    float lumaUR = luma(SampleTonemapped(uv + float2( rcpFrame.x,  rcpFrame.y), exposure, gamma));
    float lumaDL = luma(SampleTonemapped(uv + float2(-rcpFrame.x, -rcpFrame.y), exposure, gamma));
    float lumaDR = luma(SampleTonemapped(uv + float2( rcpFrame.x, -rcpFrame.y), exposure, gamma));

    // Combine neighbours into weighted gradient estimates per axis
    float edgeHorizontal = abs(-2.0*lumaLeft  + lumaUL + lumaDL)
                         + abs(-2.0*lumaCenter + lumaUp + lumaDown) * 2.0
                         + abs(-2.0*lumaRight  + lumaUR + lumaDR);
    float edgeVertical   = abs(-2.0*lumaUp     + lumaUL + lumaUR)
                         + abs(-2.0*lumaCenter + lumaLeft + lumaRight) * 2.0
                         + abs(-2.0*lumaDown   + lumaDL  + lumaDR);

    bool isHorizontal = (edgeHorizontal >= edgeVertical);

    // --- 4. Step perpendicular to the edge to find its two endpoints ---
    float luma1 = isHorizontal ? lumaDown : lumaLeft;
    float luma2 = isHorizontal ? lumaUp   : lumaRight;
    float gradient1 = abs(luma1 - lumaCenter);
    float gradient2 = abs(luma2 - lumaCenter);

    // The steeper gradient side is the edge — step toward it
    bool  is1Steepest  = (gradient1 >= gradient2);
    float gradientScaled = 0.25 * max(gradient1, gradient2);
    float stepLength  = isHorizontal ? rcpFrame.y : rcpFrame.x;
    float lumaLocalAvg;
    if (!is1Steepest) stepLength = -stepLength;
    lumaLocalAvg = 0.5 * (is1Steepest ? (luma1 + lumaCenter) : (luma2 + lumaCenter));

    // Shift UV to the center of the edge
    float2 currentUV = uv;
    if (isHorizontal) currentUV.y += stepLength * 0.5;
    else              currentUV.x += stepLength * 0.5;

    // Walk along the edge in both directions to find its length
    float2 offset    = isHorizontal ? float2(rcpFrame.x, 0.0) : float2(0.0, rcpFrame.y);
    float2 uv1 = currentUV - offset;
    float2 uv2 = currentUV + offset;

    float lumaEnd1 = luma(SampleTonemapped(uv1, exposure, gamma)) - lumaLocalAvg;
    float lumaEnd2 = luma(SampleTonemapped(uv2, exposure, gamma)) - lumaLocalAvg;

    bool reached1 = (abs(lumaEnd1) >= gradientScaled);
    bool reached2 = (abs(lumaEnd2) >= gradientScaled);

    // Continue walking up to 12 steps if edge not yet found
    static const int FXAA_QUALITY_STEPS = 12;
    static const float quality[12] = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0 };

    [unroll]
    for (int i = 0; i < FXAA_QUALITY_STEPS && !(reached1 && reached2); ++i)
    {
        if (!reached1) { uv1 -= offset * quality[i]; lumaEnd1 = luma(SampleTonemapped(uv1, exposure, gamma)) - lumaLocalAvg; }
        if (!reached2) { uv2 += offset * quality[i]; lumaEnd2 = luma(SampleTonemapped(uv2, exposure, gamma)) - lumaLocalAvg; }
        reached1 = (abs(lumaEnd1) >= gradientScaled);
        reached2 = (abs(lumaEnd2) >= gradientScaled);
    }

    // --- 5. Blend proportional to distance to nearest endpoint ---
    float dist1 = isHorizontal ? (uv.x - uv1.x) : (uv.y - uv1.y);
    float dist2 = isHorizontal ? (uv2.x - uv.x) : (uv2.y - uv.y);
    float distFinal = min(dist1, dist2) / (dist1 + dist2);

    // Only blend if luma delta at the endpoint confirms we're on the correct
    // side of the edge — prevents blurring in the wrong direction
    bool lumaCenterSmaller = lumaCenter < lumaLocalAvg;
    bool correctVariation  = ((dist1 < dist2 ? lumaEnd1 : lumaEnd2) < 0.0) != lumaCenterSmaller;
    float edgeOffset = correctVariation ? (-distFinal + 0.5) * stepLength : 0.0;

    // Sub-pixel blend — reduces single-pixel shimmer
    float subPixelOffset = 0.0;
    {
        float lumaAvg = (1.0 / 12.0) * (2.0 * (lumaUp + lumaDown + lumaLeft + lumaRight)
                      + lumaUL + lumaUR + lumaDL + lumaDR);
        subPixelOffset = clamp(abs(lumaAvg - lumaCenter) / lumaRange, 0.0, 1.0);
        subPixelOffset = (-2.0 * subPixelOffset + 3.0) * subPixelOffset * subPixelOffset;
        subPixelOffset *= subPixelOffset * 0.75; // Blend strength cap
    }

    float subPixelOffsetSigned = subPixelOffset * stepLength;
    float finalOffset = (abs(edgeOffset) >= abs(subPixelOffsetSigned)) ? edgeOffset : subPixelOffsetSigned;
    float2 finalUV = uv;
    if (isHorizontal) finalUV.y += finalOffset;
    else              finalUV.x += finalOffset;

    finalUV = clamp(finalUV, rcpFrame * 0.5, float2(1.0, 1.0) - rcpFrame * 0.5);

    float3 finalColor = SampleTonemapped(finalUV, exposure, gamma);

    return float4(finalColor, 1.0);
}