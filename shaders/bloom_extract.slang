struct VSOutput {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D    sceneColor;
[[vk::binding(0, 0)]] SamplerState sceneSampler;

struct BloomExtractPushConstants
{
    float threshold;
    float softKnee;
    float prefilterScale;
    float _pad;
};

[[vk::push_constant]] BloomExtractPushConstants pc;

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

[shader("fragment")]
float4 fragMain(VSOutput input) : SV_TARGET
{
    float3 color = sceneColor.Sample(sceneSampler, input.UV).rgb;

    // Use luminance instead of max channel for more stable extraction.
    float brightness = dot(color, float3(0.2126, 0.7152, 0.0722));

    // Soft-knee thresholding to preserve highlight rolloff.
    float knee = max(1e-5, pc.threshold * pc.softKnee);
    float soft = saturate((brightness - pc.threshold + knee) / (2.0 * knee));
    float contribution = max(brightness - pc.threshold, 0.0) + soft * soft * knee;
    contribution /= max(brightness, 1e-4);

    float3 outColor = color * contribution * pc.prefilterScale;
    return float4(outColor, 1.0);
}
