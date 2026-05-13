struct VSOutput {
    float4 Position : SV_POSITION;
    float2 UV       : TEXCOORD0;
};

[[vk::binding(0, 0)]] Texture2D    inputColor;
[[vk::binding(0, 0)]] SamplerState inputSampler;

struct BloomBlurPushConstants
{
    float2 direction;
    float blurScale;
    float _pad;
};

[[vk::push_constant]] BloomBlurPushConstants pc;

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
    float2 uv = input.UV;
    float2 d = pc.direction * pc.blurScale;

    float3 color = float3(0.0, 0.0, 0.0);
    color += inputColor.Sample(inputSampler, uv).rgb * 0.227027;
    color += inputColor.Sample(inputSampler, uv + d * 1.384615).rgb * 0.316216;
    color += inputColor.Sample(inputSampler, uv - d * 1.384615).rgb * 0.316216;
    color += inputColor.Sample(inputSampler, uv + d * 3.230769).rgb * 0.070270;
    color += inputColor.Sample(inputSampler, uv - d * 3.230769).rgb * 0.070270;

    return float4(color, 1.0);
}
