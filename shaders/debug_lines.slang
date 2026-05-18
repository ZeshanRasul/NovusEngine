struct VSIn
{
    float3 pos   : POSITION;
    float3 color : COLOR0;
};

struct VSOut
{
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

[[vk::push_constant]]
cbuffer DebugLinePC
{
    float4x4 uViewProj;
};

[shader("vertex")]
VSOut vertMain(VSIn input)
{
    VSOut o;
    o.position = mul(uViewProj, float4(input.pos, 1.0));
    o.color = input.color;
    return o;
}

[shader("fragment")]
float4 fragMain(VSOut input) : SV_Target0
{
    return float4(input.color, 1.0);
}