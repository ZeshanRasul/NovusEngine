struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTexCoord;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

struct VertexOutput 
{
	float4 sv_position : SV_POSITION;
    float3 color : COLOR;
    float2 fragTexCoord;
};

[shader ("vertex")]
VertexOutput vertMain(VSInput input)
{
	VertexOutput output;
    output.sv_position = mul(ubo.proj, mul(ubo.view, mul(ubo.model, float4(input.inPosition, 1.0))));
    output.color = input.inColor;
	output.fragTexCoord = input.inTexCoord;
	return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_TARGET
{
    return texture.Sample(inVert.fragTexCoord);
}