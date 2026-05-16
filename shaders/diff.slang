import vekl;

// Blackbody-inspired heatmap: encodes per-pixel max channel error as a
// thermal gradient controlled by a smoothstep.
//
//   black  → dark blue → bright blue → orange → white
//   (cold)                                        (hot)
//
// `smoothA` / `smoothB` let the user pick where the sigmoid ramp lives:
// below smoothA is black, above smoothB is white, between is the gradient.
//
// Tolerance fields still gate the pass/fail count used by metrics.

struct DiffParams
{
    float tolR;
    float tolG;
    float tolB;
    float tolA;
    float smoothA;
    float smoothB;
    uint _pad0;
    uint _pad1;
};

float3 error_to_color(float maxErr, float a, float b)
{
    float t = smoothstep(a, b, maxErr);

    // Blackbody-inspired piecewise ramp:
    //   0.00 → 0.25 : black → dark blue
    //   0.25 → 0.50 : dark blue → bright blue
    //   0.50 → 0.75 : bright blue → orange
    //   0.75 → 1.00 : orange → white
    float3 c;
    if (t < 0.25)
    {
        float s = t * 4.0;
        c = lerp(float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.4), s);
    }
    else if (t < 0.5)
    {
        float s = (t - 0.25) * 4.0;
        c = lerp(float3(0.0, 0.0, 0.4), float3(0.1, 0.2, 1.0), s);
    }
    else if (t < 0.75)
    {
        float s = (t - 0.5) * 4.0;
        c = lerp(float3(0.1, 0.2, 1.0), float3(1.0, 0.5, 0.0), s);
    }
    else
    {
        float s = (t - 0.75) * 4.0;
        c = lerp(float3(1.0, 0.5, 0.0), float3(1.0, 1.0, 1.0), s);
    }
    return c;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void diff(
    uint3 threadId : SV_DispatchThreadID,
    StructuredBuffer<uint> outgoing,
    StructuredBuffer<uint> incoming,
    RWStructuredBuffer<uint> dst,
    ConstantBuffer<FrameParams> frame,
    ConstantBuffer<DiffParams> params)
{
    TextureView rendered  = TextureView(outgoing, frame.outDesc);
    TextureView reference = TextureView(incoming, frame.inDesc);
    RWTextureView output  = RWTextureView(dst, frame.dstDesc);

    uint2 size = output.Size(0u);
    if (threadId.x >= size.x || threadId.y >= size.y)
        return;

    float4 a = rendered.Load(threadId.xy, 0u);
    float4 b = reference.Load(threadId.xy, 0u);

    float4 diff = abs(a - b);
    float maxErr = max(max(diff.r, diff.g), max(diff.b, diff.a));

    output.Store(threadId.xy, float4(error_to_color(maxErr, params.smoothA, params.smoothB), 1.0));
}
