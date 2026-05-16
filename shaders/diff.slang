import vekl;

// Pixel-difference heatmap kernel. Compares `outgoing` (rendered) against
// `incoming` (reference) and writes a colour-coded diff to `dst`.
//
// Green  → within tolerance
// Red    → above tolerance, intensity = max channel error

struct DiffParams
{
    float tolR;
    float tolG;
    float tolB;
    float tolA;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

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
    float4 tol  = float4(params.tolR, params.tolG, params.tolB, params.tolA);
    bool4  over = diff > tol;

    float maxErr = max(max(diff.r, diff.g), max(diff.b, diff.a));

    if (any(over))
    {
        // Red: intensity = max error, clamped to visible range
        float intensity = saturate(maxErr / max(0.01, max(max(tol.r, tol.g), max(tol.b, tol.a)) * 10.0));
        output.Store(threadId.xy, float4(intensity, 0.0, 0.0, 1.0));
    }
    else
    {
        // Dim green
        output.Store(threadId.xy, float4(0.0, 0.25, 0.0, 1.0));
    }
}
