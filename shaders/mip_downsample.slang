import vekl;

// Generates mip level srcLod+1 from srcLod via a 2x2 box average. Source and
// dest share the buffer; mipOffsetBytes keeps the regions disjoint. Bound to
// the prgpu 5-buffer convention (outgoing/incoming/dst/frame/params); only
// `dst` is used here, but the slot layout matches every other effect kernel.

struct MipDownsampleParams
{
    uint srcLod;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void mip_downsample(
    uint3 threadId : SV_DispatchThreadID,
    StructuredBuffer<uint> outgoing,
    StructuredBuffer<uint> incoming,
    RWStructuredBuffer<uint> dst,
    ConstantBuffer<FrameParams> frame,
    ConstantBuffer<MipDownsampleParams> params)
{
    RWTextureView mip = RWTextureView(dst, frame.outDesc);

    uint2 dstSize = mip.Size(params.srcLod + 1u);
    if (threadId.x >= dstSize.x || threadId.y >= dstSize.y)
        return;

    uint2 srcSize = mip.Size(params.srcLod);
    uint2 srcMax  = srcSize - uint2(1u, 1u);
    uint2 base    = threadId.xy * 2u;

    float4 c00 = mip.Load(min(base,                  srcMax), params.srcLod);
    float4 c10 = mip.Load(min(base + uint2(1u, 0u),  srcMax), params.srcLod);
    float4 c01 = mip.Load(min(base + uint2(0u, 1u),  srcMax), params.srcLod);
    float4 c11 = mip.Load(min(base + uint2(1u, 1u),  srcMax), params.srcLod);

    float4 avg = (c00 + c10 + c01 + c11) * 0.25;
    mip.Store(threadId.xy, avg, params.srcLod + 1u);
}
