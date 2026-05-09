import vekl;

// prgpu built-in: generate mip level `srcLod + 1` from mip level `srcLod`
// using a 2x2 box average. Source and dest live in the SAME buffer; their
// regions never overlap because `TextureDesc.mipOffsetBytes[]` keeps them
// disjoint. The host dispatches this kernel N-1 times to fill lods 1..N.
//
// Each thread writes one destination pixel (at `srcLod + 1`). Reading 4
// source pixels (clamped at the boundary) is cheap enough that a single
// pass stays memory-bound even on laptop GPUs.
//
// Binding convention: this kernel follows the prgpu 5-buffer layout
// (`outgoing`, `incoming`, `dst`, `frame`, `params`) so it dispatches
// through the exact same Metal/CUDA `setBuffer` path as every other effect
// kernel. `outgoing` and `incoming` are unused here — `generate_mips`
// points all three pointers at the same mip buffer and writes land through
// `dst` at the level-indexed offset carried by `frame.outDesc`.

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
