import vekl;

// Bounded text overlay. Dispatched over the text's bounding box only (not the
// whole frame), so the cost is proportional to the banner, not the canvas.
//
// prgpu 5-buffer convention, repurposed for this private kernel:
//   outgoing = SDF glyph atlas (StructuredBuffer<uint>, 4 bytes/word)
//   incoming = glyph metrics   (StructuredBuffer<uint>, 8 floats/glyph)
//   dst      = destination frame (composited in place)
//   frame    = FrameParams; dstDesc carries the frame pitch/storage/layout
//   params   = TextDrawParams (vekl), byte-identical to Rust TextOverlayParams
//
// `frame.dstDesc.width/height` are the dispatch bbox (config.width/height), but
// the GPU path has flipY=0 so Load/Store address by absolute pixel and the full
// `dstDesc.pitchBytes`, which is the real frame row pitch.

[shader("compute")]
[numthreads(16, 16, 1)]
void text_overlay(
    uint3 threadId : SV_DispatchThreadID,
    StructuredBuffer<uint> outgoing,
    StructuredBuffer<uint> incoming,
    RWStructuredBuffer<uint> dst,
    ConstantBuffer<FrameParams> frame,
    ConstantBuffer<TextDrawParams> params)
{
    if (threadId.x >= params.bboxW || threadId.y >= params.bboxH)
        return;

    uint2 px = uint2(params.bboxX + threadId.x, params.bboxY + threadId.y);
    if (px.x >= params.frameW || px.y >= params.frameH)
        return;

    RWTextureView out = RWTextureView(dst, frame.dstDesc);
    float4 cur = out.Load(px);

    // Background band: straight-alpha "over" across the whole dispatch box.
    float bgA = params.bgColor[3];
    float3 bgRgb = float3(params.bgColor[0], params.bgColor[1], params.bgColor[2]);
    float4 banded = float4(lerp(cur.rgb, bgRgb, bgA), max(cur.a, bgA));

    float4 res = DrawString(banded, float2(px) + 0.5, outgoing, incoming, params);
    out.Store(px, res);
}
