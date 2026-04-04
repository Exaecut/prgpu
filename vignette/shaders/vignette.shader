#include "common.h"
#include "types.h"

struct VignetteParams {
    float strength;
    float softness;
};

kernel void vignette(
    param_dev_ro(pixel_format, outgoing, 0),
    param_dev_ro(pixel_format, incoming, 1),
    param_dev_wo(pixel_format, dest, 2),
    param_dev_cbuf(TransitionParams, tp, 3),
    param_dev_cbuf(VignetteParams, up, 4)
) {
    uint2 gid = dispatch_id();

    if (gid.x >= tp.width || gid.y >= tp.height)
        return;

    uint idx = gid.y * tp.out_pitch + gid.x;
    float4 color = float4(outgoing[idx].x, outgoing[idx].y,
                           outgoing[idx].z, outgoing[idx].w);

    float2 uv = float2((float)gid.x / (float)tp.width,
                        (float)gid.y / (float)tp.height);
    float2 center = float2(0.5f, 0.5f);
    float2 delta = uv - center;
    float dist = sqrtf(delta.x * delta.x + delta.y * delta.y);

    float falloff = clamp(1.0f - dist * up.strength, 0.0f, 1.0f);
    falloff = mix(1.0f, falloff, up.softness);

    color = float4(color.x * falloff, color.y * falloff,
                   color.z * falloff, color.w);

    dest[idx] = pixel_format(color.x, color.y, color.z, color.w);
}
