#include "vekl.h"

/// Compute normalized texture coordinates (UV) from thread grid position.
inline float2 tex_coord(uint2 gid, uint2 size_px)
{
	return (float2(gid) + 0.5f) / float2(size_px);
}

/// Convert normalized UV into pixel coordinates.
inline float2 pixel_coord(float2 uv, uint2 size_px)
{
	return uv * float2(size_px) - 0.5f;
}

/// Rotate normalized UV coordinates around center (0.5, 0.5).
/// `aspect` is width / height (to preserve aspect ratio).
inline float2 rotate_uv(float2 uv, float angle, float aspect)
{
    // Shift to center
    float2 centered = uv - 0.5;

    // Correct aspect ratio
    centered.x *= aspect;

    // Rotation matrix
    float c = cos(angle);
    float s = sin(angle);
    float2 rotated = float2(
        c * centered.x - s * centered.y,
        s * centered.x + c * centered.y
    );

    // Undo aspect correction
    rotated.x /= aspect;

    // Shift back
    return rotated + 0.5;
}

/// Scale normalized UV around center (0.5, 0.5).
inline float2 scale_uv(float2 uv, float2 scale)
{
    // Shift to center
    float2 centered = uv - 0.5;

    // Apply scaling
    centered *= scale;

    // Shift back
    return centered + 0.5;
}

/// Overload for uniform scale
inline float2 scale_uv(float2 uv, float scale)
{
    return scale_uv(uv, float2(scale, scale));
}

inline float2 uniform_aspect_ratio(float2 uv, uint2 size_px)
{
    float aspect = float(size_px.x) / float(size_px.y);
    return scale_uv(uv, float2(aspect, 1.0));
}
