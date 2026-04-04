#include "vekl.h"

// Convert 2D coordinates to a linear index given the pitch (width) in pixels
inline uint index_of(uint2 coords, uint pitch_px) { return coords.y * pitch_px + coords.x; }

// Clamp 2D coordinates to be within image bounds
inline uint2 clamp_xy(uint2 coords, uint2 size) {
    return uint2(vekl::min(coords.x, size.x - 1),
                 vekl::min(coords.y, size.y - 1));
}

// Read a value from an image at given 2D coordinates, clamping to image bounds
inline float4 image_read(device_ptr(float4) data, uint pitch_px, uint2 size_px, uint2 xy) {
    uint2 clamped_xy = clamp_xy(xy, size_px);
    uint idx = index_of(clamped_xy, pitch_px);
    return data[idx];
}

// Write a value to an image at given 2D coordinates, clamping to image bounds
inline void image_write(device_ptr(float4) data, uint pitch_px, uint2 size_px, uint2 xy, float4 value) {
    uint2 clamped_xy = clamp_xy(xy, size_px);
    uint idx = index_of(clamped_xy, pitch_px);
    data[idx] = value;
}

// Read a linearly interpolated value from an image at given UV coordinates [0,1]
inline float4 image_read_linear(device_ptr(float4) data, uint pitch_px, uint2 size_px, float2 uv) {
    // Convert UV [0,1] to pixel coordinates
    float2 pixel_coords = uv * float2(size_px);
    
    // Get integer and fractional parts
    uint2 xy0 = uint2(vekl::min(uint(pixel_coords.x), size_px.x - 1),
                      vekl::min(uint(pixel_coords.y), size_px.y - 1));
    uint2 xy1 = uint2(vekl::min(xy0.x + 1, size_px.x - 1),
                      vekl::min(xy0.y + 1, size_px.y - 1));
    float2 f = pixel_coords - float2(xy0);
    
    // Bilinear interpolation
    float4 c00 = image_read(data, pitch_px, size_px, xy0);
    float4 c10 = image_read(data, pitch_px, size_px, uint2(xy1.x, xy0.y));
    float4 c01 = image_read(data, pitch_px, size_px, uint2(xy0.x, xy1.y));
    float4 c11 = image_read(data, pitch_px, size_px, xy1);
    
    float4 c0 = c00 * (1.0f - f.x) + c10 * f.x;
    float4 c1 = c01 * (1.0f - f.x) + c11 * f.x;
    return c0 * (1.0f - f.y) + c1 * f.y;
}