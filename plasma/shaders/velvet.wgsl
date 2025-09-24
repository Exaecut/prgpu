// Constants and parameter struct
struct Params {
    time:                         f32,
    time_factor:                  f32,
    time_offset:                  f32,
    debug:                        u32,
    is_premiere:                  u32,
    color:                        vec4<f32>,
    height_cos_divisor:           f32,
    height_base_offset:           f32,
    vertical_rot_amplitude_scale: f32,
    vertical_rot_base_offset:     f32,
    horizontal_rot_amplitude_scale: f32,
    horizontal_rot_base_offset:     f32,
    horizontal_rot_angle_scale:   f32,
    ray_steps:                    u32,
    distance_accum_scale:         f32,
    fog_density:                  f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var    input_texture:  texture_2d<f32>;
@group(0) @binding(2) var    output_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var    input_sampler:  sampler;
@group(0) @binding(4) var    debug_texture:  texture_2d<f32>;

// 2×2 rotation matrix
fn rotate2D(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(
        c, -s,
        s,  c
    );
}

// Height‐field function
fn getHeight(position: vec2<f32>) -> f32 {
    let iTime = params.time * params.time_factor + params.time_offset;
    return sin(position.x)
         + sin(position.x + position.y)
         + cos(position.y) / params.height_cos_divisor
         + sin(iTime + position.x)
         + params.height_base_offset;
}

// Signed distance to the terrain
fn distanceToTerrain(point: vec3<f32>) -> f32 {
    return point.y - getHeight(point.xz);
}

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) * 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Bounds check
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    // Pixel coords and resolution
    let fragCoord   = vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5);
    let iResolution = vec2<f32>(f32(dims.x), f32(dims.y));

    // Normalized UV
    let uvCoord = (fragCoord * 2.0 - iResolution) / min(iResolution.x, iResolution.y);

    // Compute time
    let iTime = params.time * params.time_factor + params.time_offset;

    // Initialize ray
    var rayDirection = normalize(vec3<f32>(uvCoord, 1.0));

    // First rotation (YZ plane)
    let angle1 = cos(iTime) / params.vertical_rot_amplitude_scale
               + params.vertical_rot_base_offset;
    let yz_in  = vec2<f32>(rayDirection.y, rayDirection.z);
    let yz_out = yz_in * rotate2D(angle1);
    rayDirection.y = yz_out.x;
    rayDirection.z = yz_out.y;

    // Second rotation (XZ plane)
    let angle2 = (sin(iTime) / params.horizontal_rot_amplitude_scale
                 + params.horizontal_rot_base_offset)
                / params.horizontal_rot_angle_scale;
    let xz_in  = vec2<f32>(rayDirection.x, rayDirection.z);
    let xz_out = xz_in * rotate2D(angle2);
    rayDirection.x = xz_out.x;
    rayDirection.z = xz_out.y;

    // Ray‐march loop
    var travelDistance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.ray_steps; i = i + 1u) {
        let samplePos = vec3<f32>(iTime, 0.0, iTime * 0.5)
                      + rayDirection * travelDistance;
        travelDistance = travelDistance
                       + distanceToTerrain(samplePos)
                         * params.distance_accum_scale;
    }

    // Fog and color
    let fogFactor = 1.0 / (1.0 + travelDistance * travelDistance * params.fog_density);
    let finalColor = vec3<f32>(fogFactor * fogFactor, fogFactor * 0.5, fogFactor);
    let baseColor  = vec4<f32>(1.0, finalColor);
    let outColor   = vec4<f32>(1.0, rgb_to_lightness(baseColor.gba) * params.color.rgb);

    // Write to output (ARGB interpretation: R=alpha, G/B/A=finalColor)
    textureStore(
        output_texture,
        vec2<i32>(i32(global_id.x), i32(global_id.y)),
        outColor
    );
}
