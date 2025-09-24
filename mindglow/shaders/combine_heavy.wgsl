// Constants
const PI: f32 = 3.14159265359;

// Parameters
struct Params {
    time: f32,                  // Time in seconds
    bloom_strength: f32,        // Bloom intensity multiplier
    radius: f32,
    real_radius: f32,
    tint_color_r: f32,
    tint_color_g: f32,
    tint_color_b: f32,
    chromatic_aberration: f32,  // Chromatic aberration
    flicker: u32,               // Flicker toggle
    flicker_frequency: f32,     // Flicker frequency
    flicker_randomness: f32,    // Flicker randomness
    flicker_bias: f32,          // Flicker bias
    tonemap_intensity: f32,     // Tonemapper intensity (0.0 to 1.0)
    debug: u32,
    is_premiere: u32,
    preview_layer: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var bloom_texture: texture_2d<f32>;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(4) var input_sampler: sampler;

fn rand(time: f32) -> f32 {
    return fract(sin(time * 12.9898) * 43758.5453);
}

fn flicker(time: f32, frequency: f32, randomness: f32) -> f32 {
    let basis = (sin(time * frequency * 2.0 * PI) + 1.0) * 0.5;

    let noise = (rand(time) - 0.5) * randomness;
    return saturate(basis + noise);
}

// Screen blend mode for RGB
fn screen_blend(base: vec3<f32>, blend: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(1.0) - (vec3<f32>(1.0) - base) * (vec3<f32>(1.0) - blend);
}

fn compute_luminance(color: vec3<f32>) -> f32 {
    return saturate(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)));
}

fn filmic_tonemap(color: vec4<f32>) -> vec4<f32> {
    // ACES-inspired filmic tonemap parameters
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    // Apply tonemap to RGB channels
    let rgb = color.gba;
    let toned = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e);
    
    // Clamp to avoid negative values and ensure valid range
    let clamped = clamp(toned, vec3<f32>(0.0), vec3<f32>(1.0));
    
    // Return with original alpha
    return vec4<f32>(color.r, clamped);
}

fn chromatic_aberration_sample(uv: vec2<f32>, input_size_f32: vec2<f32>) -> vec4<f32> {
    let offset_strength = params.chromatic_aberration * 100.0;
    let texel_size = 1.0 / input_size_f32;

    let r_offset = vec2<f32>(-1.0, -1.0) * texel_size * offset_strength; // Top-left
    let g_offset = vec2<f32>(1.0, -1.0) * texel_size * offset_strength;  // Top-right
    let b_offset = vec2<f32>(0.0, 1.0) * texel_size * offset_strength;   // Bottom

    let r = textureSampleLevel(bloom_texture, input_sampler, uv + r_offset, 0.0).g;
    let g = textureSampleLevel(bloom_texture, input_sampler, uv + g_offset, 0.0).b;
    let b = textureSampleLevel(bloom_texture, input_sampler, uv + b_offset, 0.0).a;
    let a = textureSampleLevel(bloom_texture, input_sampler, uv, 0.0).r;

    return vec4<f32>(a, r, g, b);
}

fn polar_to_cartesian(angle: f32, distance: f32) -> vec2<f32> {
    return vec2<f32>(
        distance * cos(angle),
        distance * sin(angle)
    );
}

fn maximize_brightness(color: vec3<f32>) -> vec3<f32> {
    let max_comp = max(max(color.r, color.g), color.b);
    if (max_comp == 0.0) {
        return color; // Avoid division by zero; return black
    }
    return color / max_comp;
}

fn sample_3x3_gaussian_blur(input: texture_2d<f32>, input_sampler: sampler, uv: vec2<f32>, tx_sz: vec2<f32>) -> vec3<f32> {
    var color_sum = vec3<f32>(0.0, 0.0, 0.0);
    let offsets = array<vec2<f32>, 9>(
        vec2<f32>(-tx_sz.x, -tx_sz.y), // top-left
        vec2<f32>( 0.0,     -tx_sz.y), // top
        vec2<f32>( tx_sz.x, -tx_sz.y), // top-right
        vec2<f32>(-tx_sz.x,  0.0),     // left
        vec2<f32>( 0.0,      0.0),     // center
        vec2<f32>( tx_sz.x,  0.0),     // right
        vec2<f32>(-tx_sz.x,  tx_sz.y), // bottom-left
        vec2<f32>( 0.0,      tx_sz.y), // bottom
        vec2<f32>( tx_sz.x,  tx_sz.y)  // bottom-right
    );
    let weights = array<f32, 9>(
        1.0,  // top-left
        2.0,  // top
        1.0,  // top-right
        2.0,  // left
        4.0,  // center
        2.0,  // right
        1.0,  // bottom-left
        2.0,  // bottom
        1.0   // bottom-right
    );

    for (var i = 0; i < 9; i = i + 1) {
        let sample_color = textureSampleLevel(input, input_sampler, uv + offsets[i], 0.0).gba;
        color_sum += sample_color * weights[i];
    }
    return color_sum / 16.0; // Normalize by sum of weights (1+2+1+2+4+2+1+2+1 = 16)
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    let input_size = textureDimensions(input);
    let input_size_f32 = vec2<f32>(input_size);

    let texel_size = 1.0 / input_size_f32;

    var uv = (global_id_f32.xy + 0.5) / input_size_f32.xy;
    uv -= (params.real_radius * texel_size).xy;
    let original_color = textureSampleLevel(input, input_sampler, uv, 0.0);

    var out_color = original_color;

    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    let radius = 20.0;
    let linear_step = 2.0;
    let angle_step = 6.0;

    for (var i = linear_step; i > -linear_step; i = i - 1.0) { // STEPS - QUALITY
        let tx_sz = 1.0 / input_size_f32;
        let distance = f32(i) * tx_sz.x * radius;

        var max_lum = 0.0;
        for (var j = 0.0; j < angle_step; j = j + 1.0) {
            let angle = f32(j) * (2.0 * 3.1415926 / f32(angle_step));
            // let color = textureSampleLevel(input, input_sampler, uv + polar_to_cartesian(angle, distance), 0.0).gba;
            let color = sample_3x3_gaussian_blur(input, input_sampler, uv + polar_to_cartesian(angle, distance), tx_sz);
            let max_color = maximize_brightness(color);

            total_color += max_color * (1.0 / angle_step);
        }
    }

    let max_color = total_color;
    // let max_color = maximize_brightness(total_color);

    if params.debug == 1u {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, vec2<f32>(uv.xy), 0.0));
    } else {
        out_color = vec4f(1.0, max_color);
        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}