// Constants
const PI: f32 = 3.14159265359;

// Parameters
struct Params {
    time: f32,                  // Time in seconds
    debug: u32,
    is_premiere: u32,
    preview_layer: u32,
    threshold: f32,
    threshold_smoothness: f32,
    key_color: vec4<f32>,
    key_color_sensitivity: f32,
    exposure: f32,
    decay: f32,
    length: f32, // 0%..100%
    length_multiplier: f32,
    center: vec2<f32>,
    samples: f32,
    tint_color: vec4<f32>,
    blend_mode: u32,
    blur: f32,
    clip_bounds: u32, // 0: no clip, 1: clip
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

fn rand(time: f32) -> f32 {
    return fract(sin(time * 12.9898) * 43758.5453);
}

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) / 2.0;
}

fn key_color_mask(color: vec4<f32>, key_color: vec4<f32>) -> f32 {
    // Compute Euclidean distance between the two colors
    let threshold = params.key_color_sensitivity; // Threshold for color matching
    let diff = distance(color.rgb, key_color.rgb);

    // Normalize the distance to [0,1] based on the threshold
    // Colors closer than threshold will be close to 1.0 after smoothstep
    let normalized = 1.0 - smoothstep(0.0, threshold, diff);

    return normalized;
}

fn threshold_mask(color: vec4<f32>, threshold: f32, key_color: vec4<f32>) -> vec4<f32> {
    let kmask = key_color_mask(color, key_color.argb);
    let lmask = rgb_to_lightness(color.rgb * kmask);
    return smoothstep(threshold - params.threshold_smoothness, threshold + params.threshold_smoothness, lmask) * color;
}

fn apply_blur(uv: vec2<f32>, input_size_f32: vec2<f32>) -> vec4<f32> {
    let raw = textureSampleLevel(input, input_sampler, uv, 0.0);

    if params.blur <= 0.0 {
        return raw;
    }

    let blur_strength = clamp(params.blur / 100.0, 0.0, 1.0);
    let blur_offset = (blur_strength / input_size_f32) * 100.0;

    let s1 = textureSampleLevel(input, input_sampler, uv + vec2<f32>(blur_offset.x, 0.0), 0.0);
    let s2 = textureSampleLevel(input, input_sampler, uv + vec2<f32>(-blur_offset.x, 0.0), 0.0);
    let s3 = textureSampleLevel(input, input_sampler, uv + vec2<f32>(0.0, blur_offset.y), 0.0);
    let s4 = textureSampleLevel(input, input_sampler, uv + vec2<f32>(0.0, -blur_offset.y), 0.0);

    return (raw + s1 + s2 + s3 + s4) / 5.0;
}

fn compute_light_shafts(uv: vec2<f32>, center: vec2<f32>, input_size_f32: vec2<f32>) -> vec4<f32> {
    let base_offset = uv - center;
    let length_uv = (params.length / 100.0) * params.length_multiplier;
    var color_sum = vec4<f32>(0.0);
    var decay_factor = 1.0;

    for (var i = 0.0; i < params.samples; i += 1.0) {
        let t = i / max(1.0, params.samples - 1.0); // goes from 0.0 to 1.0
        let sample_uv = uv - base_offset * t * length_uv;
        let blurred = apply_blur(sample_uv, input_size_f32);
        let filtered = threshold_mask(blurred, params.threshold / 100.0, params.key_color);
        color_sum += (filtered * vec4<f32>(1.0, params.tint_color.rgb) * decay_factor);
        decay_factor *= params.decay;
    }

    return color_sum * (params.exposure);
}

fn blend_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a + b;
}

fn blend_screen(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return 1.0 - (1.0 - a) * (1.0 - b);
}

fn blend_overlay(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), step(vec4<f32>(0.5), a));
}

fn blend_color_dodge(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return a / max(vec4<f32>(1e-5), 1.0 - b);
}

fn blend_pixel(base: vec4<f32>, shaft: vec4<f32>) -> vec4<f32> {
    switch (params.blend_mode) {
        case 1u:  { return blend_add(base, shaft); }
        case 2u:  { return blend_screen(base, shaft); }
        case 3u:  { return blend_overlay(base, shaft); }
        case 4u:  { return blend_color_dodge(base, shaft); }
        default: { return shaft; }
    }
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

    var uv = (global_id_f32 + 0.5) / input_size_f32;
    if params.clip_bounds == 0u {
        uv -= ((params.length * params.length_multiplier) * 2.0) / input_size_f32;
    }

    var original_color = textureSampleLevel(input, input_sampler, uv, 0.0);
    var out_color = vec4<f32>(0.0);

    let center = (params.center + vec2<f32>(0.5, 0.5)) / output_size_f32;

    if params.debug == 1u {
        let debug_color = textureSampleLevel(input, input_sampler, uv, 0.0);
        textureStore(output, vec2<i32>(global_id.xy), (debug_color * 1.0) + (vec4<f32>(1.0, vec2<f32>(uv.xy), 0.0) * 1.0));
    } else {
        switch (params.preview_layer - 1u) {
            case 0u: { // Final output
                let shaft_color = compute_light_shafts(uv, center, input_size_f32);
                out_color = blend_pixel(original_color, shaft_color);
                break;
            }
            case 1u: { // Key color mask
                out_color = vec4f(key_color_mask(original_color, params.key_color.argb));
                break;
            }
            case 2u: { // Threshold mask
                out_color = vec4f(threshold_mask(original_color, params.threshold / 100.0, params.key_color));
                break;
            }
            case 3u: { // Light shafts only
                out_color = compute_light_shafts(uv, center, input_size_f32);
                break;
            }
            default: {
                out_color = vec4<f32>(0.0); // Default case
            }
        }

        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}