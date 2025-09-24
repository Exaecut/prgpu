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
@group(0) @binding(5) var debug_texture: texture_2d<f32>;

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
    return saturate(dot(color, vec3<f32>(0.8126, 0.7152, 0.8722)));
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

fn maximize_brightness(color: vec3<f32>) -> vec3<f32> {
    let max_comp = max(max(color.r, color.g), color.b);
    if (max_comp == 0.0) {
        return color; // Avoid division by zero; return black
    }
    return color / max_comp;
}

fn rgb_to_hsl(rgb: vec3<f32>) -> vec3<f32> {
    let cmax = max(max(rgb.r, rgb.g), rgb.b);
    let cmin = min(min(rgb.r, rgb.g), rgb.b);
    let delta = cmax - cmin;

    var h: f32 = 0.0;
    var s: f32 = 0.0;
    let l: f32 = (cmax + cmin) * 0.5;

    if delta != 0.0 {
        s = delta / (1.0 - abs(2.0 * l - 1.0));
        if cmax == rgb.r {
            h = 60.0 * ((rgb.g - rgb.b) / delta % 6.0);
        } else if cmax == rgb.g {
            h = 60.0 * ((rgb.b - rgb.r) / delta + 2.0);
        } else if cmax == rgb.b {
            h = 60.0 * ((rgb.r - rgb.g) / delta + 4.0);
        }
    }

    if h < 0.0 {
        h += 360.0;
    }
    return vec3<f32>(h / 360.0, s, l);
}

fn hsl_to_rgb(hsl: vec3<f32>) -> vec3<f32> {
    let h = hsl.x * 360.0;
    let s = hsl.y;
    let l = hsl.z;

    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    let m = l - c * 0.5;

    var rgb: vec3<f32>;
    if h < 60.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h < 120.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h < 180.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h < 240.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h < 300.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    return rgb + vec3<f32>(m, m, m);
}

fn enhance_vibrance(color: vec3<f32>, scale: f32) -> vec3<f32> {
    let hsl = rgb_to_hsl(color);

    // Scale saturation, clamping to [0, 1] to avoid over-saturation artifacts
    let new_saturation = clamp(hsl.y * scale, 0.0, 1.0);

    // Slightly boost lightness but prevent it from reaching 1.0 (white)
    let lightness_scale = 1.0 + (scale - 1.0) * 0.05; // Less aggressive than saturation
    let new_lightness = clamp(hsl.z * lightness_scale, 0.0, 0.90); // Cap to avoid white

    // Keep hue unchanged to preserve the color's identity
    let new_hsl = vec3<f32>(hsl.x, max(new_saturation, 0.5), new_lightness);

    return hsl_to_rgb(new_hsl);
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

    var out_uv = (global_id_f32.xy + 0.5) / output_size_f32.xy;

    let original_color = textureSampleLevel(input, input_sampler, uv, 0.0);
    let bloom_color = chromatic_aberration_sample(out_uv, output_size_f32) * vec4<f32>(1.0, params.tint_color_r, params.tint_color_g, params.tint_color_b);

    var out_color = vec4<f32>(1.0);

    if params.debug == 1u {
        let debug_color = textureSampleLevel(debug_texture, input_sampler, out_uv, 0.0);
        textureStore(output, vec2<i32>(global_id.xy), (debug_color * 1.0) + (vec4<f32>(1.0, vec2<f32>(out_uv.xy), 0.0) * 0.0));
        // textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, vec2<f32>(out_uv.xy), 0.0));
    } else {
        let bloom_rgb = (enhance_vibrance(maximize_brightness(bloom_color.gba), params.bloom_strength));
        // let bloom_rgb = textureSampleLevel(bloom_texture, input_sampler, out_uv, 0.0).gba;

        switch (params.preview_layer - 1u) {
            case 0u: {
                var bloom_tonemapped = mix(
                    vec4<f32>(compute_luminance(bloom_color.gba), bloom_rgb),
                    filmic_tonemap(vec4<f32>(compute_luminance(bloom_color.gba), bloom_rgb)),
                    params.tonemap_intensity
                );

                if (params.flicker == 1u) {
                    let flicker_factor = flicker(params.time, params.flicker_frequency, params.flicker_randomness);
                    bloom_tonemapped *= flicker_factor + params.flicker_bias;
                }

                let USE_SCREEN_BLENDING = 0u;
                if (USE_SCREEN_BLENDING == 1u) {
                    out_color = vec4<f32>(original_color.r + (compute_luminance(bloom_color.gba) * (params.bloom_strength + 0.05)), screen_blend(original_color.gba, bloom_tonemapped.gba));
                } else {
                    out_color = vec4<f32>(original_color.r + (compute_luminance(bloom_color.gba) * 0.5 * (params.bloom_strength + 0.05)), original_color.gba + (bloom_tonemapped.gba * params.bloom_strength));
                }
                break;
            }
            case 1u: {
                // Bloom only
                let bloom_view_color = vec4<f32>(compute_luminance(bloom_color.gba) * (params.bloom_strength + 0.05), bloom_rgb);
                var bloom_tonemapped = mix(
                    bloom_view_color,
                    filmic_tonemap(vec4<f32>(compute_luminance(bloom_color.gba) * (params.bloom_strength + 0.05), bloom_rgb)),
                    params.tonemap_intensity
                );

                out_color = bloom_tonemapped;
                break;
            }
            case 2u : {
                var bloom_tonemapped = mix(
                    bloom_color,
                    filmic_tonemap(vec4<f32>(compute_luminance(bloom_color.gba), bloom_color.gba)),
                    params.tonemap_intensity
                );

                let lum = pow(compute_luminance(bloom_tonemapped.gba) * params.bloom_strength, 1.0);
                out_color = vec4<f32>(1.0, lum, lum, lum);
                break;
            }
            default: {
                out_color = vec4<f32>(0.0); // Default case
            }
        }

        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}