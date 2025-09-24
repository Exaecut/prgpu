// Constants
const PI: f32 = 3.14159265359;

// Parameters
struct Params {
    time: f32,                  // Time in seconds
    debug: u32,
    tint: vec4<f32>,
    darken_strength: f32,
    blur_strength: f32,
    anchor: vec2<f32>,
    scale: vec2<f32>,
    noise: f32,
    noise_timeoffset: f32,
    noise_size: f32,
    darken_min: f32,
    darken_max: f32,
    blur_quality: f32,
    blur_radius: f32,
    blur_inner: f32,
    blur_outer: f32,
    is_premiere: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

fn permute(x: f32) -> f32 {
    return fract(sin(x) * 43758.5453);
}

fn gradient(p: vec2<i32>) -> vec2<f32> {
    var x = p.x * 73856093 ^ p.y * 19349663;
    x = (x ^ (x >> 13)) * 1274126177;
    x = x ^ (x >> 16);
    let angle = f32(x & 0xFFFF) / 65535.0 * 6.2831853;
    return vec2<f32>(cos(angle), sin(angle));
}

fn fade(t: vec2<f32>) -> vec2<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn gradNoise2(p: vec2<f32>) -> f32 {
    let i = vec2<i32>(floor(p));
    let f = fract(p);

    let g00 = gradient(i + vec2<i32>(0, 0));
    let g10 = gradient(i + vec2<i32>(1, 0));
    let g01 = gradient(i + vec2<i32>(0, 1));
    let g11 = gradient(i + vec2<i32>(1, 1));

    let d00 = f - vec2<f32>(0.0, 0.0);
    let d10 = f - vec2<f32>(1.0, 0.0);
    let d01 = f - vec2<f32>(0.0, 1.0);
    let d11 = f - vec2<f32>(1.0, 1.0);

    let v00 = dot(g00, d00);
    let v10 = dot(g10, d10);
    let v01 = dot(g01, d01);
    let v11 = dot(g11, d11);

    let u = fade(f);

    let nx0 = mix(v00, v10, u.x);
    let nx1 = mix(v01, v11, u.x);
    let n = mix(nx0, nx1, u.y);

    return 0.5 * n;
}

fn noise(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        gradNoise2(p),
        gradNoise2(p + vec2<f32>(5.2, 1.3)) // offset to decorrelate
    );
}

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) / 2.0;
}

fn radial_mask(uv: vec2<f32>, center: vec2<f32>, min: f32, max: f32) -> f32 {
    let norm_mask = length(uv - center) / (sqrt(2.0) / 2.0);
    return 1.0 - smoothstep(min, max, norm_mask);
}

fn distance_mask(uv: vec2<f32>, center: vec2<f32>, min: f32, max: f32) -> f32 {
    let norm_mask = length(uv - center) / (sqrt(2.0) / 2.0);
    return smoothstep(min, max, norm_mask);
}

fn gain_mask(uv: vec2<f32>, center: vec2<f32>, min: f32, max: f32) -> f32 {
    let norm_mask = length(uv - center) / (sqrt(2.0) / 2.0);
    return mix(min, max, norm_mask);
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

    var uv = (global_id_f32.xy + 0.5) / input_size_f32.xy;
    var out_uv = (global_id_f32.xy + 0.5) / output_size_f32.xy;

    var original_color = textureSampleLevel(input, input_sampler, uv, 0.0);
    var out_color = vec4<f32>(1.0);

    let darken_factor = params.darken_strength / 100.0;
    let blur_factor = params.blur_strength / 100.0;

    var anchor_normalized = params.anchor / input_size_f32.xy;
    anchor_normalized -= 0.5;
    anchor_normalized /= max(params.scale, vec2f(0.00001));
    anchor_normalized += 0.5;

    var uv_anchored = uv;
    uv_anchored -= 0.5;
    uv_anchored /= max(params.scale, vec2f(0.00001));
    uv_anchored += 0.5;

    let noise = noise((uv + vec2f(sin(params.noise_timeoffset), cos(params.noise_timeoffset))) * input_size_f32.x / params.noise_size) * params.noise / 100.0;
    let vignette_mask = distance_mask(uv_anchored + noise, anchor_normalized , params.darken_min, params.darken_max);
    let blur_mask = distance_mask(uv_anchored + noise, anchor_normalized , params.blur_inner, params.blur_outer);

    let DIRECTIONS = 8.0 * params.blur_quality;
    let QUALITY = 2.0 * params.blur_quality;
    let SIZE = mix(0.0, 1.0, blur_mask * blur_factor * 100.0);
    let RADIUS = (SIZE / input_size_f32.xy) * params.blur_radius;


    if (blur_mask > 0.001) {
        for (var d: f32 = 1.0 / QUALITY; d < PI * 2.0; d += PI / DIRECTIONS) {
            for (var i = 1.0 / QUALITY; i < 1.001; i += 1.0 / QUALITY) {
                original_color += textureSampleLevel(input, input_sampler, uv + vec2<f32>(cos(d), sin(d)) * i * RADIUS, 0.0);
            }
        }

        let WEIGHT =  (QUALITY * DIRECTIONS) * 2.0;

        original_color.g /= WEIGHT;
        original_color.b /= WEIGHT;
        original_color.a /= WEIGHT;
    }

    out_color = mix(original_color, original_color * vec4f(1.0, vec3f(params.tint.rgb)), vignette_mask * darken_factor * params.tint.a);

    if params.debug == 1u {
        let debug_color = textureSampleLevel(input, input_sampler, uv, 0.0);

        textureStore(output, vec2<i32>(global_id.xy), vec4f(1.0, vec3f(SIZE, vignette_mask * darken_factor, 1.0)));
        textureStore(output, vec2<i32>(global_id.xy), vec4f(1.0, vec3f(noise.x, noise.y, 0.0)));
    } else {
        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}