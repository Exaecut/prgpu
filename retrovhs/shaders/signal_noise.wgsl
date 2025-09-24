const PI: f32 = 3.1415926;
const LAYER_FLAG_SIGNAL_NOISE: u32 = 1u << 2;

fn layerFlagEnabled(layer: u32, flag: u32) -> bool {
    return (layer & flag) != 0;
}

struct Params {
    time: f32,
    tape_noise_lowfreq_glitch: f32,
	tape_noise_highfreq_glitch: f32,
	tape_noise_horizontal_offset: f32,
	tape_noise_vertical_offset: f32,
	crease_phase_frequency: f32,
	crease_speed: f32,
	crease_height: f32,
	crease_depth: f32,
	crease_intensity: f32,
	crease_noise_frequency: f32,
	crease_stability: f32,
	crease_minimum: f32,
	extremis_noise_height_proportion: f32,
	side_leak_intensity: f32,
	bloom_exposure: f32,
    enabled_layers: u32,
    is_premiere: u32,
}

@group(0) @binding(0)
var input: texture_2d<f32>;
@group(0) @binding(1)
var input_sampler: sampler;
@group(0) @binding(2)
var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3)
var<uniform> params: Params;

fn cubic(v: f32) -> vec4<f32> {
    let n = vec4<f32>(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    let x = s.x;
    let y = s.y - 4.0 * s.x;
    let z = s.z - 4.0 * s.y + 6.0 * s.x;
    let w = 6.0 - x - y - z;
    return vec4<f32>(x, y, z, w) * (1.0 / 6.0);
}

fn bicubicSample(tex: texture_2d<f32>, samp: sampler, coord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(textureDimensions(tex));
    let invResolution = 1.0 / resolution;

    var coordAdjusted = coord * resolution - 0.5;
    let fxy = fract(coordAdjusted);
    coordAdjusted -= fxy;

    let xcubic = cubic(fxy.x);
    let ycubic = cubic(fxy.y);

    let c = vec4<f32>(coordAdjusted.x - 0.5, coordAdjusted.x + 1.5, coordAdjusted.y - 0.5, coordAdjusted.y + 1.5);

    let s = vec4<f32>(xcubic.x + xcubic.y, xcubic.z + xcubic.w, ycubic.x + ycubic.y, ycubic.z + ycubic.w);

    let offset = c + vec4<f32>(xcubic.y / s.x, xcubic.w / s.y, ycubic.y / s.z, ycubic.w / s.w);

    let sample0 = textureSampleLevel(tex, samp, vec2<f32>(offset.x * invResolution.x, offset.z * invResolution.y), 0.0);
    let sample1 = textureSampleLevel(tex, samp, vec2<f32>(offset.y * invResolution.x, offset.z * invResolution.y), 0.0);
    let sample2 = textureSampleLevel(tex, samp, vec2<f32>(offset.x * invResolution.x, offset.w * invResolution.y), 0.0);
    let sample3 = textureSampleLevel(tex, samp, vec2<f32>(offset.y * invResolution.x, offset.w * invResolution.y), 0.0);

    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);

    return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
}

fn fallbackTextureSampleLevel(tex: texture_2d<f32>, uv: vec2<f32>, fallback_color: vec3<f32>) -> vec3<f32> {
    var color: vec3<f32> = vec3<f32>(0.0);
    let resolution: vec2<u32> = textureDimensions(tex);
    let resolution_f32: vec2<f32> = vec2<f32>(f32(resolution.x), f32(resolution.y));

    color = bicubicSample(tex, input_sampler, uv).gba;

    if (0.5 < abs(uv.x - 0.5)) {
        color = fallback_color;
    }

    return color;
}

const HASH_SEEDS: array<f32, 3> = array<f32, 3>(127.1, 311.7, 43758.5453);
fn hash(x: vec2<f32>) -> f32 {
    let h = fract(sin(dot(x, vec2<f32>(HASH_SEEDS[0], HASH_SEEDS[1]))) * HASH_SEEDS[2]);
    return h;
}

fn interpHash(v: vec2<f32>, r: vec2<f32>) -> f32 {
    let h00 = hash(floor(v * r + vec2<f32>(0.0, 0.0)) / r);
    let h10 = hash(floor(v * r + vec2<f32>(1.0, 0.0)) / r);
    let h01 = hash(floor(v * r + vec2<f32>(0.0, 1.0)) / r);
    let h11 = hash(floor(v * r + vec2<f32>(1.0, 1.0)) / r);
    let ip = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), fract(v * r));
    return (h00 * (1.0 - ip.x) + h10 * ip.x) * (1.0 - ip.y) + (h01 * (1.0 - ip.x) + h11 * ip.x) * ip.y;
}

fn noise(v: vec2<f32>) -> f32 {
    var sum: f32 = 0.0;
    for (var i: i32 = 1; i < 9; i = i + 1) {
        let fi: f32 = f32(i);
        sum += interpHash(v + vec2<f32>(fi, fi), vec2<f32>(2.0 * pow(2.0, fi))) / pow(2.0, fi);
    }
    return sum;
}

fn tapeCrease(in_uv: vec2<f32>) -> vec2<f32> {
    let phase: f32 = clamp((sin(in_uv.y * params.crease_phase_frequency - params.time * PI * params.crease_speed) - (1.0 - params.crease_height)) * noise(vec2<f32>(params.time)), 0.0, params.crease_depth) * params.crease_intensity;
    let noise: f32 = max(noise(vec2<f32>(in_uv.y * params.crease_noise_frequency, params.time * 10.0)) - params.crease_stability, params.crease_minimum);

    return vec2<f32>(phase, noise);
}

fn tapeWave(in_uv: vec2<f32>) -> vec2<f32> {
    var noise_uv: vec2<f32> = in_uv;

    noise_uv.x += (noise(vec2<f32>(noise_uv.y, params.time)) - params.tape_noise_horizontal_offset) * params.tape_noise_lowfreq_glitch;
    noise_uv.x += (noise(vec2<f32>(noise_uv.y * 100.0, params.time * 10.0)) - params.tape_noise_horizontal_offset) * params.tape_noise_highfreq_glitch;

    noise_uv.y += params.tape_noise_vertical_offset - 0.5;

    return noise_uv;
}

fn tapeNoise(in_color: vec4<f32>, in_uv: vec2<f32>) -> vec4<f32> {
    var noise_uv: vec2<f32> = in_uv;
    var final_color: vec3<f32> = vec3<f32>(0.0);
    var bloom_color: vec3<f32> = vec3<f32>(0.0);

    noise_uv = tapeWave(in_uv);

    let crease: vec2<f32> = tapeCrease(in_uv);
    noise_uv.x = noise_uv.x - crease.y * crease.x; // noise_uv.x - crease_noise * crease_phase;

    // Intense noise phasing
    let switch_phase: f32 = smoothstep(params.extremis_noise_height_proportion, 0.0, noise_uv.y);
    noise_uv.y += switch_phase * 0.3;
    noise_uv.x += switch_phase * ((noise(vec2<f32>(noise_uv.y * 100.0, params.time * 10.0)) - 0.5) * 0.2);

    final_color = fallbackTextureSampleLevel(input, noise_uv, vec3<f32>(params.side_leak_intensity));

    final_color *= 1.0 - crease.x;

    final_color = mix(final_color, final_color.yzx, switch_phase);
    bloom_color = final_color;

    // Color leaks
    for (var i: f32 = -4.0; i < 2.5; i = i + 1.0) {
        bloom_color.x += fallbackTextureSampleLevel(input, noise_uv + vec2<f32>(i, 0.0) * 0.0065, vec3<f32>(params.side_leak_intensity)).x * params.bloom_exposure;
        bloom_color.y += fallbackTextureSampleLevel(input, noise_uv + vec2<f32>(i - 2.0, 0.0) * 0.0065, vec3<f32>(params.side_leak_intensity)).y * params.bloom_exposure;
        bloom_color.z += fallbackTextureSampleLevel(input, noise_uv + vec2<f32>(i - 4.0, 0.0) * 0.0065, vec3<f32>(params.side_leak_intensity)).z * params.bloom_exposure;
    }

    final_color += bloom_color;
    final_color *= 0.6;

    final_color *= 1.0 + clamp(noise(vec2<f32>(0.0, noise_uv.y + params.time * 0.2)) * 0.6 - 0.25, 0.0, 0.1);
    return vec4<f32>(1.0, final_color);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    var uv = (global_id_f32.xy + vec2<f32>(0.5)) / output_size_f32.xy;
    var out_color: vec4<f32> = vec4<f32>(0.0);

    if (layerFlagEnabled(params.enabled_layers, LAYER_FLAG_SIGNAL_NOISE)) {
        out_color = tapeNoise(vec4<f32>(0.0), uv);
    } else {
        out_color = textureSampleLevel(input, input_sampler, uv, 0.0);
    }

    textureStore(output, vec2<i32>(global_id.xy), out_color);
}