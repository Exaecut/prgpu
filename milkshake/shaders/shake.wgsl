struct Params {
    anchor_point_x: f32,
    anchor_point_y: f32,
    time: f32,
    amplitude: f32,
    frequency: f32,
    h_amplitude: f32,
    v_amplitude: f32,
    h_frequency: f32,
    v_frequency: f32,
    phase: f32,
    seed: u32,
    xframe_size_x: f32,
    xframe_size_y: f32,
    repeat_mode_x: u32,
    repeat_mode_y: u32,
    style: u32,
    clip: u32,
    motion_blur: u32,
    motion_blur_time_offset: f32,
    motion_blur_length: f32,
    motion_blur_samples: i32,
    tilt_amplitude: f32,
    tilt_frequency: f32,
    tilt_phase: f32,
    debug: u32,
    is_premiere: u32,
}

const PI: f32 = 3.14;

const REPEAT_NONE: u32 = 1u;
const REPEAT_TILE: u32 = 2u;
const REPEAT_MIRROR: u32 = 3u;

const STYLE_PERLIN: u32 = 1u;
const STYLE_WAVE: u32 = 2u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<u32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8uint, read_write>;

fn fade(t: vec2<f32>) -> vec2<f32> {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn grad(p: vec2<i32>, seed: f32) -> vec2<f32> {
    var x = f32(p.x) * 127.1 + f32(p.y) * 311.7 + seed;
    var s = sin(x) * 43758.5453;
    var angle = fract(s) * 6.2831853;
    return vec2<f32>(cos(angle), sin(angle));
}

fn perlin_noise(pos: vec2<f32>, seed: f32) -> f32 {
    let i = vec2<i32>(i32(floor(pos.x)), i32(floor(pos.y)));
    let f = pos - vec2<f32>(f32(i.x), f32(i.y));

    let u = fade(f);

    let n00 = dot(grad(i + vec2<i32>(0, 0), seed), f - vec2<f32>(0.0, 0.0));
    let n10 = dot(grad(i + vec2<i32>(1, 0), seed), f - vec2<f32>(1.0, 0.0));
    let n01 = dot(grad(i + vec2<i32>(0, 1), seed), f - vec2<f32>(0.0, 1.0));
    let n11 = dot(grad(i + vec2<i32>(1, 1), seed), f - vec2<f32>(1.0, 1.0));

    let nx0 = mix(n00, n10, u.x);
    let nx1 = mix(n01, n11, u.x);

    let nxy = mix(nx0, nx1, u.y);

    return nxy;
}

fn perlin_noise_vec2(pos: vec2<f32>, seed: f32) -> vec2<f32> {
    let n1 = perlin_noise(pos, seed);
    let n2 = perlin_noise(pos + vec2<f32>(100.0, 100.0), f32(pow(seed, 2.0)));
    return vec2<f32>(n1, n2);
}

fn perlin_noise_offset(time: f32, horizontal_amplitude: f32, vertical_amplitude: f32, global_amplitude: f32, horizontal_frequency: f32, vertical_frequency: f32, global_frequency: f32, phase: f32, seed: f32) -> vec2<f32> {
    let pos = vec2<f32>(time * horizontal_frequency * global_frequency + phase, time * vertical_frequency * global_frequency + phase);

    let noise = perlin_noise_vec2(pos, seed);

    let offset_x = noise.x * horizontal_amplitude * global_amplitude;
    let offset_y = noise.y * vertical_amplitude * global_amplitude;

    return vec2<f32>(offset_x, offset_y);
}

fn perlin_noise_velocity(time: f32, time_offset: f32, horizontal_amplitude: f32, vertical_amplitude: f32, global_amplitude: f32, horizontal_frequency: f32, vertical_frequency: f32, global_frequency: f32, phase: f32, seed: f32) -> vec2<f32> {
    let offset_now = perlin_noise_offset(time, horizontal_amplitude, vertical_amplitude, global_amplitude, horizontal_frequency, vertical_frequency, global_frequency, phase, seed);
    let offset_previous = perlin_noise_offset(time - time_offset, horizontal_amplitude, vertical_amplitude, global_amplitude, horizontal_frequency, vertical_frequency, global_frequency, phase, seed);

    let velocity = offset_now - offset_previous;

    return velocity / max(params.amplitude, 1.0);
}

fn compute_motion_blur(coords: vec2<i32>, uv_mask: f32, velocity: vec2<f32>, texture_size: vec2<i32>, texture: texture_2d<u32>, max_blur_samples: i32, min_velocity_threshold: f32) -> vec4<u32> {
    let blur_length = length(velocity);

    if blur_length < min_velocity_threshold {
        return textureLoad(texture, coords, 0);
    }

    let blur_length_clamped = blur_length * params.motion_blur_length;

    let sample_scaling_factor = 1.0;
    let blur_samples_f = blur_length_clamped * sample_scaling_factor;
    let blur_samples = clamp(i32(blur_samples_f), 1, max_blur_samples);

    let velocity_direction = velocity / max(blur_length, 0.000001);

    let step = velocity_direction * blur_length_clamped / f32(max(blur_samples, 1));

    var color_accum: vec4<f32> = vec4<f32>(0.0);
    var total_weight: f32 = 0.0;

    for (var i: i32 = 0; i <= blur_samples; i = i + 1) {
        let t = (f32(i) - f32(blur_samples) * 0.5) / f32(max(blur_samples, 1));

        let offset = t * velocity_direction * blur_length_clamped;

        var sample_coords_f = (vec2f(coords) * uv_mask + offset);
        var sample_coords = vec2<i32>(floor(sample_coords_f + 0.5));

        if params.repeat_mode_x != REPEAT_NONE || params.repeat_mode_y != REPEAT_NONE {
            sample_coords = clamp(sample_coords, vec2<i32>(0), texture_size - vec2<i32>(1));
        } else {
            sample_coords = sample_coords % vec2i(wg_texSize);
        }

        let color_u32 = textureLoad(texture, sample_coords, 0);
        let color_f32 = vec4<f32>(color_u32) / 255.0;

        let weight = exp(-10.0 * t * t); // Adjust spread as needed N control the sharpness

        color_accum = color_accum + color_f32 * weight;
        total_weight = total_weight + weight;
    }

    let blurred_color_f32 = color_accum / total_weight;
    let blurred_color_u32 = vec4<u32>(clamp(blurred_color_f32 * 255.0, vec4<f32>(0.0), vec4<f32>(255.0)));

    return blurred_color_u32;
}

fn rotate_tex_coord(tex_coord: vec2<f32>, anchor: vec2<f32>, angle: f32, tex_size: vec2<f32>) -> vec2<f32> {
    let angleInRad = angle * PI / 180.0;
    let pos = tex_coord * tex_size;

    let translated_pos = pos - anchor;

    let cos_theta = cos(angleInRad);
    let sin_theta = sin(angleInRad);
    let rotated_pos = vec2<f32>(cos_theta * translated_pos.x - sin_theta * translated_pos.y, sin_theta * translated_pos.x + cos_theta * translated_pos.y);

    let final_pos = rotated_pos + anchor;

    return final_pos / tex_size;
}

fn flip_uv_if_odd(uv: vec2<f32>, tile_index: vec2<i32>) -> vec2<f32> {
    let flip = (tile_index % 2) != vec2<i32>(0);
    return select(uv, 1.0 - uv, flip);
}

struct WaveShake {
    time: f32,
    amplitude: f32,
    frequency: f32,
    h_amplitude: f32,
    v_amplitude: f32,
    h_frequency: f32,
    v_frequency: f32,
    phase: f32,
    seed: u32,
}

fn wave_shake(params: WaveShake) -> vec2<f32> {
    let time_freq = params.time * params.frequency;
    let seed_phase = params.phase + f32(params.seed);
    let h_phase = time_freq * params.h_frequency + seed_phase;
    let v_phase = time_freq * params.v_frequency + seed_phase;

    let offset = vec2<f32>(sin(h_phase) * params.h_amplitude, cos(v_phase) * params.v_amplitude);

    return offset * params.amplitude;
}

fn wave_shake_velocity(params: WaveShake) -> vec2<f32> {
    let time_freq = params.time * params.frequency;
    let seed_phase = params.phase + f32(params.seed);
    let h_phase = time_freq * params.h_frequency + seed_phase;
    let v_phase = time_freq * params.v_frequency + seed_phase;

    let dh_phase_dt = params.frequency * params.h_frequency;
    let dv_phase_dt = params.frequency * params.v_frequency;

    let velocity_x = params.amplitude * params.h_amplitude * cos(h_phase) * dh_phase_dt;
    let velocity_y = -params.amplitude * params.v_amplitude * sin(v_phase) * dv_phase_dt;

    return vec2<f32>(velocity_x, velocity_y) / params.amplitude;
}

fn apply_repeat_mode(coord: f32, frac_coord: f32, index: i32, mode: u32) -> f32 {
    switch (mode) {case REPEAT_TILE: {
        return frac_coord;
    }case REPEAT_MIRROR: {
        let flip = (index % 2) != 0;
        return select(frac_coord, 1.0 - frac_coord, flip);
    }case REPEAT_NONE, default: {
        return coord;
    }}
}

var<workgroup> wg_texSize: vec2<u32>;
var<workgroup> wg_texSizeF32: vec2<f32>;
var<workgroup> wg_angle_shake: f32;
var<workgroup> wg_shake_offset: vec2<f32>;
var<workgroup> wg_velocity: vec2<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if local_id.x == 0u && local_id.y == 0u {
        wg_texSize = textureDimensions(input, 0);
        wg_texSizeF32 = vec2<f32>(wg_texSize);

        // Initialize shared variables
        wg_angle_shake = 0.0;
        wg_shake_offset = vec2<f32>(0.0);
        wg_velocity = vec2<f32>(0.0);

        let tilt_phase = params.time * params.tilt_frequency + params.tilt_phase;
        wg_angle_shake = sin(tilt_phase) * params.tilt_amplitude;

        switch (params.style) { 
            case STYLE_PERLIN: {
                wg_shake_offset = perlin_noise_offset(params.time, params.h_amplitude, params.v_amplitude, params.amplitude, params.h_frequency, params.v_frequency, params.frequency, params.phase, f32(params.seed));
                wg_velocity = perlin_noise_velocity(params.time, params.motion_blur_time_offset, params.h_amplitude, params.v_amplitude, params.amplitude, params.h_frequency, params.v_frequency, params.frequency, params.phase, f32(params.seed));
            } case STYLE_WAVE: {
                let shake_params = WaveShake(params.time, params.amplitude, params.frequency, params.h_amplitude, params.v_amplitude, params.h_frequency, params.v_frequency, params.phase, params.seed);
                wg_shake_offset = wave_shake(shake_params);
                wg_velocity = wave_shake_velocity(shake_params);
            }default: {
                // No action needed for no style
            }
        }
    }

    workgroupBarrier();

    let xframe_size = vec2<f32>(params.xframe_size_x, params.xframe_size_y);
    let orig_uv = vec2<f32>(global_id.xy) - xframe_size;
    let shifted_uv = orig_uv + wg_shake_offset;

    let norm_uv = orig_uv / wg_texSizeF32;
    let norm_shifted_uv = shifted_uv / wg_texSizeF32;
    var rotated_uv = rotate_tex_coord(norm_shifted_uv, vec2<f32>(params.anchor_point_x, params.anchor_point_y), wg_angle_shake, wg_texSizeF32);

    let use_norm_uv = params.clip == 1u;

    let uv_mask_max = step(select(rotated_uv, norm_uv, use_norm_uv), vec2<f32>(1.0));
    let uv_mask_min = step(1 - select(rotated_uv, norm_uv, use_norm_uv), vec2f(1.0));
    let uv_mask: f32 = 1.0 - step(min(uv_mask_min.x, uv_mask_max.x) + min(uv_mask_min.y, uv_mask_max.y), 1.0);

    let tile_index = vec2<i32>(floor(rotated_uv));
    let fractional_uv = fract(rotated_uv);

    rotated_uv.x = apply_repeat_mode(rotated_uv.x, fractional_uv.x, tile_index.x, params.repeat_mode_x);
    rotated_uv.y = apply_repeat_mode(rotated_uv.y, fractional_uv.y, tile_index.y, params.repeat_mode_y);

    var texelCoords = vec2<i32>(rotated_uv * wg_texSizeF32);

    texelCoords = vec2<i32>(vec2f(texelCoords % vec2<i32>(wg_texSize)) * uv_mask);

    var color = vec4<u32>(0u);
    if params.debug == 1u {
        var debug_uv = vec2<f32>(fractional_uv);
        color = vec4<u32>(255u, u32(debug_uv.x * 255.0), u32(debug_uv.y * 255.0), 80u);
    } else {
        if params.motion_blur == 1u {
            color = compute_motion_blur(texelCoords, uv_mask, wg_velocity, vec2i(wg_texSizeF32), input, params.motion_blur_samples, 0.01);
        } else {
            color = textureLoad(input, texelCoords, 0);
        }

        color = color * select(u32(0), u32(1), uv_mask > 0.0);
    }

    textureStore(output, global_id.xy, color);
}
