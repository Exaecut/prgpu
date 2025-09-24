const PI: f32 = 3.1415926;

// Parameters
struct Params {
    time: f32,                  // Time in seconds
    angle: f32,
    xframe: f32,
    inner_spread: f32,
    outer_spread: f32,
    spread_fade: f32,
    blur_type: u32,
    samples: u32,
    origin_x: f32,
    origin_y: f32,
    red_offset: f32,
    green_offset: f32,
    blue_offset: f32,
    blur_alpha: u32,
    uniform_aspect_ratio: u32,
    debug: u32,
    is_premiere: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;

const TYPE_RADIAL: u32 = 0u;
const TYPE_LINEAR: u32 = 1u;

fn cubic(v: f32) -> vec4<f32> {
    let n = vec4<f32>(1.0, 2.0, 3.0, 4.0) - v;
    let s = n * n * n;
    let x = s.x;
    let y = s.y - 4.0 * s.x;
    let z = s.z - 4.0 * s.y + 6.0 * s.x;
    let w = 6.0 - x - y - z;
    return vec4<f32>(x, y, z, w) * (1.0 / 6.0);
}

fn bicubic_sample(tex: texture_2d<f32>, samp: sampler, coord: vec2<f32>) -> vec4<f32> {
    let resolution = vec2<f32>(textureDimensions(tex));
    let invResolution = 1.0 / resolution;

    var coordAdjusted = coord * resolution - 0.5;
    let fxy = fract(coordAdjusted);
    coordAdjusted -= fxy;

    let xcubic = cubic(fxy.x);
    let ycubic = cubic(fxy.y);

    let c = vec4<f32>(
        coordAdjusted.x - 0.5,
        coordAdjusted.x + 1.5,
        coordAdjusted.y - 0.5,
        coordAdjusted.y + 1.5
    );

    let s = vec4<f32>(
        xcubic.x + xcubic.y,
        xcubic.z + xcubic.w,
        ycubic.x + ycubic.y,
        ycubic.z + ycubic.w
    );

    let offset = c + vec4<f32>(
        xcubic.y / s.x,
        xcubic.w / s.y,
        ycubic.y / s.z,
        ycubic.w / s.w
    );

    let sample0 = textureSampleLevel(
        tex,
        samp,
        vec2<f32>(offset.x * invResolution.x, offset.z * invResolution.y),
        0.0
    );
    let sample1 = textureSampleLevel(
        tex,
        samp,
        vec2<f32>(offset.y * invResolution.x, offset.z * invResolution.y),
        0.0
    );
    let sample2 = textureSampleLevel(
        tex,
        samp,
        vec2<f32>(offset.x * invResolution.x, offset.w * invResolution.y),
        0.0
    );
    let sample3 = textureSampleLevel(
        tex,
        samp,
        vec2<f32>(offset.y * invResolution.x, offset.w * invResolution.y),
        0.0
    );

    let sx = s.x / (s.x + s.y);
    let sy = s.z / (s.z + s.w);

    return mix(
        mix(sample3, sample2, sx),
        mix(sample1, sample0, sx),
        sy
    );
}

fn linear_sampling(center: vec2<f32>, origin: vec2<f32>, distance: f32, index: i32, samples: i32, aspect_ratio: vec2<f32>) -> vec2<f32> {
    var step_distance = distance / f32(samples - 1);
    step_distance = step_distance * f32(index);

    var direction: vec2<f32>;
    if (params.uniform_aspect_ratio == 1u) {
        direction = normalize((center - origin) / aspect_ratio);
    } else {
        direction = normalize(center - origin);
    }

    var sample_uv: vec2<f32>;
    if (params.uniform_aspect_ratio == 1u) {
        sample_uv = center + (direction * step_distance) * aspect_ratio;
    } else {
        sample_uv = center + (direction * step_distance);
    }

    return sample_uv;
}

fn arc_sampling(center: vec2<f32>, origin: vec2<f32>, angle: f32, index: i32, samples: i32, aspect_ratio: vec2<f32>) -> vec2<f32> {
    var delta: vec2<f32>;
    if (params.uniform_aspect_ratio == 1u) {
        delta = (center - origin) / aspect_ratio;
    } else {
        delta = center - origin;
    }

    let radius: f32 = length(delta);
    let theta = atan2(delta.y, delta.x);

    let half_angle = angle * 0.5;
    let theta_i = theta + (-half_angle + angle * f32(index) / f32(samples - 1));

    var u: f32;
    var v: f32;
    if (params.uniform_aspect_ratio == 1u) {
        u = origin.x + (radius * cos(theta_i)) * aspect_ratio.x;
        v = origin.y + (radius * sin(theta_i)) * aspect_ratio.y;
    } else {
        u = origin.x + radius * cos(theta_i);
        v = origin.y + radius * sin(theta_i);
    }

    return vec2<f32>(u, v);
}

// Main compute shader
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);
    let origin = vec2<f32>(params.origin_x, params.origin_y);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    let input_size = textureDimensions(input);
    let input_size_f32 = vec2<f32>(input_size);

    let aspect_ratio = input_size_f32.x / input_size_f32.y;
    var aspect_scale: vec2<f32>;
    if (aspect_ratio > 1.0) {
        aspect_scale = vec2<f32>(input_size_f32.y / input_size_f32.x, 1.0);
    } else {
        aspect_scale = vec2<f32>(1.0, input_size_f32.x / input_size_f32.y);
    }

    let texel_size = 1.0 / input_size_f32;
    var uv = (global_id_f32.xy + vec2<f32>(0.5)) / input_size_f32.xy;
    uv -= params.xframe * texel_size;
    
    var color = textureSampleLevel(input, input_sampler, uv, 0.0);

    var total_color = vec4<f32>(0.0);
    let relative_origin = (origin - params.xframe) / input_size_f32.xy;

    for (var i: i32 = 0; i < i32(params.samples); i = i + 1) {
        var sample_uv_red: vec2<f32>;
        var sample_uv_green: vec2<f32>;
        var sample_uv_blue: vec2<f32>;

        if (params.blur_type - 1u == TYPE_RADIAL) {
            sample_uv_red = saturate(arc_sampling(uv, relative_origin, params.angle + params.red_offset, i, i32(params.samples), aspect_scale));
            sample_uv_green = saturate(arc_sampling(uv, relative_origin, params.angle + params.green_offset, i, i32(params.samples), aspect_scale));
            sample_uv_blue = saturate(arc_sampling(uv, relative_origin, params.angle + params.blue_offset, i, i32(params.samples), aspect_scale));
        } else { // TYPE_LINEAR
            sample_uv_red = saturate(linear_sampling(uv, relative_origin, -params.angle + params.red_offset, i, i32(params.samples), aspect_scale));
            sample_uv_green = saturate(linear_sampling(uv, relative_origin, -params.angle + params.green_offset, i, i32(params.samples), aspect_scale));
            sample_uv_blue = saturate(linear_sampling(uv, relative_origin, -params.angle + params.blue_offset, i, i32(params.samples), aspect_scale));
        }

        var sample_color: vec4<f32> = vec4<f32>(
            textureSampleLevel(input, input_sampler, sample_uv_red, 0.0).r,
            textureSampleLevel(input, input_sampler, sample_uv_green, 0.0).g,
            textureSampleLevel(input, input_sampler, sample_uv_blue, 0.0).b,
            textureSampleLevel(input, input_sampler, sample_uv_red, 0.0).a
        );

        total_color += sample_color;
    }

    var final_color: vec4<f32> = total_color / f32(params.samples);
    
    var distance: f32;
    if (params.uniform_aspect_ratio == 1u) {
        let scaled_delta = (uv - relative_origin) / aspect_scale;
        distance = length(scaled_delta);
    } else {
        distance = length(uv - relative_origin);
    }

    let aa_width: f32 = (1.0 / input_size_f32.x) * params.spread_fade;

    let outer_mask: f32 = smoothstep(params.outer_spread - aa_width, params.outer_spread + aa_width, distance);

    var inner_mask: f32;
    if (params.inner_spread <= 0.0) {
        inner_mask = 1.0;
    } else {
        inner_mask = smoothstep(params.inner_spread - aa_width, params.inner_spread + aa_width, distance);
    }

    var spread_mask: f32 = inner_mask * (1.0 - outer_mask);

    final_color = mix(
        color,
        final_color,
        spread_mask
    );

    // final_color = vec4<f32>(spread_mask);

    if params.debug == 1u {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, vec2<f32>(uv.xy), 0.0));
    } else {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(select(color.r, final_color.r, params.blur_alpha == 1u), final_color.gba));
    }
}
