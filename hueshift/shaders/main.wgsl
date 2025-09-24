// Parameters
struct Params {
    time: f32,                  // Time in seconds
    shift: f32,                 // Shift in degrees
    debug: u32,
    is_premiere: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;

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

const PI: f32 = 3.141592653589793;

fn applyHue(color: vec3<f32>, shift: f32) -> vec4<f32> {
    let base: vec3<f32> = vec3<f32>(0.57735026);
    let p: vec3<f32>    = base * dot(base, color);
    let u: vec3<f32>    = color - p;
    let v: vec3<f32>    = cross(base, u);
    
    let angle: f32      = (shift / 360.0) * 6.2832;
    let c: f32          = cos(angle);
    let s: f32          = sin(angle);
    
    let rotated: vec3<f32> = u * c + v * s + p;
    return vec4<f32>(rotated, 1.0);
}

// Main compute shader
@compute @workgroup_size(16, 16, 1)
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

    var out_color = vec4<f32>(0.0);
    out_color = textureSampleLevel(input, input_sampler, uv, 0.0);

    let raw_color: vec3<f32> = out_color.gba;
    let hue_shifted_color = applyHue(raw_color, params.shift);

    out_color = vec4<f32>(out_color.r, hue_shifted_color.rgb);

    if (params.debug == 1) {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, uv.x, uv.y, 0.0));
    } else {
        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}