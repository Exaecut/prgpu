const PI: f32 = 3.1415926;

const LAYER_FLAG_COMPRESSION: u32 = 1u << 1;

fn layerFlagEnabled(layer: u32, flag: u32) -> bool {
    return (layer & flag) != 0;
}

struct Params {
    time: f32,
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

const blur_amount: vec2<f32> = vec2<f32>(3.5, 0.5);

const rgb2yiq: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(0.299, 0.587, 0.114),
    vec3<f32>(0.596, -0.274, -0.322),
    vec3<f32>(0.211, -0.523, 0.312)
);

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

fn downsample_video(in_uv: vec2<f32>, in_pixel_size: vec2<f32>, in_samples: vec2<i32>) -> vec3<f32> {
    var uv_start: vec2<f32> = in_uv - (in_pixel_size * vec2<f32>(0.5));
    var uv_end: vec2<f32> = in_uv + in_pixel_size;

    var out_color: vec3<f32> = vec3<f32>(0.0);
    for (var i: i32 = 0; i < in_samples.x; i = i + 1) {
        var u: f32 = mix(uv_start.x, uv_end.x, f32(i) / f32(in_samples.x));
        for (var j: i32 = 0; j < in_samples.y; j = j + 1) {
            var v: f32 = mix(uv_start.y, uv_end.y, f32(j) / f32(in_samples.y));
            out_color += textureSampleLevel(input, input_sampler, vec2<f32>(u, v), 0.0).gba;
        }
    }

    out_color /= f32(in_samples.x * in_samples.y);
    return out_color * rgb2yiq;
}

fn pre_downsample(in_coord: vec2<f32>, in_downsampled_res: vec2<f32>) -> vec3<f32> {
    if (in_coord.x > in_downsampled_res.x || in_coord.y > in_downsampled_res.y) {
        return vec3<f32>(0.0);
    }

    var uv: vec2<f32> = in_coord / in_downsampled_res;
    var pixel_size: vec2<f32> = vec2<f32>(1.0) / in_downsampled_res;
    var samples: vec2<i32> = vec2<i32>(8, 3);

    pixel_size *= 1.0 + blur_amount;

    return downsample_video(uv, pixel_size, samples);
}

const yiq2rgb: mat3x3<f32> = mat3x3<f32>(
    vec3<f32>(1.0, 0.956, 0.621),
    vec3<f32>(1.0, -0.272, -0.647),
    vec3<f32>(1.0, -1.106, 1.703)
);

fn uncompress(in_color: vec4<f32>, resolution: vec2<f32>, uv: vec2<f32>) -> vec4<f32> {
    let resLuminance = resolution;
    let resChroma = resolution;

    let uvLuminance = uv * (resLuminance / resolution);
    let uvChroma = uv * (resChroma / resolution);

    let luminance = in_color.y;
    let chroma = in_color.zw;

    return vec4<f32>(1.0, vec3<f32>(luminance, chroma) * yiq2rgb);
}

fn compress(in_coord: vec2<f32>, resolution: vec2<f32>) -> vec4<f32> {
    let resLuminance = resolution;
    let resChroma = resolution;

    let luminance: f32 = pre_downsample(in_coord, resLuminance).r;
    let chroma: vec2<f32> = pre_downsample(in_coord, resChroma).gb;

    return vec4<f32>(1.0, luminance, chroma);
}

fn bleach_yiq_color(color: vec4<f32>) -> vec4<f32> {
    let r = color.r;
    let g = color.g;
    let b = color.b;
    
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let i = 0.596 * r - 0.275 * g - 0.321 * b;
    let q = 0.212 * r - 0.523 * g + 0.311 * b;
    
    let bleached_y = y * 1.1;
    let bleached_i = i * 0.6;
    let bleached_q = q * 0.6;
    
    let r_out = bleached_y + 0.956 * bleached_i + 0.621 * bleached_q;
    let g_out = bleached_y - 0.272 * bleached_i - 0.647 * bleached_q;
    let b_out = bleached_y - 1.106 * bleached_i + 1.703 * bleached_q;
    
    return vec4<f32>(
        clamp(r_out, 0.0, 1.0),
        clamp(g_out, 0.0, 1.0),
        clamp(b_out, 0.0, 1.0),
        color.a
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_size = textureDimensions(input);
    let input_size_f32 = vec2<f32>(input_size);

    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    var uv = (global_id_f32.xy + vec2<f32>(0.5)) / output_size_f32.xy;
    var out_color: vec4<f32> = vec4<f32>(0.0);

    if (layerFlagEnabled(params.enabled_layers, LAYER_FLAG_COMPRESSION)) {
        out_color = compress(global_id_f32.xy, output_size_f32.xy);
    } else {
        out_color = textureSampleLevel(input, input_sampler, uv, 0.0);
    }
    
    if layerFlagEnabled(params.enabled_layers, LAYER_FLAG_COMPRESSION) {
        out_color = uncompress(out_color, input_size_f32.xy, uv);
        out_color = bleach_yiq_color(out_color);
    }

    textureStore(output, vec2<i32>(global_id.xy), out_color);
}