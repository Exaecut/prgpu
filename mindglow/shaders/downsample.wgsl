struct Params {
    time: f32,
    debug: u32,
    strength: f32,
    threshold: f32,
    threshold_smoothness: f32,
    is_premiere: u32,
}

struct DownsampleConstants {
    current_mip: u32,
    user_brightness_factor: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
var<push_constant> downsample_constants: DownsampleConstants;

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) / 2.0;
}

fn threshold_sample(sample: vec4<f32>, threshold: f32, epsilon: f32) -> vec4<f32> {
    let lum = rgb_to_lightness(sample.gba);

    // Optional nonlinear shaping to boost contrast around threshold
    let x = smoothstep(threshold, threshold + epsilon, lum);
    let shaped = x * x * (3.0 - 2.0 * x); // s-curve (polynomial smootherstep)

    let boosted = pow(shaped, 0.8); // gamma lift (optional)

    return vec4<f32>(
        sample.r,
        sample.g * boosted,
        sample.b * boosted,
        sample.a * boosted,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(dst_tex);
    let output_size_f32 = vec2<f32>(output_size);
    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));

    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    let input_size = textureDimensions(src_tex);
    let input_size_f32 = vec2<f32>(input_size);

    let texel_size = 1.0 / input_size_f32;
    var uv = (global_id_f32.xy + 0.5) / output_size_f32.xy;

    let x = texel_size.x;
    let y = texel_size.y;

    var downsample = vec4<f32>(0.0);

    // a - b - c
    // - j - k -
    // d - e - f
    // - l - m -
    // g - h - i

    let a = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - 2*x, uv.y + 2*y), 0.0);
    let b = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - x, uv.y + 2*y), 0.0);
    let c = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x + 2*x, uv.y + 2*y), 0.0);
    
    let d = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - 2*x, uv.y), 0.0);
    let e = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x, uv.y), 0.0);
    let f = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x + 2*x, uv.y), 0.0);

    let g = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - 2*x, uv.y - 2*y), 0.0);
    let h = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x, uv.y - 2*y), 0.0);
    let i = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x + 2*x, uv.y - 2*y), 0.0);

    let j = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - x, uv.y + y), 0.0);
    let k = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x + x, uv.y + y), 0.0);
    let l = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x - x, uv.y - y), 0.0);
    let m = textureSampleLevel(src_tex, input_sampler, vec2<f32>(uv.x + x, uv.y - y), 0.0);

    downsample = e * 0.125;
    downsample += (a + c + g + i) * 0.03125;
    downsample += (b + d + f + h) * 0.0625;
    downsample += (j + k + l + m) * 0.125;

    downsample = threshold_sample(downsample, params.threshold, params.threshold_smoothness);

    downsample = max(downsample, vec4<f32>(0.0));

    
    var intensity_factor = pow(f32(downsample_constants.current_mip + 1), -1.329);
    

    downsample.g *= downsample_constants.user_brightness_factor * intensity_factor;
    downsample.b *= downsample_constants.user_brightness_factor * intensity_factor;
    downsample.a *= downsample_constants.user_brightness_factor * intensity_factor;
    
    // Write the downsampled value to the output texture
    textureStore(dst_tex, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0, downsample.gba));
}