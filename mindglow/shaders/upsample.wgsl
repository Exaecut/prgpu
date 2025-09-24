struct Params {
    time: f32,
    debug: u32,
    threshold: f32,
    threshold_smoothness: f32,
    is_premiere: u32,
}

var<push_constant> layer_index: f32;
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var src_lower_texture: texture_2d<f32>;
@group(0) @binding(2) var additive_texture: texture_2d<f32>;
@group(0) @binding(3) var dst_texture: texture_storage_2d<rgba16float, read_write>;
@group(0) @binding(4) var input_sampler: sampler;

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) / 2.0;
}

fn threshold_sample(sample: vec4<f32>, threshold: f32, epsilon: f32) -> vec4<f32> {
    let lum = rgb_to_lightness(sample.gba);
    let factor = smoothstep(threshold - epsilon, threshold + epsilon, lum);
    return vec4<f32>(sample.r, sample.gba * factor);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let input_size = textureDimensions(src_lower_texture);
    let input_size_f32 = vec2<f32>(input_size);

    let output_size = textureDimensions(dst_texture);
    let output_size_f32 = vec2<f32>(output_size);
    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));

    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    let additive_size = textureDimensions(additive_texture);
    let additive_size_f32 = vec2<f32>(additive_size);

    var upsample = vec4<f32>(0.0);

    let texel_size = 1.0 / input_size_f32;
    let otxl_size = 1.0 / output_size_f32;
    // let scale_factor = (user_scale_factor / output_size_f32.x) + 1.0;
    var uv = (global_id_f32.xy + 0.5) / output_size_f32.xy;

    // Scale UV from center by scale factor
    var high_res_uv = uv;
    // high_res_uv = ((uv - 0.5) / user_scale_factor) + 0.5;

    let x = texel_size.x;
    let y = texel_size.y;

    // a - b - c
    // d - e - f
    // g - h - i
    let a = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x - x, uv.y + y), 0.0);
    let b = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x, uv.y + y), 0.0);
    let c = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x + x, uv.y + y), 0.0);

    let d = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x - x, uv.y), 0.0);
    let e = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x, uv.y), 0.0);
    let f = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x + x, uv.y), 0.0);

    let g = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x - x, uv.y - y), 0.0);
    let h = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x, uv.y), 0.0);
    let i = textureSampleLevel(src_lower_texture, input_sampler, vec2<f32>(uv.x + x, uv.y - y), 0.0);

    upsample = e * 4.0;
    upsample += (b + d + f + h) * 2.0;
    upsample += (a + c + g + i);
    upsample *= 1.0 / 16.0;

    var additive_value = textureSampleLevel(additive_texture, input_sampler, high_res_uv, 0.0);
    additive_value = threshold_sample(additive_value, params.threshold, params.threshold_smoothness);

    var final_color = upsample + additive_value;
    final_color = max(final_color, vec4<f32>(0.0, 0.0001, 0.0001, 0.0001));

    // Write the downsampled value to the output texture
    if (params.debug == 1u) {
        let debug_color = textureSampleLevel(src_lower_texture, input_sampler, uv, 0.0);
        textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0, uv, 0.0));
        textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), debug_color);
        textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), additive_value);
        textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0, final_color.gba));
        // textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0));
        // textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0, vec3f(f32(current_mip) / 7.0)));
    } else {
        textureStore(dst_texture, vec2<u32>(global_id.x, global_id.y), vec4<f32>(1.0, final_color.gba));
    }
}