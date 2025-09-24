const e: f32 = 2.71828182845904523536;

struct BlurParams {
    is_horizontal: u32,
    radius: f32,
    debug: u32,
};

@group(0) @binding(0) var<uniform> params: BlurParams;
@group(0) @binding(1) var original_texture: texture_2d<f32>;
@group(0) @binding(2) var input_texture: texture_2d<f32>;
@group(0) @binding(3) var all_pass_texture: texture_2d<f32>;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var input_sampler: sampler;
var<push_constant> layer_index: f32;

fn compute_weight(offset: f32, radius: f32) -> f32 {
    let x = f32(offset);
    let variance = radius * radius / 4.0;
    return exp(-(x * x) / (2.0 * variance));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let full_texture_size = vec2<i32>(textureDimensions(all_pass_texture));
    let full_texture_size_f = vec2<f32>(full_texture_size);
    let tex_size = vec2<i32>(textureDimensions(input_texture));
    let tex_size_f = vec2<f32>(tex_size);

    let output_tex_size = vec2<i32>(textureDimensions(output_texture));
    let output_tex_size_f = vec2<f32>(output_tex_size);
    let uv_int = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let uv = (vec2<f32>(uv_int) + 0.5) / tex_size_f;
    let out_uv = (vec2<f32>(uv_int) + 0.5) / output_tex_size_f;

    if (uv_int.x >= output_tex_size.x || uv_int.y >= output_tex_size.y) {
        return;
    }

    let dir = select(
        vec2<f32>(0.0, 1.0 / output_tex_size_f.y),
        vec2<f32>(1.0 / output_tex_size_f.x, 0.0),
        params.is_horizontal == 1u
    );

    var color = vec4<f32>(0.0);
    var sum = 0.0;

    let radius_multiplier = 0.1 * pow(e, 1.151 * layer_index);
    let adjusted_radius = radius_multiplier * params.radius;

    for (var offset = -adjusted_radius; offset <= adjusted_radius; offset = offset + 1) {
        let weight = compute_weight(offset, adjusted_radius);
        if (weight > 0.001) {
            let sample_uv = uv + dir * f32(offset);
            color += textureSampleLevel(input_texture, input_sampler, sample_uv, 0.0) * weight;
            sum += weight;
        }
    }

    var final_color = color / max(sum, 0.0001);

    if (params.debug == 1u) {
        if (params.is_horizontal == 1u) {
            final_color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
        } else {
            final_color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        }
    }

    textureStore(output_texture, uv_int, final_color);
}