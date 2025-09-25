// Constants
const PI: f32 = 3.141592654;

// Parameters
struct Params {
    time:         f32,  // Time in seconds
    debug:        u32,
    is_premiere:  u32,
    time_factor:  f32,  // Multiplier for animation speed
    time_offset:  f32,  // Added offset to time
    color: vec4<f32>,
    pattern_determinism: f32,
    frequency: f32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input:         texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) * 0.5;
}

fn wave(x: f32) -> f32 {
    let s = sin(x);
    return s * abs(s * params.frequency);
}

fn pattern(coords: vec2<f32>, res: vec2<f32>, time: f32) -> vec4<f32> {
    var uv = (coords - res * 0.5) / vec2<f32>(res.y, res.y) * 3.0;

    var a = atan2(uv.x, uv.y) / 2.0 / PI + 0.5;
    var l = length(sin(uv * params.frequency));
    var w = 0.1 / (params.pattern_determinism + l);

    l *= (1.0 + w * wave(a * 10.0 * PI));
    a += w * wave(l * 1.1 * PI) - time * 0.05;

    l *= (1.0 + w * wave(a * 14.0 * PI));
    a += w * wave(l * 2.6 * PI) + time * 0.075;

    l *= (1.0 + w * wave(a * 12.0 * PI));
    a += w * wave(l * 4.1 * PI) - time * 0.035;

    l *= (1.0 + w * wave(a * 8.0 * PI));
    a += w * wave(l * 5.6 * PI) + time * 0.045;

    let r = a * 14.0 + l - time * 0.2;
    let g = l * 2.0  + a + time * 0.3;
    let b = -a * 8.0 + l * 2.0 + time * 0.25;

    var c = (vec3<f32>(
        sin(r*2.0*PI),
        sin(g*2.0*PI),
        sin(b*2.0*PI)
    ) + 1.0) * 0.5;

    let base1 = vec3<f32>(0.7, 0.9, 1.0);
    let base2 = vec3<f32>(0.2, 0.5, 1.0);
    let t     = sqrt(dot(c, c));
    c = mix(base1, base2, t) * (1.2 - c * 0.4);

    return vec4<f32>(c, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let coords = vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5);
    let res = vec2<f32>(f32(dims.x), f32(dims.y));
    let time = params.time * params.time_factor + params.time_offset;

    let color = pattern(coords, res, time);

    var out_color = vec4<f32>(params.color.a, rgb_to_lightness(color.rgb) * params.color.rgb);

    textureStore(
        output_texture,
        vec2<i32>(i32(global_id.x), i32(global_id.y)),
        out_color
    );
}
