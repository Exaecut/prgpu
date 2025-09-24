// Constants
const PI: f32 = 3.141592654;

// Parameters
struct Params {
    time:         f32,  // Time in seconds
    time_factor:  f32,
    time_offset:  f32,
    debug:        u32,
    is_premiere:  u32,
    iterations:   u32,  // originally N = 80.0
    warp_base:    f32,  // originally wb = 0.012
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input:         texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

fn rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a * 15.83);
    let s = sin(a * 15.83);

    return vec2<f32>(
        p.x * s + p.y * c,
        p.x * c - p.y * s
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 0) Bounds check
    let size = textureDimensions(output_texture);
    if (global_id.x >= size.x || global_id.y >= size.y) {
        return;
    }

    // 1) Compute coords and res
    let coords   = vec2<f32>(f32(global_id.x), f32(global_id.y)) + vec2<f32>(0.5);
    let res = vec2<f32>(f32(size.x), f32(size.y));

    // 2) Build uv exactly as in GLSL
    var uv = coords / res.x;
    uv = vec2<f32>(0.125, 0.75) + (uv - vec2<f32>(0.125, 0.75)) * 0.015;

    // 3) Time parameter
    let T = (params.time * 0.5 * params.time_factor) + params.time_offset;

    // 4) Initial color vector c
    var c = normalize(vec3<f32>(
        0.75 - 0.25 * sin(length(uv - vec2<f32>(0.1, 0.0)) * 132.0 + T * 3.3),
        0.75 - 0.25 * sin(length(uv - vec2<f32>(0.9, 0.0)) * 136.0 - T * 2.5),
        0.75 - 0.25 * sin(length(uv - vec2<f32>(0.5, 1.0)) * 129.0 + T * 4.1)
    ));

    // 5) Accumulate over iterations
    var c0 = vec3<f32>(0.0);
    var w0 = 0.0;
    let N  = f32(params.iterations);
    let wb = params.warp_base;

    for (var i: f32 = 0.0; i < N; i = i + 1.0) {
        let wt = (i * i / N / N - 0.2) * 0.3;
        let wp = 0.5 + (i + 1.0) * (i + 1.5) * 0.001;

        let zx_in = vec2<f32>(c.z, c.x);
        let zx_out = rot(zx_in, 1.6 + T * 0.65 * wt + (uv.x + 0.7) * 23.0 * wp);
        c.z = zx_out.x;
        c.x = zx_out.y;

        let xy_in = vec2<f32>(c.x, c.y);
        let xy_out = rot(xy_in, c.z * c.x * wb + 1.7 + T * wt + (uv.y + 1.1) * 15.0 * wp);
        c.x = xy_out.x;
        c.y = xy_out.y;

        let yz_in = vec2<f32>(c.y, c.z);
        let yz_out = rot(yz_in,
            c.x * c.y * wb + 2.4 - T * 0.79 * wt +
            (uv.x + uv.y * (fract(i / 2.0) - 0.25) * 4.0) * 17.0 * wp
        );
        c.y = yz_out.x;
        c.z = yz_out.y;

        let zx2_in = vec2<f32>(c.z, c.x);
        let zx2_out = rot(zx2_in, c.y * c.z * wb + 1.6 - T * 0.65 * wt + (uv.x + 0.7) * 23.0 * wp);
        c.z = zx2_out.x;
        c.x = zx2_out.y;

        let xy2_in = vec2<f32>(c.x, c.y);
        let xy2_out = rot(xy2_in, c.z * c.x * wb + 1.7 - T * wt + (uv.y + 1.1) * 15.0 * wp);
        c.x = xy2_out.x;
        c.y = xy2_out.y;

        let w = 1.5 - i / N;
        let sw = sqrt(w);
        c0 += c * sw;
        w0 += sw;
    }

    // 6) Final color math
    c0 = c0 / w0 * 3.0 + 0.5;
    let outColor = vec4<f32>(
        sqrt(c0.r) * 1.2,
        c0.r * c0.r * 0.9,
        c0.r * c0.r * c0.r * 0.4,
        1.0
    );

    // 7) Write to the storage texture
    textureStore(
        output_texture,
        vec2<i32>(i32(global_id.x), i32(global_id.y)),
        vec4<f32>(outColor.a, outColor.rgb)
    );
}
