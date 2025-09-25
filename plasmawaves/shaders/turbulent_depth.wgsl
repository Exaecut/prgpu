const PI: f32 = 3.14159265359;

struct Params {
    time: f32,           // Time in seconds
    debug: u32,
    is_premiere: u32,
    scale: f32,
    time_factor: f32,
    time_offset: f32,
    fractal_repetition: f32,
    twist_frequency: f32,
    twist_amplitude: f32,
    color1: vec4<f32>,   // Color 1
    color2: vec4<f32>,   // Color 2
    color3: vec4<f32>,   // Color 3
    color4: vec4<f32>,   // Color 4
    // color5: vec4<f32>,   // Color 5
    // color6: vec4<f32>,   // Color 6
    // color7: vec4<f32>,   // Color 7
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

// Utility (unchanged)
fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let max_val = max(max(rgb.r, rgb.g), rgb.b);
    let min_val = min(min(rgb.r, rgb.g), rgb.b);
    return (max_val + min_val) * 0.5;
}

fn gradientColor(t: f32, colors: array<vec4<f32>, 4>) -> vec4<f32> {
    let scaled: f32 = clamp(t * 4.0, 0.0, 3.0);
    let idx: u32 = u32(floor(scaled));
    let nxt: u32 = min(idx + 1u, 3u);
    let f: f32 = scaled - f32(idx);
    
    return mix(colors[idx], colors[nxt], f);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Dimensions
    let outSize    = textureDimensions(output);
    let outSizeF32 = vec2<f32>(outSize);
    let gidF32     = vec2<f32>(global_id.xy);

    // Bounds check
    if (gidF32.x >= outSizeF32.x || gidF32.y >= outSizeF32.y) {
        return;
    }

    // Input UV
    let inSize    = textureDimensions(input);
    let inSizeF32 = vec2<f32>(inSize);
    let uv        = (gidF32 + vec2<f32>(0.5)) / inSizeF32;

    var out_color: vec4<f32>;
    let fragCenter = gidF32 + vec2<f32>(0.5);

        // 1) Initialization
    let time = (params.time + params.time_offset) * params.time_factor;
    var depth: f32 = 0.0;
    var colorAccum: vec4<f32> = vec4<f32>(0.0);
    let MAX_STEPS: i32 = i32(params.scale);

    let colors: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
        params.color1,
        params.color2,
        params.color3,
        params.color4,
    );

    // 2) Primary ray-march loop
    for (var iter: i32 = 1; iter <= MAX_STEPS; iter = iter + 1) {
        // Compute ray direction
        let dir = normalize(vec3<f32>(
            2.0 * fragCenter.x - outSizeF32.x,
            2.0 * fragCenter.y - outSizeF32.y,
            -outSizeF32.y
        ));

        // Sample point along ray
        var p: vec3<f32> = depth * dir;
        p.z = p.z - time;

        // 3) Detail-perturbation loop
        var s: f32 = 0.1;
        while (s < 3.0) {
            p = p - dot(cos(time + p * (s * params.fractal_repetition)), vec3<f32>(0.01)) / s;
            p = p + sin(p.yzx * params.twist_frequency) * params.twist_amplitude;
            s = s * 1.42;
        }

        // 4) Advance depth based on “tube” distance
        let tubeRadius = length(p.yx);
        let thickness  = 0.02 + abs(3.0 - tubeRadius) * 0.1;
        depth = depth + thickness;

        // 5) Accumulate color
        let colorIndex = abs(cos(depth));
        colorAccum = colorAccum + (vec4<f32>(0.0) + gradientColor(colorIndex, colors)) / thickness;
    }

    let tonemapped_color = tanh(colorAccum / 2000.0);
    out_color = vec4<f32>(tonemapped_color.a, tonemapped_color.rgb);

    textureStore(output, vec2<i32>(global_id.xy), out_color);
}
