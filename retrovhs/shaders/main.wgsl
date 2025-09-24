const PI: f32 = 3.1415926;

const LAYER_FLAG_DISTORTION: u32 = 1u << 0;
const LAYER_FLAG_COMPRESSION: u32 = 1u << 1;
const LAYER_FLAG_FILTER: u32 = 1u << 3;

fn layerFlagEnabled(layer: u32, flag: u32) -> bool {
    return (layer & flag) != 0;
}

const TEMP_HORIZONTAL_DISTORTION: f32 = 1.5;
const TEMP_VERTICAL_DISTORTION: f32 = 1.5;

const TEMP_VIGNETTE_STRENGTH: f32 = 0.3;

const TEMP_TAPE_NOISE_LOWFREQ_GLITCH: f32 = 0.005;
const TEMP_TAPE_NOISE_HIGHFREQ_GLITCH: f32 = 0.01;

const TEMP_TAPE_NOISE_HORIZONTAL_OFFSET: f32 = 0.5;
const TEMP_TAPE_NOISE_VERTICAL_OFFSET: f32 = 0.5;

const TEMP_CREASE_PHASE_FREQUENCY: f32 = 9.0;
const TEMP_CREASE_SPEED: f32 = 1.3;
const TEMP_CREASE_HEIGHT: f32 = 0.9;
const TEMP_CREASE_DEPTH: f32 = 0.01;
const TEMP_CREASE_INTENSITY: f32 = 10.0;
const TEMP_CREASE_NOISE_FREQUENCY: f32 = 100.0;
const TEMP_CREASE_STABILITY: f32 = 0.5;
const TEMP_CREASE_MINIMUM: f32 = 0.0;

const TEMP_INTENSE_NOISE_HEIGHT_PROPORTION: f32 = 0.025;
const TEMP_SIDE_LEAK_INTENSITY: f32 = 0.8;

const TEMP_BLOOM_EXPOSURE: f32 = 0.1;

struct Params {
    // Time in seconds
    time: f32,
    debug: u32,

    uv_mode: u32, // 0: normal, 1: 4:3
    horizontal_distortion: f32,
    vertical_distortion: f32,
    vignette_strength: f32,
    tint_color_r: u32,
    tint_color_g: u32,
    tint_color_b: u32,
    tint_color_a: u32,
    bloom_exposure: f32,
    pixel_cell_size: f32,
    scanline_hardness: f32,
    pixel_hardness: f32,
    bloom_scanline_hardness: f32,
    bloom_pixel_hardness: f32,
    crt_contrast: f32,
    enabled_layers: u32,
    is_premiere: u32,
}

const MASK_DARKNESS: f32 = 0.5;
const MASK_BRIGHTNESS: f32 = 1.5;

// Bindings
@group(0) @binding(0)
var<uniform> params: Params;
@group(0) @binding(1)
var input: texture_2d<f32>;
@group(0) @binding(2)
var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3)
var input_sampler: sampler;

fn fitUVToAspect(texture: texture_2d<f32>, uv: vec2<f32>) -> vec2<f32> {
    let resolution = vec2<f32>(textureDimensions(texture));
    
    if (resolution.x == 0.0 || resolution.y == 0.0) {
        return uv;
    }

    let src_aspect = resolution.x / resolution.y;
    let target_aspect = select(4.0 / 3.0, 3.0 / 4.0, src_aspect < 1.0);

    var out_uv = uv;

    if (src_aspect > target_aspect) {
        let crop_x = target_aspect / src_aspect;
        out_uv.x = uv.x / crop_x - (1.0 - crop_x) * 0.5;
    } else {
        let crop_y = src_aspect / target_aspect;
        out_uv.y = uv.y / crop_y - (1.0 - crop_y) * 0.5;
    }

    return out_uv;
}

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

fn curve(in_uv: vec2<f32>) -> vec2<f32> {
    var warp: f32 = 0.75;

    var dist: vec2<f32> = abs(in_uv - vec2<f32>(0.5));
    dist *= dist;

    var uv: vec2<f32> = in_uv;
    uv.x -= 0.5;
    uv.x *= 1.0 + (dist.y * (0.3 * (warp * params.horizontal_distortion)));
    uv.x += 0.5;

    uv.y -= 0.5;
    uv.y *= 1.0 + (dist.x * (0.3 * (warp * params.vertical_distortion)));
    uv.y += 0.5;

    return uv;
}

fn vignette(in_uv: vec2<f32>) -> f32 {
    var vignette: f32 = (0.0 + 1.0 * 16.0 * in_uv.x * in_uv.y * (1.0 - in_uv.x) * (1.0 - in_uv.y));
    vignette = pow(vignette, params.vignette_strength);
    return vignette;
}

fn distortionClipping(in_color: vec4<f32>, in_raw_uv: vec2<f32>, in_curved_uv: vec2<f32>) -> vec4<f32> {
    var out_color: vec4<f32> = in_color;

    if (in_curved_uv.x < 0.0 || in_curved_uv.x > 1.0 || in_curved_uv.y < 0.0 || in_curved_uv.y > 1.0) {
        out_color *= vec4<f32>(1.0, 0.0, 0.0, 0.0);
        out_color.r = 1.0;
    }

    out_color.g *= vignette(in_curved_uv);
    out_color.b *= vignette(in_curved_uv);
    out_color.a *= vignette(in_curved_uv);

    return out_color;
}

fn crtContrast(in_color: vec3<f32>) -> vec3<f32> {
    return in_color * (1.0 / params.crt_contrast) + in_color * in_color * in_color;
}

fn safeFetch(tex: texture_2d<f32>, uv: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
    let resolution: vec2<f32> = vec2<f32>(textureDimensions(tex)) / vec2<f32>(params.pixel_cell_size);
    let position: vec2<f32> = floor(uv * resolution + offset) / resolution;

    if (max(abs(position.x - 0.5), abs(position.y - 0.5)) > 0.5) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    return crtContrast(textureSampleLevel(tex, input_sampler, position, 0.0).gba);
}

fn distanceToNearestPixel(tex: texture_2d<f32>, in_uv: vec2<f32>) -> vec2<f32> {
    let resolution: vec2<f32> = vec2<f32>(textureDimensions(tex)) / vec2<f32>(params.pixel_cell_size);
    let position: vec2<f32> = resolution * in_uv;

    return -((position - floor(position)) - vec2<f32>(0.5));
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    return exp2(sigma * x * x);
}

fn pixelMask(in_uv: vec2<f32>) -> vec3<f32> {
    var uv: vec2<f32> = floor(in_uv * vec2<f32>(1.0, 0.5));
    uv.x += uv.y * 3.0;

    var mask: vec3<f32> = vec3<f32>(MASK_DARKNESS, MASK_DARKNESS, MASK_DARKNESS);
    uv.x = fract(uv.x / 6.0);
    if (uv.x < 0.333) {
        mask.r = MASK_BRIGHTNESS;
    } else if (uv.x < 0.666) {
        mask.g = MASK_BRIGHTNESS;
    } else {
        mask.b = MASK_BRIGHTNESS;
    }

    return mask;
}

fn crtFilter(in_tex: texture_2d<f32>, in_coord: vec2<f32>, in_uv: vec2<f32>) -> vec4<f32> {
    var uv: vec2<f32> = in_coord / vec2<f32>(textureDimensions(in_tex));
    let distance: vec2<f32> = distanceToNearestPixel(in_tex, uv);

    // Horizontal weights for original input
    let hw_input: array<f32, 5> = array<f32, 5>(
        gaussian(distance.x - 2.0, params.pixel_hardness),
        gaussian(distance.x - 1.0, params.pixel_hardness),
        gaussian(distance.x, params.pixel_hardness),
        gaussian(distance.x + 1.0, params.pixel_hardness),
        gaussian(distance.x + 2.0, params.pixel_hardness),
    );

    // Horizontal weights for bloom
    let hw_bloom: array<f32, 7> = array<f32, 7>(
        gaussian(distance.x - 3.0, params.bloom_pixel_hardness),
        gaussian(distance.x - 2.0, params.bloom_pixel_hardness),
        gaussian(distance.x - 1.0, params.bloom_pixel_hardness),
        gaussian(distance.x, params.bloom_pixel_hardness),
        gaussian(distance.x + 1.0, params.bloom_pixel_hardness),
        gaussian(distance.x + 2.0, params.bloom_pixel_hardness),
        gaussian(distance.x + 3.0, params.bloom_pixel_hardness),
    );

    // Scanline weights for original input
    let sw_input: array<f32, 3> = array<f32, 3>(
        gaussian(distance.y - 1.0, params.scanline_hardness),
        gaussian(distance.y, params.scanline_hardness),
        gaussian(distance.y + 1.0, params.scanline_hardness),
    );

    // Scanline weights for bloom
    let sw_bloom: array<f32, 5> = array<f32, 5>(
        gaussian(distance.y - 2.0, params.bloom_scanline_hardness),
        gaussian(distance.y - 1.0, params.bloom_scanline_hardness),
        gaussian(distance.y, params.bloom_scanline_hardness),
        gaussian(distance.y + 1.0, params.bloom_scanline_hardness),
        gaussian(distance.y + 2.0, params.bloom_scanline_hardness),
    );

    // Fetch unique samples
    // y_off: -2; x_off: -2..2
    let samples_m2: array<vec3<f32>, 5> = array<vec3<f32>, 5>(
        safeFetch(in_tex, uv, vec2<f32>(-2.0, -2.0)),
        safeFetch(in_tex, uv, vec2<f32>(-1.0, -2.0)),
        safeFetch(in_tex, uv, vec2<f32>(0.0, -2.0)),
        safeFetch(in_tex, uv, vec2<f32>(1.0, -2.0)),
        safeFetch(in_tex, uv, vec2<f32>(2.0, -2.0))
    );

    // y_off: -1; x_off: -3..3
    let samples_m1: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
        safeFetch(in_tex, uv, vec2<f32>(-3.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(-2.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(-1.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(0.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(1.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(2.0, -1.0)),
        safeFetch(in_tex, uv, vec2<f32>(3.0, -1.0))
    );

    // y_off: 0; x_off: -3..3
    let samples_0: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
        safeFetch(in_tex, uv, vec2<f32>(-3.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(-2.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(-1.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(0.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(1.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(2.0, 0.0)),
        safeFetch(in_tex, uv, vec2<f32>(3.0, 0.0))
    );

    // y_off: 1; x_off: -3..3
    let samples_p1: array<vec3<f32>, 7> = array<vec3<f32>, 7>(
        safeFetch(in_tex, uv, vec2<f32>(-3.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(-2.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(-1.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(0.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(1.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(2.0, 1.0)),
        safeFetch(in_tex, uv, vec2<f32>(3.0, 1.0))
    );

    // y_off: 2; x_off: -2..2
    let samples_p2: array<vec3<f32>, 5> = array<vec3<f32>, 5>(
        safeFetch(in_tex, uv, vec2<f32>(-2.0, 2.0)),
        safeFetch(in_tex, uv, vec2<f32>(-1.0, 2.0)),
        safeFetch(in_tex, uv, vec2<f32>(0.0, 2.0)),
        safeFetch(in_tex, uv, vec2<f32>(1.0, 2.0)),
        safeFetch(in_tex, uv, vec2<f32>(2.0, 2.0))
    );

    // Compute three taps triangle
    let h3_tri: f32 = hw_input[1] + hw_input[2] + hw_input[3];
    let horz3_m1: vec3<f32> = (
        samples_m1[2] * hw_input[1] +  // x_off=-1
        samples_m1[3] * hw_input[2] +  // x_off=0
        samples_m1[4] * hw_input[3]    // x_off=1
    ) / h3_tri;

    let horz3_p1: vec3<f32> = (
        samples_p1[2] * hw_input[1] +  // x_off=-1
        samples_p1[3] * hw_input[2] +  // x_off=0
        samples_p1[4] * hw_input[3]    // x_off=1
    ) / h3_tri;

    let h5_tri: f32 = hw_input[0] + hw_input[1] + hw_input[2] + hw_input[3] + hw_input[4];
    let horz5_0: vec3<f32> = (
        samples_0[1] * hw_input[0] +  // x_off=-2
        samples_0[2] * hw_input[1] +  // x_off=-1
        samples_0[3] * hw_input[2] +  // x_off=0
        samples_0[4] * hw_input[3] +  // x_off=1
        samples_0[5] * hw_input[4]    // x_off=2
    ) / h5_tri;

    let tri: vec3<f32> = horz3_m1 * sw_input[0] + horz5_0 * sw_input[1] + horz3_p1 * sw_input[2];

    // Compute Bloom
    let h5_bloom: f32 = hw_bloom[1] + hw_bloom[2] + hw_bloom[3] + hw_bloom[4] + hw_bloom[5];
    let horz5_m2: vec3<f32> = (
        samples_m2[0] * hw_bloom[1] +  // x_off=-2
        samples_m2[1] * hw_bloom[2] +  // x_off=-1
        samples_m2[2] * hw_bloom[3] +  // x_off=0
        samples_m2[3] * hw_bloom[4] +  // x_off=1
        samples_m2[4] * hw_bloom[5]    // x_off=2
    ) / h5_bloom;

    let horz5_p2: vec3<f32> = (
        samples_p2[0] * hw_bloom[1] +  // x_off=-2
        samples_p2[1] * hw_bloom[2] +  // x_off=-1
        samples_p2[2] * hw_bloom[3] +  // x_off=0
        samples_p2[3] * hw_bloom[4] +  // x_off=1
        samples_p2[4] * hw_bloom[5]    // x_off=2
    ) / h5_bloom;

    let h7_bloom: f32 = hw_bloom[0] + hw_bloom[1] + hw_bloom[2] + hw_bloom[3] + hw_bloom[4] + hw_bloom[5] + hw_bloom[6];
    let horz7_m1: vec3<f32> = (
        samples_m1[0] * hw_bloom[0] +  // x_off=-3
        samples_m1[1] * hw_bloom[1] +  // x_off=-2
        samples_m1[2] * hw_bloom[2] +  // x_off=-1
        samples_m1[3] * hw_bloom[3] +  // x_off=0
        samples_m1[4] * hw_bloom[4] +  // x_off=1
        samples_m1[5] * hw_bloom[5] +  // x_off=2
        samples_m1[6] * hw_bloom[6]    // x_off=3
    ) / h7_bloom;

    let horz7_0: vec3<f32> = (
        samples_0[0] * hw_bloom[0] +  // x_off=-3
        samples_0[1] * hw_bloom[1] +  // x_off=-2
        samples_0[2] * hw_bloom[2] +  // x_off=-1
        samples_0[3] * hw_bloom[3] +  // x_off=0
        samples_0[4] * hw_bloom[4] +  // x_off=1
        samples_0[5] * hw_bloom[5] +  // x_off=2
        samples_0[6] * hw_bloom[6]    // x_off=3
    ) / h7_bloom;

    let horz7_p1: vec3<f32> = (
        samples_p1[0] * hw_bloom[0] +  // x_off=-3
        samples_p1[1] * hw_bloom[1] +  // x_off=-2
        samples_p1[2] * hw_bloom[2] +  // x_off=-1
        samples_p1[3] * hw_bloom[3] +  // x_off=0
        samples_p1[4] * hw_bloom[4] +  // x_off=1
        samples_p1[5] * hw_bloom[5] +  // x_off=2
        samples_p1[6] * hw_bloom[6]    // x_off=3
    ) / h7_bloom;

    let bloom: vec3<f32> = (
        horz5_m2 * sw_bloom[0] +
        horz7_m1 * sw_bloom[1] +
        horz7_0 * sw_bloom[2] +
        horz7_p1 * sw_bloom[3] +
        horz5_p2 * sw_bloom[4]
    );

    let color: vec3<f32> = tri * pixelMask(in_coord) + bloom * params.bloom_exposure;
    return vec4<f32>(1.0, color);
}

// Main compute shader
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let layers = params.enabled_layers;
    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }

    let input_size = textureDimensions(input);
    let input_size_f32 = vec2<f32>(input_size);

    var uv = (global_id_f32.xy + vec2<f32>(0.5)) / input_size_f32.xy;
    uv = select(uv, fitUVToAspect(input, uv), params.uv_mode - 1u == 1u);

    var curved_uv = curve(uv);
    var curved_coord = curve(uv) * input_size_f32.xy;

    var final_uv = select(uv, curved_uv, layerFlagEnabled(layers, LAYER_FLAG_DISTORTION));
    var final_coord = select(global_id_f32, curved_coord, layerFlagEnabled(layers, LAYER_FLAG_DISTORTION));
    var final_coord_f32 = vec2<f32>(f32(final_coord.x), f32(final_coord.y));

    var color = textureSampleLevel(input, input_sampler, uv, 0.0);
    var out_color = vec4<f32>(0.0);

    if params.debug == 1u {
        var debug_color = textureSampleLevel(input, input_sampler, uv, 0.0);
        textureStore(output, vec2<i32>(global_id.xy), debug_color);
    }
    else {
        out_color = textureSampleLevel(input, input_sampler, final_uv, 0.0);
        
        if layerFlagEnabled(layers, LAYER_FLAG_FILTER) {
            out_color = crtFilter(input, final_coord_f32, final_uv);
        }
        
        if layerFlagEnabled(layers, LAYER_FLAG_DISTORTION) {
            out_color = distortionClipping(out_color, uv, curved_uv);
        }

        out_color.g *= f32(params.tint_color_r) / 255.0;
        out_color.b *= f32(params.tint_color_g) / 255.0;
        out_color.a *= f32(params.tint_color_b) / 255.0;

        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, out_color.g, out_color.b, out_color.a));
    }
}