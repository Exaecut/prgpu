// Echoes - smoother blur, uniform rotation, added Max blend mode

const PI: f32 = 3.14159265359;

struct Params {
    // Toggles and modes
    debug: u32,
    preview_layer: u32,   // 1 Final, 2 Threshold, 3 Echoes
    clip_bounds: u32,     // 0 no clip, 1 clip
    blend_mode: u32,      // 1 Add, 2 Screen, 3 Overlay, 4 Color Dodge, 5 Max

    // Echo and transform
    echo: u32,
    transform_position: vec2<f32>, // pixels per echo
    transform_rotation: f32,       // degrees per echo
    transform_scale: f32,          // percent per echo - 100 no scale

    // Look
    threshold: f32,                // 0..100
    threshold_smoothness: f32,     // 0..1
    exposure: f32,
    decay: f32,                    // 0..1 per echo
    tint_color: vec4<f32>,         // 0..1 RGBA

    // Blur
    bluriness: f32                 // 0..500 from UI, treated as px budget
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input_tex: texture_2d<f32>;
@group(0) @binding(2) var output_tex: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var _debug_tex: texture_2d<f32>;

fn rgb_to_lightness(rgb: vec3<f32>) -> f32 {
    let mx = max(max(rgb.r, rgb.g), rgb.b);
    let mn = min(min(rgb.r, rgb.g), rgb.b);
    return 0.5 * (mx + mn);
}

fn threshold_mask(color: vec4<f32>, thr01: f32) -> vec4<f32> {
    let s = params.threshold_smoothness;
    let l = rgb_to_lightness(color.rgb);
    let m = smoothstep(thr01 - s, thr01 + s, l);
    return color * m;
}

fn sample_with_bounds(uv: vec2<f32>) -> vec4<f32> {
    let uvc = select(uv, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), params.clip_bounds != 0u);
    return textureSampleLevel(input_tex, samp, uvc, 0.0);
}

/* Adaptive single pass blur
   - Low radius: fast 9 tap cross+diagonals
   - High radius: 21 tap gaussian rings (axis 0.5r, 1.0r, 1.5r and diagonal r, 2r) */
fn blur9(uv: vec2<f32>, input_size: vec2<f32>, r_px: f32) -> vec4<f32> {
    let texel = 1.0 / input_size;
    let ofs = texel * r_px;
    let diag = ofs * 0.70710678;

    let w_center = 0.40;
    let w_axis = 0.10;
    let w_diag = 0.05;

    var sum = sample_with_bounds(uv) * w_center;

    sum += sample_with_bounds(uv + vec2<f32>(ofs.x, 0.0)) * w_axis;
    sum += sample_with_bounds(uv + vec2<f32>(-ofs.x, 0.0)) * w_axis;
    sum += sample_with_bounds(uv + vec2<f32>(0.0, ofs.y)) * w_axis;
    sum += sample_with_bounds(uv + vec2<f32>(0.0, -ofs.y)) * w_axis;

    sum += sample_with_bounds(uv + vec2<f32>(diag.x, diag.y)) * w_diag;
    sum += sample_with_bounds(uv + vec2<f32>(-diag.x, diag.y)) * w_diag;
    sum += sample_with_bounds(uv + vec2<f32>(diag.x, -diag.y)) * w_diag;
    sum += sample_with_bounds(uv + vec2<f32>(-diag.x, -diag.y)) * w_diag;

    return sum;
}

fn gauss_w(d: f32, sigma: f32) -> f32 {
    // Numerically stable for our ranges
    let x = d / max(sigma, 1e-4);
    return exp(-0.5 * x * x);
}

fn blur21(uv: vec2<f32>, input_size: vec2<f32>, r_px: f32) -> vec4<f32> {
    let texel = 1.0 / input_size;

    // Rings
    let r05 = 0.5 * r_px;
    let r10 = r_px;
    let r15 = 1.5 * r_px;

    // Diagonals are placed so that their radial distance is r and 2r
    let d1 = (r_px) * 0.70710678;     // components for radius r
    let d2 = (2.0 * r_px) * 0.70710678; // components for radius 2r

    let sigma = 0.5 * r_px + 1.0;

    var sum = vec4<f32>(0.0);
    var wsum = 0.0;

    // Center
        {
        let w0 = gauss_w(0.0, sigma);
        sum += sample_with_bounds(uv) * w0;
        wsum += w0;
    }

    // Axis ring 0.5r
        {
        let w = gauss_w(r05, sigma);
        let o = texel * r05;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, -o.y)) * w;
        wsum += 4.0 * w;
    }

    // Axis ring 1.0r
        {
        let w = gauss_w(r10, sigma);
        let o = texel * r10;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, -o.y)) * w;
        wsum += 4.0 * w;
    }

    // Axis ring 1.5r
        {
        let w = gauss_w(r15, sigma);
        let o = texel * r15;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, 0.0)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(0.0, -o.y)) * w;
        wsum += 4.0 * w;
    }

    // Diagonal ring r
        {
        let w = gauss_w(r10, sigma);
        let o = texel * d1;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, -o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, -o.y)) * w;
        wsum += 4.0 * w;
    }

    // Diagonal ring 2r
        {
        let w = gauss_w(2.0 * r10, sigma);
        let o = texel * d2;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(o.x, -o.y)) * w;
        sum += sample_with_bounds(uv + vec2<f32>(-o.x, -o.y)) * w;
        wsum += 4.0 * w;
    }

    return sum / max(wsum, 1e-6);
}

fn single_pass_blur(uv: vec2<f32>, input_size: vec2<f32>, radius_px: f32) -> vec4<f32> {
    if radius_px <= 0.001 { return sample_with_bounds(uv); }
    // Switch to high quality path for larger radii
    return select(blur21(uv, input_size, radius_px), blur9(uv, input_size, radius_px), radius_px < 6.0);
}

// Blends
fn blend_add(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a + b; }
fn blend_screen(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return 1.0 - (1.0 - a) * (1.0 - b); }
fn blend_overlay(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    let t = step(vec4<f32>(0.5), a);
    return mix(2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b), t);
}
fn blend_color_dodge(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return a / max(vec4<f32>(1e-5), 1.0 - b); }
fn blend_max(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> { return max(a, b); }

fn blend_pixel(base: vec4<f32>, echo_col: vec4<f32>) -> vec4<f32> {
    switch (params.blend_mode) {
        case 1u: { return blend_add(base, echo_col); }
        case 2u: { return blend_screen(base, echo_col); }
        case 3u: { return blend_overlay(base, echo_col); }
        case 4u: { return blend_color_dodge(base, echo_col); }
        case 5u: { return blend_max(base, echo_col); }
        default: { return echo_col; }
    }
}

// Echo accumulation with uniform rotation in pixel space
fn compute_echoes(uv: vec2<f32>, center_uv: vec2<f32>, input_size: vec2<f32>) -> vec4<f32> {
    // Work in pixels for uniform rotation regardless of aspect ratio
    let uv_px = uv * input_size;
    let center_px = center_uv * input_size;

    let delta_px = params.transform_position;                // pixels per echo
    let step_theta = params.transform_rotation * (PI / 180.0);
    let c = cos(step_theta);
    let s = sin(step_theta);

    var cos_i = 1.0;
    var sin_i = 0.0;

    var scale_i = 1.0;
    let scale_step = max(params.transform_scale, 0.0001) * 0.01;

    var shift_px = vec2<f32>(0.0);
    var decay_factor = 1.0;

    // Blur growth per echo - treat UI value as px budget, linear growth
    let blur_step_px = params.bluriness * 0.08; // 500 -> 40 px, 100 -> 8 px

    var acc = vec4<f32>(0.0);
    var i: u32 = 1u;
    let max_i = params.echo;

    loop {
        if i > max_i { break; }
        if decay_factor < 0.001 { break; }

        // advance rotation
        let cos_next = cos_i * c - sin_i * s;
        let sin_next = sin_i * c + cos_i * s;
        cos_i = cos_next;
        sin_i = sin_next;

        // advance scale and translation
        scale_i = scale_i * scale_step;
        shift_px = shift_px + delta_px;

        // inverse mapping in pixel space
        let p0 = uv_px - shift_px - center_px;

        // inverse rotate by theta_i
        let prx = cos_i * p0.x + sin_i * p0.y;
        let pry = -sin_i * p0.x + cos_i * p0.y;

        // inverse scale
        let inv_scale = 1.0 / max(scale_i, 1e-4);
        let sample_px = vec2<f32>(prx, pry) * inv_scale + center_px;
        let sample_uv = sample_px / input_size;

        // radius grows linearly with echo index
        let radius_px = f32(i) * blur_step_px;

        let blurred = single_pass_blur(sample_uv, input_size, radius_px);
        let masked = threshold_mask(blurred, params.threshold * 0.01);
        let tinted = vec4<f32>(masked.r * params.tint_color.a, masked.gba * params.tint_color.rgb);

        acc += tinted * decay_factor;
        decay_factor *= params.decay;

        i += 1u;
    }

    return acc * params.exposure;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_size = textureDimensions(output_tex);
    if gid.x >= out_size.x || gid.y >= out_size.y { return; }

    let in_size = textureDimensions(input_tex);
    let in_size_f = vec2<f32>(in_size);
    let out_size_f = vec2<f32>(out_size);

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / in_size_f;
    let base = textureSampleLevel(input_tex, samp, uv, 0.0);

    if params.debug != 0u {
        let dbg = vec4<f32>(uv, 0.0, 1.0);
        textureStore(output_tex, vec2<i32>(gid.xy), clamp(base + dbg, vec4<f32>(0.0), vec4<f32>(1.0)));
        return;
    }

    // Use the visual center mapped to input uv
    let center_uv = (vec2<f32>(0.5, 0.5) * out_size_f) / in_size_f;

    var out_col: vec4<f32>;

    switch (params.preview_layer) {
        case 1u: { // Final
            let echoes = compute_echoes(uv, center_uv, in_size_f);
            out_col = blend_pixel(base, echoes);
        }
        case 2u: { // Threshold
            out_col = threshold_mask(base, params.threshold * 0.01);
        }
        case 3u: { // Echoes only
            out_col = compute_echoes(uv, center_uv, in_size_f);
        }
        default: {
            out_col = base;
        }
    }

    textureStore(output_tex, vec2<i32>(gid.xy), out_col);
}
