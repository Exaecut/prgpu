// Parameters
struct Params {
    time: f32,                  // Time in seconds
    clip: u32,
    mode: u32,
    steps: u32,
	spread: f32,
    angle: f32,
    debug: u32,
    is_premiere: u32,
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(3) var input_sampler: sampler;

fn polar_to_cartesian(angle_deg: f32, distance: f32) -> vec2<f32> {
    let radians = angle_deg * 3.14159265 / 180.0;
    return vec2<f32>(cos(radians), sin(radians)) * distance;
}

const PI: f32 = 3.141592653589793;

fn applyHue(color: vec3<f32>, shift: f32) -> vec4<f32> {
    let base: vec3<f32> = vec3<f32>(0.57735026);
    let p: vec3<f32>    = base * dot(base, color);
    let u: vec3<f32>    = color - p;
    let v: vec3<f32>    = cross(base, u);
    
    let angle: f32      = shift * 6.2832;
    let c: f32          = cos(angle);
    let s: f32          = sin(angle);
    
    let rotated: vec3<f32> = u * c + v * s + p;
    return vec4<f32>(rotated, 1.0);
}

fn compute_offset(uv: vec2<f32>, texel_size: vec2<f32>, step: f32, bias: f32) -> vec2<f32> {
    return uv + (texel_size * step);
}

fn reinhard_extended(rgb: vec3<f32>, white_point: f32) -> vec3<f32> {
    let luminance = dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)); // BT.709 luminance
    let scale = 1.0 + luminance / (white_point * white_point);
    return rgb * scale / (1.0 + luminance);
}

fn distance_from_center(uv: vec2<f32>) -> f32 {
    let center: vec2<f32> = vec2<f32>(0.5, 0.5);
    let max_distance: f32 = sqrt(2.0) / 2.0; // Distance from (0.5, 0.5) to a corner
    let dist: f32 = length(uv - center); // Euclidean distance from center
    return dist / max_distance; // Normalize so edge (corner) outputs 1.0
}

fn twirl_uv(uv: vec2<f32>, max_angle_degrees: f32) -> vec2<f32> {
    let center: vec2<f32> = vec2<f32>(0.5, 0.5);
    let normalized_dist: f32 = pow(distance_from_center(uv), 4.0); // Distance in [0, 1]
    let angle_radians: f32 = normalized_dist * max_angle_degrees * 3.14159265359 / 180.0; // Scale angle and convert to radians
    
    // Translate UV to origin (relative to center)
    let translated_uv: vec2<f32> = uv - center;
    
    // Create 2D rotation matrix
    let cos_a: f32 = cos(angle_radians);
    let sin_a: f32 = sin(angle_radians);
    let rotation_matrix: mat2x2<f32> = mat2x2<f32>(
        vec2<f32>(cos_a, -sin_a),
        vec2<f32>(sin_a, cos_a)
    );
    
    // Apply rotation and translate back
    let rotated_uv: vec2<f32> = rotation_matrix * translated_uv + center;
    
    return rotated_uv;
}

// Main compute shader
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_size = textureDimensions(output);
    let output_size_f32 = vec2<f32>(output_size);
    let input_size = textureDimensions(input);
    let input_size_f32 = vec2<f32>(input_size);

    let global_id_f32 = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if global_id_f32.x >= output_size_f32.x || global_id_f32.y >= output_size_f32.y {
        return;
    }


    var uv = (global_id_f32.xy + 0.5) / input_size_f32.xy;
    let texel_size: vec2<f32> = vec2<f32>(1.0 / input_size_f32.x, 1.0 / input_size_f32.y);
    let distance = params.spread / 100.0;
    
    if (params.is_premiere == 0 && params.clip == 0) {
        uv -= (input_size_f32.xy * distance) / input_size_f32.xy;
    }

    var out_color: vec4<f32> = textureSampleLevel(input, input_sampler, uv, 0.0);

    let step_size: f32 = distance / f32(max(params.steps - 1u, 1u));
    var accumulated_color = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (params.mode == 1) {
        // Directional blur mode
        for (var i = 0.0; i < f32(params.steps); i += 1.0) {
            let t: f32 = i * step_size;
            let t2: f32 = i * step_size - distance / 2.0;
            
            accumulated_color.g += textureSampleLevel(input, input_sampler, uv + polar_to_cartesian(params.angle, 1.0) * t, 0.0).g / f32(params.steps);
            accumulated_color.b += textureSampleLevel(input, input_sampler, uv + polar_to_cartesian(params.angle, 0.5) * t2, 0.0).b / f32(params.steps);
            accumulated_color.a += textureSampleLevel(input, input_sampler, uv + polar_to_cartesian(params.angle, -1.0) * t, 0.0).a / f32(params.steps);
        }
    } else if (params.mode == 2) {
        // Radial blur mode
        for (var i = 0.0; i < f32(params.steps); i += 1.0) {
            let distanceFromCenter = length(uv - vec2<f32>(0.5, 0.5)) / (sqrt(2.0) / 2.0);
            let direction = normalize(uv - vec2<f32>(0.5, 0.5)) * pow(distanceFromCenter, 4.0);
            let t: f32 = i * step_size;
            let t2: f32 = (i * step_size - distance / 2.0);
            
            accumulated_color.g += textureSampleLevel(input, input_sampler, twirl_uv(uv, (params.angle * params.spread / 100.0)) + direction * t, 0.0).g / f32(params.steps);
            accumulated_color.b += textureSampleLevel(input, input_sampler, twirl_uv(uv, (params.angle * params.spread / 100.0) * 1.5) + direction * 0.5 * t2, 0.0).b / f32(params.steps);
            accumulated_color.a += textureSampleLevel(input, input_sampler, twirl_uv(uv, (params.angle * params.spread / 100.0) * 2.5) + direction * -t, 0.0).a / f32(params.steps);
        }
    }
    
    accumulated_color.r = saturate(out_color.r + accumulated_color.g + accumulated_color.b + accumulated_color.a);
    let original = textureSampleLevel(input, input_sampler, uv, 0.0);
    out_color = accumulated_color * 0.6 + original * (1.0 - 0.6);

    if (params.debug == 1) {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, uv.x, uv.y, 0.0));
    } else {
        textureStore(output, vec2<i32>(global_id.xy), out_color);
    }
}