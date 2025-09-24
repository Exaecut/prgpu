// Parameters
struct Params {
    time: f32,                  // Time in seconds
    debug: u32,
    is_premiere: u32,
    preview_layer: u32,
    threshold1: f32,           // First luminance threshold (default: 0.0)
    threshold2: f32,           // Second luminance threshold (default: 0.5)
    increment_base: f32,       // Base increment divisor (default: 30.0)
    increment_random_range: f32 // Random increment range (default: 6.0)
};

// Bindings
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var input: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var input_sampler: sampler;
@group(0) @binding(4) var debug_texture: texture_2d<f32>;

// Shared memory for per-column threshold rows
var<workgroup> first_row_shared: array<u32, 8>;
var<workgroup> second_row_shared: array<u32, 8>;

//"Random" function borrowed from The Book of Shaders: Random
fn random(xy: vec2<f32>) -> f32 {
    return fract(sin(dot(xy, vec2<f32>(12.0, 78.0))));
}

fn luminance(color: vec4<f32>) -> f32 {
    return (color.r * 0.3 + color.g * 0.6 + color.b * 0.1) * color.a;
}

// Returns the y coordinate of the first pixel that is brighter than the threshold
fn getFirstThresholdPixel(xy: vec2<f32>, threshold: f32, resolution: vec2<f32>) -> f32 {
    var luma = luminance(textureSampleLevel(input, input_sampler, xy / resolution, 0.0));

    //Looking at every sequential pixel is very resource intensive,
    //thus, we'll increment the inspected pixel by dividing the image height in sections,
    //and add a little randomness across the x axis to hide the division of said sections
    let increment = resolution.y / (params.increment_base + (random(xy.xx) * params.increment_random_range));

    //Check if the luminance of the current pixel is brighter than the threshold,
    //if not, check the next pixel
    var current_y = xy.y;
    while (luma <= threshold) {
        current_y -= increment;
        if (current_y <= 0.0) {
            return 0.0;
        }
        let uv = vec2<f32>(xy.x, current_y) / resolution;
        luma = luminance(textureSampleLevel(input, input_sampler, uv, 0.0));
    }
    return current_y;
}

//Puts 10 pixels in an array
fn putItIn(startxy: vec2<f32>, size: f32, resolution: vec2<f32>) -> array<vec4<f32>, 10> {
    var colorarray: array<vec4<f32>, 10>;
    for (var j = 9; j >= 0; j = j - 1) {
        //Divide the line of pixels into 10 sections,
        //then store the pixel found at the junction of each section
        let xy = vec2<f32>(startxy.x, startxy.y + (size / 9.0) * f32(j));
        colorarray[u32(j)] = textureSampleLevel(input, input_sampler, xy / resolution, 0.0);
    }
    return colorarray;
}

//An attempt at Bubble sort for 10 pixels, sorting them from darkest to brightest, top to bottom
fn sortArray(colorarray: array<vec4<f32>, 10>) -> array<vec4<f32>, 10> {
    var arr = colorarray;
    var swapped = true;
    while (swapped) {
        swapped = false;
        for (var j = 9; j > 0; j = j - 1) {
            if (luminance(arr[j]) > luminance(arr[j - 1])) {
                let tempcolor = arr[j];
                arr[j] = arr[j - 1];
                arr[j - 1] = tempcolor;
                swapped = true;
            }
        }
    }
    return arr;
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let output_size = textureDimensions(output);
    let resolution = vec2<f32>(output_size);
    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {
        return;
    }

    let fragCoord = vec2<f32>(f32(global_id.x), f32(global_id.y));
    if params.debug == 1u {
        let uv = fragCoord / resolution;
        let debug_color = textureSampleLevel(input, input_sampler, uv, 0.0);
        textureStore(output, vec2<i32>(global_id.xy), (debug_color * 1.0) + (vec4<f32>(1.0, vec2<f32>(uv.xy), 0.0) * 1.0));
        return;
    }

    // Compute firsty and secondy for the column, only for local_id.y == 0
    if (local_id.y == 0u) {
        let firsty = getFirstThresholdPixel(vec2<f32>(fragCoord.x, resolution.y), params.threshold1, resolution);
        let secondy = getFirstThresholdPixel(vec2<f32>(fragCoord.x, firsty - 1.0), params.threshold2, resolution);
        first_row_shared[local_id.x] = u32(firsty);
        second_row_shared[local_id.x] = u32(secondy);
    }

    // Synchronize workgroup to ensure shared memory is populated
    workgroupBarrier();

    if (params.preview_layer - 1u == 1u) {
        textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(1.0, f32(first_row_shared[local_id.x]), f32(second_row_shared[local_id.x]), 0.0));
        return;
    }

    // Retrieve threshold rows for this column
    let firsty = f32(first_row_shared[local_id.x]);
    let secondy = f32(second_row_shared[local_id.x]);

    //Only work on the pixels that are between the two threshold pixels
    if (fragCoord.y < firsty && fragCoord.y > secondy) {
        let size = firsty - secondy;
        var colorarray = putItIn(vec2<f32>(fragCoord.x, secondy), size, resolution);
        colorarray = sortArray(colorarray);

        let sectionSize = size / 9.0;
        let location = floor((fragCoord.y - secondy) / sectionSize);
        let bottom = secondy + (sectionSize * location);
        let locationBetween = (fragCoord.y - bottom) / sectionSize;

        //A simple method for "fading" between the colors of our ten sampled pixels
        let topColor = colorarray[i32(location) + 1] * locationBetween;
        let bottomColor = colorarray[i32(location)] * (1.0 - locationBetween);

        let fragColor = topColor + bottomColor;
        textureStore(output, vec2<i32>(global_id.xy), fragColor);
    } else {
        let uv = fragCoord / resolution;
        let original_color = textureSampleLevel(input, input_sampler, uv, 0.0);
        textureStore(output, vec2<i32>(global_id.xy), original_color);
    }
}