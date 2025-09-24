@group(0) @binding(0) var input: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

var<push_constant> dst_offset: vec2<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let color = textureLoad(input, vec2<i32>(gid.xy), 0);
    textureStore(output, vec2<i32>(gid.xy + dst_offset), color);
}