use std::fs;
use std::path::Path;

/// Writes tightly-packed BGRA pixels to a PNG. Creates parent directories.
pub fn write_png(
    path: impl AsRef<Path>,
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
) -> Result<(), String> {
    let path = path.as_ref();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir -p {}: {e}", parent.display()))?;
    }

    // The `image` crate expects RGBA. Our data is BGRA.
    let rgba = if bytes_per_pixel == 4 {
        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(format!(
                "data length {} != expected {} ({}x{}x4)",
                data.len(),
                expected,
                width,
                height
            ));
        }
        let mut swizzled = data.to_vec();
        for chunk in swizzled.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        swizzled
    } else {
        data.to_vec()
    };

    let img = match image::RgbaImage::from_raw(width, height, rgba) {
        Some(img) => img,
        None => return Err("image::RgbaImage::from_raw failed".into()),
    };

    img.save(path)
        .map_err(|e| format!("failed to write {}: {e}", path.display()))
}

pub fn write_raw(path: impl AsRef<Path>, data: &[u8]) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir -p {}: {e}", parent.display()))?;
    }
    fs::write(path, data).map_err(|e| format!("write {}: {e}", path.display()))
}
