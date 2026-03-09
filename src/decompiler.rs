use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::constants::{Flags, GIN_MAGIC_NUMBER};

// ─── Binary layout constants ──────────────────────────────────────────────────

// Main header: u32 magic, u32 ver, u8 res[8], u8 id[16], u32 res2, u8 path[256], u32 count, u8 check[16]
const HEADER_MAGIC_OFFSET: usize = 0;
const HEADER_SECTION_COUNT_OFFSET: usize = 4 + 4 + 8 + 16 + 4 + 256; // = 292
const HEADER_SIZE: usize = HEADER_SECTION_COUNT_OFFSET + 4 + 16; // = 312

// Section entry: u8 name[64], u64 offset, u32 size, u32 c_size, u32 flags, u8 id[16], u32 ver, u8 id2[16], u8 id3[16]
const SECT_NAME_SIZE: usize = 64;
const SECT_OFFSET_OFF: usize = SECT_NAME_SIZE; // 64
const SECT_SIZE_OFF: usize = SECT_OFFSET_OFF + 8; // 72
const SECT_CSIZE_OFF: usize = SECT_SIZE_OFF + 4; // 76
const SECT_FLAGS_OFF: usize = SECT_CSIZE_OFF + 4; // 80
const SECT_ENTRY_SIZE: usize = SECT_FLAGS_OFF + 4 + 16 + 4 + 16 + 16; // = 136

// Offset within a section's decompressed data where the embedded .gin path starts
const GIN_PATH_OFFSET: usize = 20;

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

/// Read a null-terminated UTF-8 string from a fixed-size byte slice.
fn read_cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Sanitise a section name so it is safe to use as a filename.
fn safe_filename(raw: &str) -> String {
    raw.chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
        .collect()
}

/// Decompress section data according to its flags.
fn decompress_data(data: &[u8], flags: Flags, original_size: usize) -> Option<Vec<u8>> {
    if flags.contains(Flags::ZSTD) {
        zstd::decode_all(data).ok()
    } else if flags.contains(Flags::LZ4) {
        lz4_flex::decompress(data, original_size).ok()
    } else {
        Some(data.to_vec())
    }
}

/// Read up to 512 bytes from `path` starting at `offset`, then scan for either
/// a null byte or a `.gin` suffix — whichever comes first.  Returns the bytes
/// up to (but not including) that terminator.
///
/// This replaces the original Python implementation that opened the file and
/// read one byte at a time, which was catastrophically slow at 150 k files.
fn read_gin_path_from_binary(path: &Path) -> Option<String> {
    let mut f = fs::File::open(path).ok()?;
    f.seek(SeekFrom::Start(GIN_PATH_OFFSET as u64)).ok()?;

    let mut buf = [0u8; 512];
    let n = f.read(&mut buf).ok()?;
    let buf = &buf[..n];

    // Find the end: null byte or the first position after ".gin"
    let mut end = buf.len();
    for i in 0..buf.len() {
        if buf[i] == 0 {
            end = i;
            break;
        }
        if i >= 3 && &buf[i - 3..=i] == b".gin" {
            end = i + 1;
            break;
        }
    }

    String::from_utf8(buf[..end].to_vec()).ok()
}

/// Remove all extensions from a path stem.
fn remove_all_suffixes(p: &Path) -> PathBuf {
    let mut p = p.to_path_buf();
    while p.extension().is_some() {
        p = p.with_extension("");
    }
    p
}

/// Strip extensions until the `.gin` extension is the current one (i.e. the
/// next strip would remove `.gin`).
fn remove_suffix_until_gin(p: &Path) -> PathBuf {
    let mut p = p.to_path_buf();
    while p.extension().is_some() && p.extension().unwrap() != "gin" {
        p = p.with_extension("");
    }
    p
}

/// Collect all extensions that come *after* `.gin` in a multi-extension name,
/// returned in forward order (e.g. `foo.gin.bar.baz` → `[".bar", ".baz"]`).
fn get_suffixes_after_gin(p: &Path) -> Vec<String> {
    let mut suffixes: Vec<String> = Vec::new();
    let mut p = p.to_path_buf();
    while p.extension().is_some() && p.extension().unwrap() != "gin" {
        suffixes.push(format!(".{}", p.extension().unwrap().to_string_lossy()));
        p = p.with_extension("");
    }
    suffixes.reverse();
    suffixes
}

/// Recursively collect all files under `dir`.
fn walk_dir(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(walk_dir(&path));
            } else {
                out.push(path.canonicalize().unwrap_or(path));
            }
        }
    }
    out
}

/// Returns true if `path` is a valid Windows file path and is not equal to
/// `base_dir`.  This replaces the `pathvalidate` Python dependency.
fn is_valid_windows_path(path: &Path, base_dir: &Path) -> bool {
    if path == base_dir {
        return false;
    }
    let s = path.to_string_lossy();
    // Reject paths with characters that are illegal on Windows.
    const ILLEGAL: &[char] = &['"', '<', '>', '|', '?', '*'];
    !s.chars().any(|c| ILLEGAL.contains(&c))
}

// ─── Section descriptor (parsed from the section table) ──────────────────────

struct SectionInfo {
    name: String,
    offset: u64,
    size_uncompressed: usize,
    size_compressed: usize,
    flags: Flags,
}

// ─── Core decompiler ─────────────────────────────────────────────────────────

/// Parse and decompile a single `.gin` file.
///
/// Returns a list of output file paths (as strings) that were written.
pub fn decompile_file_inner(
    file_path: &Path,
    output_dir: &Path,
    file_count_offset: usize,
    include_number_prefix: bool,
    silent: bool,
) -> PyResult<Vec<PathBuf>> {
    let data = fs::read(file_path).map_err(|e| PyFileNotFoundError::new_err(e.to_string()))?;

    if data.len() < HEADER_SIZE {
        return Err(PyValueError::new_err("File too small to be a .gin file"));
    }

    // --- 1. Validate magic ---
    let magic = read_u32_le(&data, HEADER_MAGIC_OFFSET);
    if magic != GIN_MAGIC_NUMBER {
        return Err(PyValueError::new_err("Not a .gin file (bad magic number)"));
    }

    // --- 2. Read section count ---
    let section_count = read_u32_le(&data, HEADER_SECTION_COUNT_OFFSET) as usize;

    if !silent {
        println!("Found {} sections. Starting extraction...\n", section_count);
    }

    // --- 3. Parse section table ---
    let table_start = HEADER_SIZE;
    if data.len() < table_start + section_count * SECT_ENTRY_SIZE {
        return Err(PyValueError::new_err("File truncated in section table"));
    }

    let mut sections: Vec<SectionInfo> = Vec::with_capacity(section_count);
    for i in 0..section_count {
        let base = table_start + i * SECT_ENTRY_SIZE;
        let name_bytes = &data[base..base + SECT_NAME_SIZE];
        let name = read_cstr(name_bytes);
        let offset = read_u64_le(&data, base + SECT_OFFSET_OFF);
        let size_uncompressed = read_u32_le(&data, base + SECT_SIZE_OFF) as usize;
        let size_compressed = read_u32_le(&data, base + SECT_CSIZE_OFF) as usize;
        let flags_raw = read_u32_le(&data, base + SECT_FLAGS_OFF);
        let flags = Flags::from_bits_truncate(flags_raw);

        sections.push(SectionInfo {
            name,
            offset,
            size_uncompressed,
            size_compressed,
            flags,
        });
    }

    // --- 4. Extract sections ---
    fs::create_dir_all(output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut output_paths: Vec<PathBuf> = Vec::new();

    for (i, section) in sections.iter().enumerate() {
        let read_size = if section.size_compressed > 0 {
            section.size_compressed
        } else {
            section.size_uncompressed
        };

        let start = section.offset as usize;
        let end = start + read_size;

        if end > data.len() {
            if !silent {
                eprintln!("    [!] Section {} data out of bounds, skipping", i);
            }
            continue;
        }

        let raw_data = &data[start..end];

        let final_data = match decompress_data(raw_data, section.flags, section.size_uncompressed) {
            Some(d) => d,
            None => {
                if !silent {
                    eprintln!("    [!] Decompression failed for section {}", i);
                }
                continue;
            }
        };

        let safe_name = safe_filename(&section.name);
        let out_name = if include_number_prefix {
            format!("{:04}_{}", i + file_count_offset, safe_name)
        } else {
            safe_name.clone()
        };

        let out_path = output_dir.join(&out_name);
        fs::write(&out_path, &final_data).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        if !silent {
            let comp_tag = if section.flags.contains(Flags::ZSTD) {
                "[ZSTD]"
            } else if section.flags.contains(Flags::LZ4) {
                "[LZ4]"
            } else {
                "[RAW]"
            };
            println!(
                "Extracted: {} {} ({} bytes)",
                out_name,
                comp_tag,
                final_data.len()
            );
        }

        output_paths.push(out_path);
    }

    Ok(output_paths)
}

// ─── PyO3 wrappers ────────────────────────────────────────────────────────────

#[pyclass]
pub struct GinDecompiler {
    silent: bool,
}

#[pymethods]
impl GinDecompiler {
    #[new]
    #[pyo3(signature = (silent = true))]
    fn new(silent: bool) -> Self {
        GinDecompiler { silent }
    }

    /// Returns `True` if the file at `file_path` has a valid .gin magic number.
    fn check_if_gin_file(&self, file_path: PathBuf) -> PyResult<bool> {
        if !file_path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "File not found: {}",
                file_path.display()
            )));
        }
        let mut f =
            fs::File::open(&file_path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut buf = [0u8; 4];
        if f.read_exact(&mut buf).is_err() {
            return Ok(false);
        }
        Ok(u32::from_le_bytes(buf) == GIN_MAGIC_NUMBER)
    }

    /// Decompile a single `.gin` file into `output_dir`.
    ///
    /// Returns a list of output file path strings.
    #[pyo3(signature = (file_path, output_dir, file_count_offset = 0, include_number_prefix = true))]
    fn decompile_file(
        &self,
        file_path: PathBuf,
        output_dir: PathBuf,
        file_count_offset: usize,
        include_number_prefix: bool,
    ) -> PyResult<Vec<String>> {
        let paths = decompile_file_inner(
            &file_path,
            &output_dir,
            file_count_offset,
            include_number_prefix,
            self.silent,
        )?;
        Ok(paths
            .into_iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect())
    }

    /// Decompile multiple `.gin` files into subdirectories of `output_dir`.
    ///
    /// Returns a list of all output file path strings.
    #[pyo3(signature = (input_paths, output_dir, include_number_prefix = true))]
    fn decompile_multi(
        &self,
        input_paths: Vec<PathBuf>,
        output_dir: PathBuf,
        include_number_prefix: bool,
    ) -> PyResult<Vec<String>> {
        if output_dir.exists() {
            fs::remove_dir_all(output_dir.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        fs::create_dir_all(&output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Validate and filter input paths
        let valid_paths: Vec<PathBuf> = input_paths
            .into_iter()
            .filter_map(|p| {
                if !p.exists() || !p.is_file() {
                    return None;
                }
                match self.check_if_gin_file(p.clone()).ok()? {
                    true => Some(p),
                    false => None,
                }
            })
            .collect();

        if valid_paths.is_empty() {
            return Err(PyValueError::new_err(
                "No .gin files found. Please select at least one .gin file.",
            ));
        }

        let mut all_output_paths: Vec<String> = Vec::new();
        let mut file_count_offset = 0usize;

        for file in &valid_paths {
            let file_output_dir = output_dir.join(file.file_stem().unwrap_or_default());
            fs::create_dir_all(&file_output_dir)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if !self.silent {
                println!("Decompiling \"{}\"...", file.display());
            }

            let paths = decompile_file_inner(
                file,
                &file_output_dir,
                file_count_offset,
                include_number_prefix,
                self.silent,
            )?;

            file_count_offset += paths.len();
            for p in paths {
                all_output_paths.push(p.to_string_lossy().into_owned());
            }
        }

        Ok(all_output_paths)
    }

    /// Full pipeline: decompile all `.gin` files then organise the output into
    /// a human-readable directory structure and emit a `mappings.json`.
    ///
    /// This method parallelises both the path-reading step and the file-copy
    /// step using Rayon, replacing the sequential Python implementation that
    /// was catastrophically slow at ~150 k files.
    #[pyo3(signature = (input_paths, output_dir))]
    fn decompile_to_structure(
        &self,
        input_paths: Vec<PathBuf>,
        output_dir: PathBuf,
    ) -> PyResult<()> {
        if output_dir.exists() {
            fs::remove_dir_all(&output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        fs::create_dir_all(&output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let temp_dir = output_dir.join("decompiled");
        let ship_dir = output_dir.join("ship");
        let mappings_file = output_dir.join("mappings.json");

        fs::create_dir_all(&temp_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        fs::create_dir_all(&ship_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // --- Step 1: Decompile all .gin files into temp_dir ---
        self.decompile_multi(input_paths, temp_dir.clone(), false)?;

        // --- Step 2: Walk temp_dir and collect all files ---
        let decompiled_paths = walk_dir(&temp_dir);
        println!("Structuring {} files...", decompiled_paths.len());

        // --- Step 3: Classify each file in parallel ---
        // Each file is either "skipped" (reloc/alloc/assets metadata) or
        // "placed" (needs a destination path derived from the embedded gin path).
        //
        // We do the expensive read_gin_path_from_binary calls in parallel here.
        #[derive(Debug)]
        enum Classification {
            Skip,
            Place(PathBuf), // resolved destination path
        }

        let ship_dir_ref = &ship_dir;

        let classified: Vec<(PathBuf, Classification)> = decompiled_paths
            .par_iter()
            .map(|path| {
                let ext = path
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();

                let is_metadata_ext = matches!(ext.as_str(), "reloc" | "alloc" | "assets");
                let in_assets_dir = path
                    .parent()
                    .map(|p| p.file_name().unwrap_or_default() == "assets")
                    .unwrap_or(false);

                if is_metadata_ext && !in_assets_dir {
                    return (path.clone(), Classification::Skip);
                }

                let dest: PathBuf = if in_assets_dir {
                    ship_dir_ref
                        .join("decomp_assets")
                        .join(path.file_name().unwrap_or_default())
                } else if matches!(ext.as_str(), "csv" | "otf" | "ttf") {
                    ship_dir_ref
                        .join("fonts")
                        .join(path.file_name().unwrap_or_default())
                } else {
                    // Read the embedded gin path from the file's binary data.
                    match read_gin_path_from_binary(path) {
                        Some(gin_path) => ship_dir_ref.join(&gin_path),
                        None => ship_dir_ref.join(path.file_name().unwrap_or_default()),
                    }
                };

                let dest = dest.canonicalize().unwrap_or_else(|_| dest.clone());

                if is_valid_windows_path(&dest, ship_dir_ref) {
                    (path.clone(), Classification::Place(dest))
                } else {
                    (path.clone(), Classification::Skip)
                }
            })
            .collect();

        // --- Step 4: Copy "placed" files in parallel, collect skipped ---
        // We need the structure_mappings for the skipped-file resolution step
        // below, so we build it while copying.
        let mut structure_mappings: HashMap<PathBuf, PathBuf> = HashMap::new();
        let mut skipped_paths: Vec<PathBuf> = Vec::new();

        // Separate placed from skipped so we can parallelise the copies.
        let (to_place, to_skip): (Vec<_>, Vec<_>) = classified
            .into_iter()
            .partition(|(_, c)| matches!(c, Classification::Place(_)));

        for (path, _) in to_skip {
            skipped_paths.push(path);
        }

        // Create all destination directories up front (must be sequential).
        for (_, classification) in &to_place {
            if let Classification::Place(dest) = classification {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).ok();
                }
            }
        }

        // Parallel copy.
        let copy_results: Vec<(PathBuf, PathBuf, Result<(), _>)> = to_place
            .par_iter()
            .map(|(src, classification)| {
                if let Classification::Place(dest) = classification {
                    let result = fs::copy(src, dest).map(|_| ());
                    (src.clone(), dest.clone(), result)
                } else {
                    unreachable!()
                }
            })
            .collect();

        for (src, dest, result) in copy_results {
            if result.is_ok() {
                structure_mappings.insert(src, dest);
            }
        }

        // --- Step 5: Resolve skipped (reloc/alloc) files ---
        // Each skipped file must be matched to its parent .gin file in
        // structure_mappings so it can be placed next to it in ship_dir.
        for path in &skipped_paths {
            // Special-case for the one known-broken filename.
            if path.file_name().unwrap_or_default()
                == "ST_factory_factory_pearl.ST_factory_turning_stop_pearl_inverted"
            {
                let matched = path
                    .parent()
                    .unwrap()
                    .join("ST_factory_factory_pearl.ST_factory_turning_stop_pearl.gin");
                if let Some(mapped) = structure_mappings.get(&matched) {
                    let dest = mapped.parent().unwrap().join(format!(
                        "{}.ST_factory_turning_stop_pearl_inverted",
                        mapped
                            .with_extension("")
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                    ));
                    fs::copy(path, &dest).ok();
                    structure_mappings.insert(path.clone(), dest);
                }
                continue;
            }

            let no_ext_path = remove_all_suffixes(path);
            let gin_path = {
                let mut p = remove_suffix_until_gin(path);
                // p is now e.g. foo.gin — add the .gin extension explicitly
                if p.extension().unwrap_or_default() != "gin" {
                    p.set_extension("gin");
                }
                p
            };

            let (matched_path, file_suffixes): (PathBuf, Vec<String>) =
                if structure_mappings.contains_key(&gin_path) {
                    let suffixes = {
                        let mut s = vec![".gin".to_string()];
                        s.extend(get_suffixes_after_gin(path));
                        s
                    };
                    (gin_path, suffixes)
                } else if structure_mappings.contains_key(&no_ext_path) {
                    (no_ext_path, get_suffixes_after_gin(path))
                } else {
                    println!("NO MATCH FOUND. {}", path.display());
                    continue;
                };

            let ending_extension = file_suffixes.join("");
            if let Some(mapped) = structure_mappings.get(&matched_path) {
                let dest = mapped.parent().unwrap().join(format!(
                    "{}{}",
                    mapped.file_stem().unwrap().to_string_lossy(),
                    ending_extension
                ));
                fs::copy(path, &dest).ok();
                structure_mappings.insert(path.clone(), dest);
            }
        }

        // --- Step 6: Write mappings.json ---
        let mut mappings_sorted: Vec<(String, String)> = structure_mappings
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    v.to_string_lossy().into_owned(),
                )
            })
            .collect();
        mappings_sorted.sort_by(|a, b| a.0.cmp(&b.0));

        let json_obj: serde_json::Value = serde_json::Value::Object(
            mappings_sorted
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect(),
        );

        fs::write(
            &mappings_file,
            serde_json::to_string_pretty(&json_obj).unwrap(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        Ok(())
    }
}
