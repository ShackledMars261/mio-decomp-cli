use std::collections::HashSet;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;

use memmap2::Mmap;
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

/// Extract the embedded `.gin` path from an already-mapped byte slice.
///
/// Called while the file is already mmap'd in `decompile_file_inner`, so no
/// second file open is needed.  Also used standalone during classification for
/// files whose path was not captured at decompile time.
fn extract_gin_path_from_slice(data: &[u8]) -> Option<String> {
    if data.len() <= GIN_PATH_OFFSET {
        return None;
    }
    let buf = &data[GIN_PATH_OFFSET..];
    let mut end = buf.len().min(512);
    for i in 0..end {
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

/// Strip the Windows extended-length path prefix (`\\?\`) that
/// `canonicalize()` adds on Windows, so paths written to mappings.json are
/// clean `C:\...` paths.
fn strip_unc_prefix(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        p
    }
}

/// Remove all extensions from a path stem.
fn remove_all_suffixes(p: &Path) -> PathBuf {
    let mut p = p.to_path_buf();
    while p.extension().is_some() {
        p = p.with_extension("");
    }
    p
}

/// Strip extensions until `.gin` is the current extension.
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

/// Parallel recursive directory walk using `jwalk`.
fn walk_dir(dir: &Path) -> Vec<PathBuf> {
    jwalk::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| strip_unc_prefix(e.path()))
        .collect()
}

/// Returns true if `path` is a valid Windows file path and is not equal to
/// `base_dir`.  Replaces the `pathvalidate` Python dependency.
fn is_valid_windows_path(path: &Path, base_dir: &Path) -> bool {
    if path == base_dir {
        return false;
    }
    let s = path.to_string_lossy();
    const ILLEGAL: &[char] = &['"', '<', '>', '|', '?', '*'];
    !s.chars().any(|c| ILLEGAL.contains(&c))
}

/// Build a destination path from a ship directory and a raw gin path string,
/// without calling `canonicalize()`.
///
/// `canonicalize()` makes a syscall per file to verify the path exists on
/// disk, which is 150 k unnecessary syscalls.  Since we construct paths from
/// known-absolute base directories we can just join and normalise manually.
fn build_dest_path(base: &Path, relative: &str) -> PathBuf {
    // Normalise any forward-slash separators the embedded path may contain.
    let normalised = relative.replace('/', std::path::MAIN_SEPARATOR_STR);
    base.join(normalised)
}

// ─── Section descriptor ───────────────────────────────────────────────────────

struct SectionInfo {
    name: String,
    offset: u64,
    size_uncompressed: usize,
    size_compressed: usize,
    flags: Flags,
}

// ─── Write task sent from worker threads to the I/O thread ───────────────────

struct WriteTask {
    path: PathBuf,
    data: Vec<u8>,
}

// ─── Core decompiler ─────────────────────────────────────────────────────────

/// Parse and decompile a single `.gin` file.
///
/// Optimisations applied here:
/// - `memmap2`: file is memory-mapped rather than heap-allocated, letting the
///   OS page data in on demand and reducing peak RSS when many files are
///   decompiled in parallel.
/// - Parallel section extraction: decompression (CPU-bound) runs on Rayon
///   worker threads; completed `WriteTask`s are sent to a dedicated I/O
///   thread via a channel, decoupling decompression from write latency.
/// - The embedded gin path is read from the mmap'd data directly, so the
///   classification step in `decompile_to_structure` does not need to
///   re-open the file.
///
/// Returns `(output_paths, gin_path_map)` where `gin_path_map` maps each
/// output `PathBuf` to the embedded gin path string found inside that section
/// (if any).  This is consumed by `decompile_to_structure` to avoid a second
/// file open per output file.
pub fn decompile_file_inner(
    file_path: &Path,
    output_dir: &Path,
    file_count_offset: usize,
    include_number_prefix: bool,
    silent: bool,
) -> PyResult<(Vec<PathBuf>, Vec<Option<String>>)> {
    // --- Memory-map the input file ---
    let file =
        fs::File::open(file_path).map_err(|e| PyFileNotFoundError::new_err(e.to_string()))?;

    // SAFETY: we treat the mapping as read-only and do not truncate the file.
    let data = unsafe { Mmap::map(&file) }.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

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

    let sections: Vec<SectionInfo> = (0..section_count)
        .map(|i| {
            let base = table_start + i * SECT_ENTRY_SIZE;
            SectionInfo {
                name: read_cstr(&data[base..base + SECT_NAME_SIZE]),
                offset: read_u64_le(&data, base + SECT_OFFSET_OFF),
                size_uncompressed: read_u32_le(&data, base + SECT_SIZE_OFF) as usize,
                size_compressed: read_u32_le(&data, base + SECT_CSIZE_OFF) as usize,
                flags: Flags::from_bits_truncate(read_u32_le(&data, base + SECT_FLAGS_OFF)),
            }
        })
        .collect();

    // --- 4. Extract sections (parallel decompress + pipelined I/O) ---
    fs::create_dir_all(output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Spawn a dedicated I/O thread that drains the channel and writes files.
    // This decouples CPU-bound decompression from disk write latency: Rayon
    // workers never block waiting for a write to complete.
    let (tx, rx) = mpsc::channel::<WriteTask>();
    let io_thread = thread::spawn(move || {
        let mut write_errors: Vec<String> = Vec::new();
        for task in rx {
            if let Err(e) = fs::write(&task.path, &task.data) {
                write_errors.push(format!("{}: {}", task.path.display(), e));
            }
        }
        write_errors
    });

    // Decompress all sections in parallel. Each section produces either a
    // WriteTask (sent to the I/O thread) or nothing (on error / OOB).
    // We collect (index, out_path, gin_path) for every successfully queued
    // section so we can return a stable ordered list.
    let section_results: Vec<Option<(usize, PathBuf, Option<String>)>> = sections
        .par_iter()
        .enumerate()
        .map(|(i, section)| {
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
                return None;
            }

            let raw_data = &data[start..end];

            let final_data =
                match decompress_data(raw_data, section.flags, section.size_uncompressed) {
                    Some(d) => d,
                    None => {
                        if !silent {
                            eprintln!("    [!] Decompression failed for section {}", i);
                        }
                        return None;
                    }
                };

            // Extract the embedded gin path from the decompressed data while
            // we already have it in memory, avoiding a second file open in
            // the structuring step.
            let gin_path = extract_gin_path_from_slice(&final_data);

            let safe_name = safe_filename(&section.name);
            let out_name = if include_number_prefix {
                format!("{:04}_{}", i + file_count_offset, safe_name)
            } else {
                safe_name
            };

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

            let out_path = output_dir.join(&out_name);
            Some((i, out_path, gin_path, final_data))
        })
        // Send write tasks to the I/O thread and strip the data field.
        .map(|opt| {
            opt.map(|(i, out_path, gin_path, final_data)| {
                tx.send(WriteTask {
                    path: out_path.clone(),
                    data: final_data,
                })
                .ok(); // I/O thread dropped = channel closed; ignore
                (i, out_path, gin_path)
            })
        })
        .collect();

    // Drop the sender so the I/O thread's recv loop terminates.
    drop(tx);

    // Wait for all writes to finish and surface any errors.
    let write_errors = io_thread.join().unwrap_or_default();
    if !write_errors.is_empty() {
        return Err(PyRuntimeError::new_err(write_errors.join("\n")));
    }

    // Collect results in section order (parallel iter may reorder).
    let mut indexed: Vec<(usize, PathBuf, Option<String>)> =
        section_results.into_iter().flatten().collect();
    indexed.sort_unstable_by_key(|(i, _, _)| *i);

    let output_paths: Vec<PathBuf> = indexed.iter().map(|(_, p, _)| p.clone()).collect();
    let gin_paths: Vec<Option<String>> = indexed.into_iter().map(|(_, _, g)| g).collect();

    Ok((output_paths, gin_paths))
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

    #[pyo3(signature = (file_path, output_dir, file_count_offset = 0, include_number_prefix = true))]
    fn decompile_file(
        &self,
        file_path: PathBuf,
        output_dir: PathBuf,
        file_count_offset: usize,
        include_number_prefix: bool,
    ) -> PyResult<Vec<String>> {
        let (paths, _) = decompile_file_inner(
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

    #[pyo3(signature = (input_paths, output_dir, include_number_prefix = true))]
    fn decompile_multi(
        &self,
        input_paths: Vec<PathBuf>,
        output_dir: PathBuf,
        include_number_prefix: bool,
    ) -> PyResult<Vec<String>> {
        if output_dir.exists() {
            fs::remove_dir_all(&output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        fs::create_dir_all(&output_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let mut valid_paths: Vec<PathBuf> = input_paths
            .into_iter()
            .filter(|p| {
                if !p.exists() || !p.is_file() {
                    if !self.silent {
                        eprintln!("Skipping \"{}\": not a readable file", p.display());
                    }
                    return false;
                }
                let mut buf = [0u8; 4];
                let ok = fs::File::open(p)
                    .and_then(|mut f| f.read_exact(&mut buf))
                    .map(|_| u32::from_le_bytes(buf) == GIN_MAGIC_NUMBER)
                    .unwrap_or(false);
                if !ok && !self.silent {
                    eprintln!("Skipping \"{}\": not a .gin file", p.display());
                }
                ok
            })
            .collect();

        if valid_paths.is_empty() {
            return Err(PyValueError::new_err(
                "No .gin files found. Please select at least one .gin file.",
            ));
        }

        valid_paths.sort();

        for file in &valid_paths {
            let dir = output_dir.join(file.file_stem().unwrap_or_default());
            fs::create_dir_all(&dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        let silent = self.silent;

        if include_number_prefix {
            // Sequential: offsets must be stable.
            let mut offset = 0usize;
            let mut all: Vec<String> = Vec::new();
            for file in &valid_paths {
                let dir = output_dir.join(file.file_stem().unwrap_or_default());
                if !silent {
                    println!("Decompiling \"{}\"...", file.display());
                }
                let (paths, _) = decompile_file_inner(file, &dir, offset, true, silent)?;
                offset += paths.len();
                all.extend(paths.into_iter().map(|p| p.to_string_lossy().into_owned()));
            }
            Ok(all)
        } else {
            // Fully parallel.
            let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());

            let results: Vec<Vec<String>> = valid_paths
                .par_iter()
                .map(|file| {
                    let dir = output_dir.join(file.file_stem().unwrap_or_default());
                    if !silent {
                        println!("Decompiling \"{}\"...", file.display());
                    }
                    match decompile_file_inner(file, &dir, 0, false, silent) {
                        Ok((paths, _)) => paths
                            .into_iter()
                            .map(|p| p.to_string_lossy().into_owned())
                            .collect(),
                        Err(e) => {
                            errors.lock().unwrap().push(e.to_string());
                            vec![]
                        }
                    }
                })
                .collect();

            let errs = errors.into_inner().unwrap();
            if !errs.is_empty() {
                return Err(PyRuntimeError::new_err(errs.join("\n")));
            }

            Ok(results.into_iter().flatten().collect())
        }
    }

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

        // --- Step 1: Parallel decompile, capturing embedded gin paths ---
        // We run decompile_multi manually here so we can harvest the gin path
        // map from decompile_file_inner without a second file open.
        let mut valid_paths: Vec<PathBuf> = input_paths
            .into_iter()
            .filter(|p| {
                if !p.exists() || !p.is_file() {
                    return false;
                }
                let mut buf = [0u8; 4];
                fs::File::open(p)
                    .and_then(|mut f| f.read_exact(&mut buf))
                    .map(|_| u32::from_le_bytes(buf) == GIN_MAGIC_NUMBER)
                    .unwrap_or(false)
            })
            .collect();

        if valid_paths.is_empty() {
            return Err(PyValueError::new_err(
                "No .gin files found. Please select at least one .gin file.",
            ));
        }

        valid_paths.sort();

        for file in &valid_paths {
            fs::create_dir_all(temp_dir.join(file.file_stem().unwrap_or_default())).ok();
        }

        let silent = self.silent;

        // Collect (output_path, embedded_gin_path) for every extracted section.
        // This map is used in the classification step below so we never need to
        // re-open any output file.
        let decompile_results: Vec<(PathBuf, Option<String>)> = valid_paths
            .par_iter()
            .flat_map(|file| {
                let dir = temp_dir.join(file.file_stem().unwrap_or_default());
                match decompile_file_inner(file, &dir, 0, false, silent) {
                    Ok((paths, gin_paths)) => paths
                        .into_iter()
                        .zip(gin_paths.into_iter())
                        .collect::<Vec<_>>(),
                    Err(_) => vec![],
                }
            })
            .collect();

        println!("Structuring {} files...", decompile_results.len());

        // --- Step 2: Classify each file in parallel ---
        // Using the pre-captured gin paths avoids re-opening every file.
        //
        // dest paths are built with `build_dest_path` instead of
        // `canonicalize()`, saving one syscall per file.
        #[derive(Debug)]
        enum Classification {
            Skip,
            Place(PathBuf),
        }

        let ship_dir_ref = &ship_dir;

        let classified: Vec<(PathBuf, Classification)> = decompile_results
            .par_iter()
            .map(|(path, embedded_gin_path)| {
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
                    match embedded_gin_path {
                        Some(gin_path) => build_dest_path(ship_dir_ref, gin_path),
                        None => ship_dir_ref.join(path.file_name().unwrap_or_default()),
                    }
                };

                if is_valid_windows_path(&dest, ship_dir_ref) {
                    (path.clone(), Classification::Place(dest))
                } else {
                    (path.clone(), Classification::Skip)
                }
            })
            .collect();

        // --- Step 3: Partition, deduplicate dirs, parallel copy ---
        let mut skipped_paths: Vec<PathBuf> = Vec::new();

        let (to_place, to_skip): (Vec<_>, Vec<_>) = classified
            .into_iter()
            .partition(|(_, c)| matches!(c, Classification::Place(_)));

        for (path, _) in to_skip {
            skipped_paths.push(path);
        }

        // Deduplicate parent directories before creating them.
        let unique_dirs: HashSet<PathBuf> = to_place
            .iter()
            .filter_map(|(_, c)| {
                if let Classification::Place(dest) = c {
                    dest.parent().map(PathBuf::from)
                } else {
                    None
                }
            })
            .collect();

        for dir in &unique_dirs {
            fs::create_dir_all(dir).ok();
        }

        // Parallel copy — collect (src, dest, ok) triples.
        let copy_results: Vec<(PathBuf, PathBuf, bool)> = to_place
            .par_iter()
            .map(|(src, classification)| {
                if let Classification::Place(dest) = classification {
                    let ok = fs::copy(src, dest).is_ok();
                    (src.clone(), dest.clone(), ok)
                } else {
                    unreachable!()
                }
            })
            .collect();

        // Build structure_mappings as a sorted Vec for cache-friendly binary
        // search in Step 4, rather than a HashMap.
        let mut structure_mappings: Vec<(PathBuf, PathBuf)> = copy_results
            .into_iter()
            .filter_map(|(src, dest, ok)| if ok { Some((src, dest)) } else { None })
            .collect();
        structure_mappings.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        // Binary search helper.
        let find_mapping = |key: &Path| -> Option<&PathBuf> {
            structure_mappings
                .binary_search_by(|(k, _)| k.as_path().cmp(key))
                .ok()
                .map(|idx| &structure_mappings[idx].1)
        };

        // --- Step 4: Resolve skipped (reloc/alloc) files in parallel ---
        // Each skipped file is matched against the sorted mappings via binary
        // search, then copied to its final destination.
        let skip_results: Vec<Option<(PathBuf, PathBuf)>> = skipped_paths
            .par_iter()
            .map(|path| {
                // Special-case for the one known-broken filename.
                if path.file_name().unwrap_or_default()
                    == "ST_factory_factory_pearl.ST_factory_turning_stop_pearl_inverted"
                {
                    let matched = path
                        .parent()
                        .unwrap()
                        .join("ST_factory_factory_pearl.ST_factory_turning_stop_pearl.gin");
                    if let Some(mapped) = find_mapping(&matched) {
                        let dest = mapped.parent().unwrap().join(format!(
                            "{}.ST_factory_turning_stop_pearl_inverted",
                            mapped
                                .with_extension("")
                                .file_name()
                                .unwrap()
                                .to_string_lossy()
                        ));
                        fs::copy(path, &dest).ok();
                        return Some((path.clone(), dest));
                    }
                    return None;
                }

                let no_ext_path = remove_all_suffixes(path);
                let gin_path = {
                    let mut p = remove_suffix_until_gin(path);
                    if p.extension().unwrap_or_default() != "gin" {
                        p.set_extension("gin");
                    }
                    p
                };

                let (matched_path, file_suffixes): (PathBuf, Vec<String>) =
                    if find_mapping(&gin_path).is_some() {
                        let mut s = vec![".gin".to_string()];
                        s.extend(get_suffixes_after_gin(path));
                        (gin_path, s)
                    } else if find_mapping(&no_ext_path).is_some() {
                        (no_ext_path, get_suffixes_after_gin(path))
                    } else {
                        println!("NO MATCH FOUND. {}", path.display());
                        return None;
                    };

                if let Some(mapped) = find_mapping(&matched_path) {
                    let dest = mapped.parent().unwrap().join(format!(
                        "{}{}",
                        mapped.file_stem().unwrap().to_string_lossy(),
                        file_suffixes.join("")
                    ));
                    fs::copy(path, &dest).ok();
                    Some((path.clone(), dest))
                } else {
                    None
                }
            })
            .collect();

        // Merge skip results into structure_mappings and re-sort.
        for entry in skip_results.into_iter().flatten() {
            structure_mappings.push(entry);
        }
        structure_mappings.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));

        // --- Step 5: Write mappings.json ---
        // structure_mappings is already sorted, so we just iterate it.
        let json_obj = serde_json::Value::Object(
            structure_mappings
                .into_iter()
                .map(|(k, v)| {
                    (
                        strip_unc_prefix(k).to_string_lossy().into_owned(),
                        serde_json::Value::String(
                            strip_unc_prefix(v).to_string_lossy().into_owned(),
                        ),
                    )
                })
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
