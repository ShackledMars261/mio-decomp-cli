# Huge thanks to @mistwreathed for creating the original decompiler this is based on.
#
import os
import struct
import sys
from pathlib import Path

import lz4.block
import typer
from rich import print
from zstandard import ZstdDecompressor

from .constants import FLAGS, GIN_MAGIC_NUMBER


class GinRecompiler:
    """A recompiler for the .gin files in MIO: Memories in Orbit.

    Reconstructs a .gin file from a folder of decompiled section files,
    preserving the original compressed blobs verbatim and recomputing
    section checksums using MurmurHash3_x64_128.

    Confirmed checksum behaviour (derived by binary analysis):
        - Section checksums: murmur3_x64_128(uncompressed_data, seed=0),
          stored as little-endian u64 pairs (h1, h2).
        - File-level checksum: algorithm / scope not yet fully determined.
          The checksum is preserved verbatim from the original .gin because
          recompilation reuses the original compressed blobs unchanged, so
          the file content — and therefore its checksum — never changes.
          Once the file checksum algorithm is confirmed, this can be updated.
    """

    def __init__(self, silent: bool = True) -> None:
        self.silent: bool = silent
        if not "win32" == sys.platform:
            print(f"OS '{sys.platform}' is not supported currently.")
            sys.exit(1)

    def __print(self, *args, **kwargs) -> None:
        """Wrapper for print."""
        if not self.silent:
            print(*args, **kwargs)

    # -------------------------------------------------------------------------
    # MurmurHash3_x64_128 — pure Python, no external dependency
    # -------------------------------------------------------------------------

    @staticmethod
    def __murmur128(data: bytes, seed: int = 0) -> tuple[int, int]:
        """MurmurHash3_x64_128.

        Returns (h1, h2) as two unsigned 64-bit integers.
        Store as struct.pack('<QQ', h1, h2) to match the .gin byte layout.

        Key correctness detail: k2 is only mixed when tail_size >= 9,
        and k1 only when tail_size >= 1. Mixing unconditionally is a
        common implementation bug that produces wrong results.

        Args:
            data (bytes): The data to hash.
            seed (int): The seed value. Confirmed seed for .gin section
                        checksums is 0.

        Returns:
            tuple[int, int]: (h1, h2)
        """
        def fmix64(k: int) -> int:
            k ^= k >> 33
            k = (k * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
            k ^= k >> 33
            k = (k * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
            k ^= k >> 33
            return k

        def rotl64(x: int, r: int) -> int:
            return ((x << r) | (x >> (64 - r))) & 0xFFFFFFFFFFFFFFFF

        length  = len(data)
        nblocks = length // 16
        h1 = h2 = seed & 0xFFFFFFFFFFFFFFFF
        c1 = 0x87C37B91114253D5
        c2 = 0x4CF5AD432745937F

        for i in range(nblocks):
            k1 = struct.unpack_from("<Q", data, i * 16)[0]
            k2 = struct.unpack_from("<Q", data, i * 16 + 8)[0]
            k1 = (k1 * c1) & 0xFFFFFFFFFFFFFFFF; k1 = rotl64(k1, 31); k1 = (k1 * c2) & 0xFFFFFFFFFFFFFFFF; h1 ^= k1
            h1 = rotl64(h1, 27); h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF; h1 = (h1 * 5 + 0x52DCE729) & 0xFFFFFFFFFFFFFFFF
            k2 = (k2 * c2) & 0xFFFFFFFFFFFFFFFF; k2 = rotl64(k2, 33); k2 = (k2 * c1) & 0xFFFFFFFFFFFFFFFF; h2 ^= k2
            h2 = rotl64(h2, 31); h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF; h2 = (h2 * 5 + 0x38495AB5) & 0xFFFFFFFFFFFFFFFF

        tail = data[nblocks * 16:]
        k1 = k2 = 0
        t = length & 15

        if t >= 15: k2 ^= tail[14] << 48
        if t >= 14: k2 ^= tail[13] << 40
        if t >= 13: k2 ^= tail[12] << 32
        if t >= 12: k2 ^= tail[11] << 24
        if t >= 11: k2 ^= tail[10] << 16
        if t >= 10: k2 ^= tail[9]  << 8
        if t >=  9: k2 ^= tail[8]
        if t >=  9:
            k2 = (k2 * c2) & 0xFFFFFFFFFFFFFFFF; k2 = rotl64(k2, 33); k2 = (k2 * c1) & 0xFFFFFFFFFFFFFFFF; h2 ^= k2

        if t >= 8: k1 ^= tail[7] << 56
        if t >= 7: k1 ^= tail[6] << 48
        if t >= 6: k1 ^= tail[5] << 40
        if t >= 5: k1 ^= tail[4] << 32
        if t >= 4: k1 ^= tail[3] << 24
        if t >= 3: k1 ^= tail[2] << 16
        if t >= 2: k1 ^= tail[1] << 8
        if t >= 1: k1 ^= tail[0]
        if t >= 1:
            k1 = (k1 * c1) & 0xFFFFFFFFFFFFFFFF; k1 = rotl64(k1, 31); k1 = (k1 * c2) & 0xFFFFFFFFFFFFFFFF; h1 ^= k1

        h1 ^= length; h2 ^= length
        h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF; h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF
        h1 = fmix64(h1); h2 = fmix64(h2)
        h1 = (h1 + h2) & 0xFFFFFFFFFFFFFFFF; h2 = (h2 + h1) & 0xFFFFFFFFFFFFFFFF
        return h1, h2

    # -------------------------------------------------------------------------
    # Struct layouts (from header.cpp / sections.cpp / constants.cpp)
    # -------------------------------------------------------------------------

    # u32 magic, u32 ver, u32 res[2], char file_id[16], u32 res2,
    # char file_path[256], u32 section_count, u64 checksum[2]
    _HEADER_FMT  = "<II8s16sI256sIQQ"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)

    # u8 name[64], u64 offset, u32 size, u32 c_size, u32 flags,
    # u32 params[4] (as 16 raw bytes), u32 section_version,
    # char section_id[16], u64 checksum[2] (as 16 raw bytes)
    _SECT_FMT  = "<64sQIII16sI16s16s"
    _SECT_SIZE = struct.calcsize(_SECT_FMT)

    # -------------------------------------------------------------------------
    # .gin parser
    # -------------------------------------------------------------------------

    def __parse_gin(self, file_path: Path) -> dict:
        """Parse a .gin file, returning header metadata, section metadata,
        raw on-disk section blobs, and the original file bytes.

        Args:
            file_path (Path): Path to the .gin file.

        Returns:
            dict: Parsed .gin data.

        Raises:
            FileNotFoundError: The file does not exist.
            AssertionError: The file is not a valid .gin.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_file = file_path.read_bytes()

        with file_path.open("rb") as f:
            raw_header = f.read(self._HEADER_SIZE)
            (magic, ver, res, file_id, res2, file_path_b,
             section_count, chk_lo, chk_hi) = struct.unpack(self._HEADER_FMT, raw_header)

            assert magic == GIN_MAGIC_NUMBER, \
                f"Not a .gin file (magic={magic:#010x}, expected {GIN_MAGIC_NUMBER:#010x})"

            sections: list[dict] = []
            for _ in range(section_count):
                raw = f.read(self._SECT_SIZE)
                (name_b, offset, size, c_size, flags,
                 params_b, sect_ver, sect_id_b, chksum_b) = struct.unpack(self._SECT_FMT, raw)

                s_chk_lo, s_chk_hi = struct.unpack("<QQ", chksum_b)

                sections.append({
                    "name_b":    name_b,
                    "name":      name_b.split(b"\x00", 1)[0].decode("utf-8", errors="replace"),
                    "offset":    offset,
                    "size":      size,        # uncompressed size
                    "c_size":    c_size,      # compressed size (0 if uncompressed)
                    "flags":     flags,
                    "params_b":  params_b,
                    "sect_ver":  sect_ver,
                    "sect_id_b": sect_id_b,
                    "chksum_b":  chksum_b,
                    "chk_lo":    s_chk_lo,
                    "chk_hi":    s_chk_hi,
                })

            # Read each section's raw on-disk bytes (compressed if applicable)
            for s in sections:
                read_size = s["c_size"] if s["c_size"] > 0 else s["size"]
                f.seek(s["offset"])
                s["raw_bytes"] = f.read(read_size)

        return {
            "ver":           ver,
            "res":           res,
            "file_id":       file_id,
            "res2":          res2,
            "file_path_b":   file_path_b,
            "section_count": section_count,
            "chk_lo":        chk_lo,
            "chk_hi":        chk_hi,
            "sections":      sections,
            "raw_file":      raw_file,
        }

    # -------------------------------------------------------------------------
    # Section file matching
    # -------------------------------------------------------------------------

    def __match_section_files(
        self,
        folder: Path,
        sections: list[dict],
    ) -> list[Path]:
        """Match decompiled section files to sections by name.

        The decompiler writes files named either:
          - '{name}'        (include_number_prefix=False)
          - '{NNNN}_{name}' (include_number_prefix=True)

        Leading numeric prefixes are stripped before matching.
        Falls back to sorted order for any sections that cannot be matched.

        Args:
            folder (Path): The decompiled output folder.
            sections (list[dict]): Parsed section metadata from the .gin.

        Returns:
            list[Path]: One matched file path per section, in section order.
        """
        all_files = [
            f for f in folder.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        # Build name -> Path map, stripping any leading 'NNNN_' numeric prefix
        name_map: dict[str, Path] = {}
        for fpath in all_files:
            parts = fpath.name.split("_", 1)
            try:
                int(parts[0])
                bare = parts[1] if len(parts) > 1 else fpath.name
            except ValueError:
                bare = fpath.name
            name_map[bare]      = fpath
            name_map[fpath.name] = fpath  # also index by full name

        matched: list[Path | None] = []
        unmatched: list[int] = []

        for i, s in enumerate(sections):
            if s["name"] in name_map:
                matched.append(name_map[s["name"]])
            else:
                unmatched.append(i)
                matched.append(None)

        if unmatched:
            self.__print(
                f"    [!] Could not match {len(unmatched)} section(s) by name: "
                + ", ".join(repr(sections[i]["name"]) for i in unmatched)
            )
            matched_set  = set(p for p in matched if p is not None)
            leftover     = sorted([f for f in all_files if f not in matched_set], key=lambda p: p.name)
            leftover_iter = iter(leftover)
            for i in unmatched:
                try:
                    matched[i] = next(leftover_iter)
                    self.__print(f"    [!] Fallback: section[{i}] '{sections[i]['name']}' -> {matched[i].name}")
                except StopIteration:
                    print(f"    [!] No fallback file available for section[{i}] '{sections[i]['name']}'")
                    typer.Abort()

        return matched  # type: ignore[return-value]

    # -------------------------------------------------------------------------
    # Section checksum
    # -------------------------------------------------------------------------

    def __compute_section_checksum(self, uncompressed: bytes) -> bytes:
        """Compute the checksum for a single section.

        Confirmed algorithm: MurmurHash3_x64_128(uncompressed_data, seed=0),
        result stored as little-endian (h1, h2).

        Args:
            uncompressed (bytes): The decompressed section data.

        Returns:
            bytes: 16-byte checksum ready to write into the section header.
        """
        h1, h2 = self.__murmur128(uncompressed, seed=0)
        return struct.pack("<QQ", h1, h2)

    # -------------------------------------------------------------------------
    # Core recompile
    # -------------------------------------------------------------------------

    def recompile_file(
        self,
        file_path: Path,
        output_path: Path,
        decompiled_folder: Path,
    ) -> Path:
        """Recompile a single decompiled .gin folder back into a .gin file.

        Strategy: the original compressed blobs are reused verbatim from the
        source .gin. Re-compressing the decompiled files is intentionally
        avoided because zstd/lz4 compression is non-deterministic across
        library versions and settings, which would produce a different
        compressed size and invalidate the original section offsets.
        Only the section header checksums are recomputed; all other metadata
        (offsets, sizes, flags, params, file checksum) is preserved as-is.

        Args:
            file_path (Path): Path to the original .gin to draw metadata from.
            output_path (Path): Destination path for the recompiled .gin.
            decompiled_folder (Path): Folder containing the decompiled section files.

        Returns:
            Path: The output_path on success.

        Raises:
            FileNotFoundError: The source .gin or decompiled folder is missing.
            AssertionError: The source file is not a valid .gin.
        """
        if not file_path.exists():
            print(f"The selected file doesn't exist.")
            typer.Abort()

        if not decompiled_folder.exists() or not decompiled_folder.is_dir():
            print(f"The decompiled folder doesn't exist or is not a directory.")
            typer.Abort()

        self.__print(f'Recompiling "{file_path}"..')

        # --- 1. Parse the original .gin ---
        gin = self.__parse_gin(file_path)
        sections      = gin["sections"]
        section_count = gin["section_count"]
        orig_bytes    = gin["raw_file"]

        # --- 2. Match decompiled section files ---
        section_files = self.__match_section_files(decompiled_folder, sections)

        # --- 3. Build section headers with recomputed checksums ---
        sect_headers = b""
        for s, fpath in zip(sections, section_files):
            uncompressed = fpath.read_bytes()
            chksum_b     = self.__compute_section_checksum(uncompressed)

            sect_headers += struct.pack(
                self._SECT_FMT,
                s["name_b"],
                s["offset"],
                s["size"],
                s["c_size"],
                s["flags"],
                s["params_b"],
                s["sect_ver"],
                s["sect_id_b"],
                chksum_b,
            )

            self.__print(f"    Checksummed: {s['name']!r}")

        # --- 4. Rebuild header (preserve file checksum verbatim) ---
        # The file-level checksum algorithm has not yet been fully determined.
        # Since we reuse the original compressed blobs unchanged, the file
        # content is identical to the original and the checksum remains valid.
        header_final = struct.pack(
            self._HEADER_FMT,
            GIN_MAGIC_NUMBER,
            gin["ver"],
            gin["res"],
            gin["file_id"],
            gin["res2"],
            gin["file_path_b"],
            section_count,
            gin["chk_lo"],
            gin["chk_hi"],
        )

        # --- 5. Assemble output, preserving all original section offsets ---
        # Sections are 16-byte aligned; there are small padding gaps between
        # them in the original file. We preserve the entire original layout by
        # pre-filling with the original bytes and only overwriting the header
        # and section table.
        out = bytearray(orig_bytes)

        out[0 : len(header_final)] = header_final

        table_start = len(header_final)
        out[table_start : table_start + len(sect_headers)] = sect_headers

        # --- 6. Write output ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(out))

        self.__print(f'    Written to "{output_path}"')
        return output_path

    def recompile_multi(
        self,
        input_paths: list[Path],
        decompiled_dir: Path,
        output_dir: Path,
    ) -> list[Path]:
        """Recompile multiple .gin files.

        Each .gin is expected to have a corresponding subfolder inside
        decompiled_dir named after the .gin stem (matching the layout
        produced by GinDecompiler.decompile_multi).

        Args:
            input_paths (list[Path]): The original .gin files to recompile.
            decompiled_dir (Path): Root folder containing one subfolder per .gin.
            output_dir (Path): Destination folder for the recompiled .gin files.

        Returns:
            list[Path]: Paths to the successfully recompiled .gin files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths: list[Path] = []

        for file_path in input_paths:
            if not os.access(file_path, os.R_OK):
                self.__print(
                    f'Unable to read path "{file_path}". Check your permissions! Skipping...'
                )
                continue

            if file_path.is_dir():
                self.__print(f'Path "{file_path}" is a directory. Skipping...')
                continue

            folder = decompiled_dir / file_path.stem
            if not folder.exists() or not folder.is_dir():
                self.__print(
                    f'No decompiled folder found for "{file_path}" '
                    f'(expected "{folder}"). Skipping...'
                )
                continue

            out_path = output_dir / file_path.name

            try:
                self.recompile_file(
                    file_path=file_path,
                    output_path=out_path,
                    decompiled_folder=folder,
                )
                output_paths.append(out_path)
                print(f'Recompiled "{file_path.name}"')
            except Exception as e:
                print(f'    [!] Failed to recompile "{file_path.name}": {e}')

        return output_paths