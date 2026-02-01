# Huge thanks to @mistwreathed for creating the original version of this tool.
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


class GinDecompiler:
    """A decompiler for the .gin files in MIO: Memories in Orbit."""

    def __init__(self, silent: bool = True) -> None:
        self.silent: bool = silent
        if not "win32" == sys.platform:
            print(f"OS '{sys.platform}' is not supported currently.")
            sys.exit(1)

    def __print(self, *args, **kwargs) -> None:
        """Wrapper for print."""
        if not self.silent:
            print(*args, **kwargs)

    def __ensure_dir(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

    def __decompress_data(
        self, data: bytes, flags: int, original_size: int
    ) -> bytes | None:
        """Handles decompression based on section flags.

        Args:
            data (bytes): The data to decompress.
            flags (_type_): _description_
            original_size (_type_): _description_

        Returns:
            bytes | None: The decompressed data. The value is None if decompression fails.
        """
        try:
            if flags & FLAGS.ZSTD:
                # ZSTD Decompression
                dctx: ZstdDecompressor = ZstdDecompressor()
                return dctx.decompress(data, max_output_size=original_size)
            elif flags & FLAGS.LZ4:
                # LZ4 Block Decompression
                return lz4.block.decompress(data, uncompressed_size=original_size)
            else:
                # Raw Data (No Compression)
                return data

        except Exception as e:
            self.__print(f"    [!] Decompression failed: {e}")
            return None

    def check_if_gin_file(self, file_path: Path) -> bool:
        """Checks if a file's magic number matches a .gin's.

        Args:
            file_path (Path): The input file's path.

        Returns:
            bool: True if the file is a .gin file.

        Raises:
            FileNotFoundError: The input file doesn't exist.
        """
        if not file_path.exists():  # File should always exist, but just to make sure
            raise FileNotFoundError

        with file_path.open("rb") as f:
            # Structure: u32 magic
            header_fmt = "<I"

            header_data = f.read(4)

            if (
                len(header_data) < 4
            ):  # Magic number is 4 bytes, so if there isn't that much data, it can't be a .gin file.
                return False

            magic = struct.unpack(header_fmt, header_data)[0]

            return magic == GIN_MAGIC_NUMBER

    def decompile_file(self, file_path: Path, output_dir: Path) -> None:
        """Decompiles a single .gin file.

        Args:
            file_path (Path): The path to the .gin file to decompile.
            output_dir (Path): The directory to output to.
        """
        if not file_path.exists():  # File should always exist, but just to make sure
            print("The selected file doesn't exist.")
            typer.Abort()

        with file_path.open("rb") as f:
            # --- 1. Read Main Header ---
            # Structure: u32 magic, u32 ver, u32 res[2], char id[16], u32 res2, char path[256], u32 count, u64 check[2]
            header_fmt = "<II8s16sI256sI16s"
            header_size = struct.calcsize(header_fmt)

            header_data = f.read(header_size)
            (magic, ver, _, _, _, _, section_count, _) = struct.unpack(
                header_fmt, header_data
            )

            if magic != GIN_MAGIC_NUMBER:
                print("The selected file is not a .gin file!")
                raise typer.Abort()

            self.__print(f"Found {section_count} sections. Starting extraction...\n")

            # --- 2. Read Section Table ---
            # Structure: u8 name[64], u64 offset, u32 size, u32 c_size, u32 flags, u32 params[4], u32 ver, char id[16], u64 check[2]
            sect_fmt = "<64sQIII16sI16s16s"
            sect_size = struct.calcsize(sect_fmt)

            sections: list = []
            for i in range(section_count):
                sect_data: bytes = f.read(sect_size)
                sections.append(struct.unpack(sect_fmt, sect_data))

            # --- 3. Extract Data ---
            for i, s_info in enumerate(sections):
                raw_name: str = (
                    s_info[0].split(b"\0", 1)[0].decode("utf-8", errors="ignore")
                )
                offset: int = s_info[1]
                size_uncompressed: int = s_info[2]
                size_compressed: int = s_info[3]
                flags: int = s_info[4]

                # Determine read size (if compressed, read compressed size, else read full size)
                read_size: int = (
                    size_compressed if (size_compressed > 0) else size_uncompressed
                )

                # Go to data offset
                current_pos: int = f.tell()
                f.seek(offset)
                raw_data: bytes = f.read(read_size)
                f.seek(current_pos)  # Return to table index just in case

                # Decompress
                final_data: bytes | None = self.__decompress_data(
                    raw_data, flags, size_uncompressed
                )

                if final_data:
                    # Construct output filename
                    # We add the index to avoid overwriting if names are duplicate
                    safe_name: str = "".join(
                        [c for c in raw_name if c.isalnum() or c in ("_", "-", ".")]
                    )
                    out_name: str = f"{i:03d}_{safe_name}.bin"
                    out_path: Path = output_dir / out_name

                    with out_path.open("wb") as out_f:
                        out_f.write(final_data)

                    comp_tag: str = (
                        "[ZSTD]"
                        if (flags & FLAGS["ZSTD"])
                        else ("[LZ4]" if (flags & FLAGS["LZ4"]) else "[RAW]")
                    )
                    self.__print(
                        f"Extracted: {out_name} {comp_tag} ({len(final_data)} bytes)"
                    )

    def decompile_multi(self, input_paths: list[Path], output_dir: Path):
        """Decompiles multiple .gin files.

        Args:
            input_paths (list[Path]): A list of all of the paths to decompile.
            output_dir (Path): The directory to output all of the decompiled files to.
        """
        file_paths: list[Path] = []

        for file_path in input_paths:
            if not os.access(file_path, os.R_OK):
                self.__print(
                    f'Unable to read path "{file_path}". Check your permissions! Skipping...'
                )
                continue

            if file_path.is_dir():
                self.__print(f'Path "{file_path}" is a directory. Skipping...')
                continue

            if not self.check_if_gin_file(file_path):
                self.__print(f'Path "{file_path}" is not a .gin file. Skipping...')
                continue

            file_paths.append(file_path)

        for file in file_paths:
            file_output_dir: Path = output_dir / file.stem
            file_output_dir.mkdir(777)
            self.decompile_file(file, file_output_dir)
