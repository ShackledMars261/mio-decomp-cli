# Huge thanks to @mistwreathed for creating the original version of this tool.
#
import base64
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import lz4.block
import mmh3
import pathvalidate
import typer
from pydantic import BaseModel, Field, field_serializer, field_validator
from rich import print
from zstandard import ZstdCompressor, ZstdDecompressor

from .constants import (
    GIN_MAGIC_NUMBER,
    GIN_MAIN_HEADER_FORMAT_STRING,
    GIN_MAX_PATH,
    GIN_SECTION_FLAGS,
    GIN_SECTION_HEADER_FORMAT_STRING,
    GIN_SECTION_NAME_SIZE,
    GIN_SECTION_PARAM_COUNT,
)
from .models import u32, u64


class Base64Model(BaseModel):
    @field_serializer("*", when_used="json")
    def serialize_bytes(self, v):
        if isinstance(v, (bytes, bytearray)):
            return base64.b64encode(v).decode("ascii")
        return v

    @field_validator("*", mode="before")
    @classmethod
    def deserialize_bytes(cls, v, info):
        field = cls.model_fields.get(info.field_name)
        if field and field.annotation is bytes and isinstance(v, str):
            return base64.b64decode(v)
        return v


class GinMainHeader(Base64Model):
    magic: u32 = 0
    ver: u32 = 0
    reserved: bytes = Field(b"", max_length=8)
    file_id: bytes = Field(b"", max_length=16)
    reserved_2: u32 = 0
    file_path: bytes = Field(b"", max_length=GIN_MAX_PATH)
    section_count: u32 = 0
    checksum: bytes = Field(b"\x00" * 16, max_length=16)


class GinSectionHeader(Base64Model):
    name: bytes = Field(b"", max_length=GIN_SECTION_NAME_SIZE)
    offset: u64 = 0
    size: u32 = 0
    compressed_size: u32 = 0
    flags: u32 = 0
    params: bytes = Field(b"", max_length=GIN_SECTION_PARAM_COUNT * 4)
    section_version: u32 = 0
    section_id: bytes = Field(b"", max_length=16)
    checksum: bytes = Field(b"\x00" * 16, max_length=16)


class GinHeaderFile(Base64Model):
    main: GinMainHeader = GinMainHeader()
    sections: dict[int, GinSectionHeader] = Field(default_factory=dict)


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
            flags (int): Compression flags.
            original_size (int): The size of the decompressed data.

        Returns:
            bytes | None: The decompressed data. The value is None if decompression fails.
        """
        try:
            if flags & GIN_SECTION_FLAGS.ZSTD:
                # ZSTD Decompression
                dctx: ZstdDecompressor = ZstdDecompressor()
                return dctx.decompress(data, max_output_size=original_size)
            elif flags & GIN_SECTION_FLAGS.LZ4:
                # LZ4 Block Decompression
                return lz4.block.decompress(data, uncompressed_size=original_size)
            else:
                # Raw Data (No Compression)
                return data

        except Exception as e:
            self.__print(f"    [!] Decompression failed: {e}")
            return None

    def __compress_data(self, data: bytes, flags: int) -> bytes:
        """Handles compression based on section flags.

        Args:
            data (bytes): The data to compress.
            flags (int): Compression flags.

        Returns:
            bytes: The compressed data.
        """
        try:
            if flags & GIN_SECTION_FLAGS.ZSTD:
                # ZSTD Compression
                cctx: ZstdCompressor = ZstdCompressor(
                    level=3,
                    write_content_size=False,
                    write_checksum=False,
                    write_dict_id=False,
                )
                return cctx.compress(data)
            elif flags & GIN_SECTION_FLAGS.LZ4:
                # LZ4 Block Compression
                return lz4.block.compress(
                    data,
                    store_size=False,
                )
            else:
                # Raw Data (No Compression)
                return data
        except Exception as e:
            self.__print(f"    [!] Compression failed: {e}")
            return b""

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

    def decompile_file(
        self,
        file_path: Path,
        output_dir: Path,
        header_file_dir: Path,
        file_count_offset: int = 0,
        include_number_prefix: bool = True,
    ) -> list[Path]:
        """Decompiles a single .gin file.

        Args:
            file_path (Path): The path to the .gin file to decompile.
            output_dir (Path): The directory to output to.
            header_file_dir (Path): The directory to output the header file to.
            file_count_offset (int): The offset for the number prefixing the filenames.
            include_number_prefix (bool): Include a unique number at the start of the filenames.

        Returns:
            list[Path]: The output file paths.
        """
        if not file_path.exists():  # File should always exist, but just to make sure
            print("The selected file doesn't exist.")
            typer.Abort()

        output: list[Path] = []

        header_file: GinHeaderFile = GinHeaderFile()

        with file_path.open("rb") as f:
            # --- 1. Read Main Header ---
            # Structure: u32 magic, u32 ver, u32 res[2], char id[16], u32 res2, char path[256], u32 count, u64 check[2]
            header_fmt = GIN_MAIN_HEADER_FORMAT_STRING
            header_size = struct.calcsize(header_fmt)

            header_data = f.read(header_size)
            (
                magic,
                ver,
                res,
                file_id,
                reserved_2,
                header_file_path,
                section_count,
                checksum,
            ) = struct.unpack(header_fmt, header_data)

            if magic != GIN_MAGIC_NUMBER:
                print("The selected file is not a .gin file!")
                raise typer.Abort()

            self.__print(f"Found {section_count} sections. Starting extraction...\n")

            header_file.main = GinMainHeader(
                magic=magic,
                ver=ver,
                reserved=res,
                file_id=file_id,
                reserved_2=reserved_2,
                file_path=header_file_path,
                section_count=section_count,
                checksum=checksum,
            )

            print(f"Header: {header_data}")
            print(f"Header checksum: {checksum}")

            # --- 2. Read Section Table ---
            # Structure: u8 name[64], u64 offset, u32 size, u32 c_size, u32 flags, u32 params[4], u32 ver, char id[16], u64 check[2]
            sect_fmt = GIN_SECTION_HEADER_FORMAT_STRING
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

                header_file.sections[i] = GinSectionHeader(
                    name=s_info[0],
                    offset=s_info[1],
                    size=s_info[2],
                    compressed_size=s_info[3],
                    flags=s_info[4],
                    params=s_info[5],
                    section_version=s_info[6],
                    section_id=s_info[7],
                    checksum=s_info[8],
                )

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

                new_checksum = mmh3.hash_bytes(
                    key=final_data,  # ty:ignore[invalid-argument-type]
                    x64arch=True,
                )
                print(f"Original checksum: {s_info[8]}")
                print(f"Computed checksum: {new_checksum}")

                if final_data:
                    # Construct output filename
                    # We add the index to avoid overwriting if names are duplicate
                    safe_name: str = "".join(
                        [c for c in raw_name if c.isalnum() or c in ("_", "-", ".")]
                    )
                    if include_number_prefix:
                        out_name: str = f"{(i + file_count_offset):04d}_{safe_name}"
                    else:
                        out_name: str = f"{safe_name}"
                    out_path: Path = output_dir / out_name

                    with out_path.open("wb") as out_f:
                        out_f.write(final_data)

                    output.append(output_dir / out_name)

                    comp_tag: str = (
                        "[ZSTD]"
                        if (flags & GIN_SECTION_FLAGS["ZSTD"])
                        else (
                            "[LZ4]" if (flags & GIN_SECTION_FLAGS["LZ4"]) else "[RAW]"
                        )
                    )
                    self.__print(
                        f"Extracted: {out_name} {comp_tag} ({len(final_data)} bytes)"
                    )

        header_file_path: Path = header_file_dir / f"{file_path.name}.json"
        header_file_path: Path = header_file_path.resolve()

        self.__print(f'Writing headers to "{str(header_file_path)}"..')
        header_file_path.write_text(header_file.model_dump_json())

        return output

    def decompile_multi(
        self,
        input_paths: list[Path],
        output_dir: Path,
        header_file_dir: Path,
        include_number_prefix: bool = True,
    ) -> list[Path]:
        """Decompiles multiple .gin files.

        Args:
            input_paths (list[Path]): A list of all of the paths to decompile.
            output_dir (Path): The directory to output all of the decompiled files to.
            header_file_dir (Path): The directory to output the header files to.
            include_number_prefix (bool): Include a unique number at the start of the filenames.

        Returns:
            list[Path]: The output file paths.
        """
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)

        file_paths: list[Path] = []
        current_file_count_offset: int = 0

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

        output_paths: list[Path] = []

        if len(file_paths) == 0:
            print("No .gin files found. Please select at least one .gin file.")
            typer.Abort()

        for file in file_paths:
            file_output_dir: Path = output_dir / file.stem
            file_output_dir.mkdir(777)
            print(f'Decompiling "{file}"..')
            paths: list[Path] = self.decompile_file(
                file_path=file,
                output_dir=file_output_dir,
                header_file_dir=header_file_dir,
                file_count_offset=current_file_count_offset,
                include_number_prefix=include_number_prefix,
            )
            for p in paths:
                output_paths.append(p)
            current_file_count_offset += len(paths)

        return output_paths

    def __walk_dir(self, dir: Path) -> list[Path]:
        output: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                path: Path = Path(dirpath) / filename
                path: Path = path.resolve()
                output.append(path)

        return output

    def __read_until_zero_byte(self, input_file: Path, start_offset: int = 0) -> bytes:
        output: bytes = b""
        with input_file.open("rb") as f:
            f.seek(start_offset)
            while True:
                new_byte = f.read(1)
                if new_byte == b"\x00":
                    break
                output += new_byte

        return output

    def __read_gin_file_path_from_binary(
        self, input_file: Path, start_offset: int = 0
    ) -> bytes:
        output: bytes = b""
        with input_file.open("rb") as f:
            f.seek(start_offset)
            while True:
                new_byte = f.read(1)
                if new_byte == b"\x00" or output.decode("utf-8").endswith(".gin"):
                    break
                output += new_byte

        return output

    def __remove_all_suffixes(self, p: Path) -> Path:
        """Removes all file extensions from a pathlib.Path object."""
        while p.suffix:
            p = p.with_suffix("")
        return p

    def __remove_suffix_until_gin(self, p: Path) -> Path:
        while p.suffix and not p.suffix == ".gin":
            p = p.with_suffix("")
        return p

    def __get_suffixes_after_gin(self, p: Path) -> list[str]:
        suffixes: list[str] = []
        while p.suffix and not p.suffix == ".gin":
            suffixes.append(p.suffix)
            p = p.with_suffix("")
        return list(reversed(suffixes))

    def decompile_to_structure(
        self,
        input_paths: list[Path],
        output_dir: Path,
    ) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        temp_dir: Path = output_dir / "decompiled"
        temp_dir: Path = temp_dir.resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)

        ship_dir: Path = output_dir / "ship"
        ship_dir: Path = ship_dir.resolve()
        ship_dir.mkdir(parents=True, exist_ok=True)

        header_file_dir: Path = output_dir / "header_files"
        header_file_dir: Path = header_file_dir.resolve()
        header_file_dir.mkdir(parents=True, exist_ok=True)

        mappings_file: Path = output_dir / "mappings.json"
        mappings_file: Path = mappings_file.resolve()
        if mappings_file.exists():
            mappings_file.unlink()
        mappings_file.touch()

        self.decompile_multi(
            input_paths,
            output_dir=temp_dir,
            header_file_dir=header_file_dir,
            include_number_prefix=False,
        )

        decompiled_paths: list[Path] = self.__walk_dir(temp_dir)

        print("Structuring files..")

        skipped_paths: list[Path] = []
        structure_mappings: dict[Path, Path] = {}

        for path in decompiled_paths:
            if (
                path.suffix[1:] in ["reloc", "alloc", "assets"]
                and not path.parent.name == "assets"
            ):
                skipped_paths.append(path)
            else:
                try:
                    if path.parent.name == "assets":
                        new_path: Path = ship_dir / "decomp_assets" / path.name
                        new_path: Path = new_path.resolve()
                    elif path.suffix[1:] in [
                        "csv",
                        "otf",
                        "ttf",
                    ]:  # filter out font files
                        new_path: Path = ship_dir / "fonts" / path.name
                        new_path: Path = new_path.resolve()
                    else:
                        new_path: Path = (
                            ship_dir
                            / self.__read_gin_file_path_from_binary(path, 20).decode(
                                "utf-8"
                            )
                        )
                        new_path: Path = new_path.resolve()
                except UnicodeDecodeError as _:
                    new_path: Path = ship_dir / path.name
                    new_path: Path = new_path.resolve()

                if pathvalidate.is_valid_filepath(
                    str(new_path), platform="windows"
                ) and not str(new_path) == str(ship_dir):
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, new_path)
                    structure_mappings[path] = new_path
                else:
                    skipped_paths.append(path)

        for path in skipped_paths:
            no_ext_path: Path = (
                path.parent / f"{self.__remove_suffix_until_gin(path).resolve().stem}"
            )
            no_ext_path: Path = no_ext_path.resolve()

            gin_path: Path = (
                path.parent
                / f"{self.__remove_suffix_until_gin(path).resolve().stem}.gin"
            )
            gin_path: Path = gin_path.resolve()

            if gin_path in structure_mappings.keys():
                # print(f"GIN FOUND! {gin_path}")
                matched_path: Path = gin_path.resolve()
                file_suffixes: list[str] = [
                    ".gin",
                    *self.__get_suffixes_after_gin(path),
                ]
            elif no_ext_path in structure_mappings.keys():
                # print(f"NO_EXT FOUND! {no_ext_path}")
                matched_path: Path = no_ext_path.resolve()
                file_suffixes: list[str] = self.__get_suffixes_after_gin(path)
            elif (
                path.name
                == "ST_factory_factory_pearl.ST_factory_turning_stop_pearl_inverted"  # this specific file breaks stuff, so we make an exception for it here
            ):
                # print(f"INVERTED PEARL BYPASS HIT!\n\tpath.name: {path.name}")
                matched_path: Path = (
                    path.parent
                    / "ST_factory_factory_pearl.ST_factory_turning_stop_pearl.gin"
                )
                ending_extension: str = "".join(
                    [".ST_factory_turning_stop_pearl_inverted"]
                )
                dest_path: Path = (
                    structure_mappings[matched_path].resolve().parent
                    / f"{structure_mappings[matched_path].resolve().with_suffix('').stem}{ending_extension}"
                )
                dest_path: Path = dest_path.resolve()
                shutil.copy(path, dest_path)
                structure_mappings[path] = dest_path
                continue
            else:
                print(f"NO MATCH FOUND. {path}")
                print(f"    UNMATCHED NO_EXT: {no_ext_path}")
                print(f"    path.name: {path.name}")
                continue

            ending_extension: str = "".join(file_suffixes)

            dest_path: Path = (
                structure_mappings[matched_path].resolve().parent
                / f"{structure_mappings[matched_path].resolve().stem}{ending_extension}"
            )
            dest_path: Path = dest_path.resolve()
            shutil.copy(path, dest_path)
            structure_mappings[path] = dest_path

        structure_mappings_str: dict[str, str] = {
            str(path): str(dest_path) for path, dest_path in structure_mappings.items()
        }
        structure_mappings_final: dict[str, str] = {
            key: value for key, value in sorted(structure_mappings_str.items())
        }  # sort alphabetically to group .gin files with their related files

        with mappings_file.open("w") as f:
            json.dump(structure_mappings_final, f, indent=4)

    def __overwrite_bytes(
        self, original_bytes: bytes, new_bytes: bytes, starting_index: int = 0
    ) -> bytes:
        """Overwrite bytes in the middle of a bytes object.

        Args:
            original_bytes (bytes): The bytes that contain what you want to overwrite.
            new_bytes (bytes): The bytes you want to set.
            starting_index (int, optional): Where to start overwriting at. Defaults to 0.

        Returns:
            bytes: The bytes with the overwritten bytes.
        """
        mutable_data: bytearray = bytearray(original_bytes)
        mutable_data[starting_index : starting_index + len(new_bytes)] = new_bytes

        return bytes(mutable_data)

    def compile_single(
        self,
        header_file: Path,
        mappings_file: Path,
        output_file: Path,
    ) -> None:
        header_model: GinHeaderFile = GinHeaderFile.model_validate_json(
            header_file.read_text()
        )
        with mappings_file.open("r") as f:
            structure_mappings: dict[Path, Path] = {
                Path(key).resolve(): Path(value).resolve()
                for key, value in json.load(f).items()
            }

        structure_mappings_by_name: dict[str, Path] = {
            key.name: value for key, value in structure_mappings.items()
        }

        new_header: GinHeaderFile = GinHeaderFile()
        new_header.main = GinMainHeader(
            magic=header_model.main.magic,
            ver=header_model.main.ver,
            reserved=header_model.main.reserved,
            file_id=header_model.main.file_id,
            reserved_2=header_model.main.reserved_2,
            file_path=header_model.main.file_path,
            section_count=header_model.main.section_count,
            checksum=b"\x00" * 16,
        )

        print(new_header.main)

        sections_data: bytes = b""
        header_length: int = struct.calcsize(GIN_MAIN_HEADER_FORMAT_STRING)
        sections_offset: int = (
            0
            + header_length
            + (
                new_header.main.section_count
                * struct.calcsize(GIN_SECTION_HEADER_FORMAT_STRING)
            )
        )

        for i, section in header_model.sections.items():
            old_sect_header: GinSectionHeader = header_model.sections[i]

            sect_header: GinSectionHeader = GinSectionHeader(
                name=old_sect_header.name,
                offset=sections_offset,
                size=0,
                compressed_size=0,
                flags=old_sect_header.flags,
                params=old_sect_header.params,
                section_version=old_sect_header.section_version,
                section_id=old_sect_header.section_id,
                checksum=b"\x00" * 16,
            )

            target_path: Path = structure_mappings_by_name[
                old_sect_header.name.decode("ascii").rstrip("\x00")
            ]
            sect_data: bytes = target_path.read_bytes()
            sect_header.size: int = len(sect_data)

            compressed_sect_data: bytes = self.__compress_data(
                sect_data, sect_header.flags
            )
            sect_header.compressed_size: int = len(compressed_sect_data)

            sect_header.checksum = mmh3.hash_bytes(sect_data, x64arch=True)

            print(f"Computed section checksum: {sect_header.checksum}")

            new_header.sections[i] = sect_header
            sections_data += compressed_sect_data
            sections_offset += len(compressed_sect_data)

        # final_data: bytes = struct.pack(
        # GIN_MAIN_HEADER_FORMAT_STRING,
        # new_header.main.magic,
        # new_header.main.ver,
        # new_header.main.reserved,
        # new_header.main.file_id,
        # new_header.main.reserved_2,
        # new_header.main.file_path,
        # new_header.main.section_count,
        # new_header.main.checksum,
        # )
        final_data: bytes = b""
        final_data += new_header.main.magic.to_bytes(length=4, byteorder="little")
        final_data += new_header.main.ver.to_bytes(length=2, byteorder="little")
        final_data += new_header.main.reserved
        final_data += new_header.main.file_id
        final_data += new_header.main.reserved_2.to_bytes(length=4, byteorder="little")
        final_data += new_header.main.file_path
        final_data += new_header.main.section_count.to_bytes(
            length=4, byteorder="little"
        )
        final_data += new_header.main.checksum

        print(f"Final data: {final_data}")
        print(f"Length of final data: {len(final_data)}")

        sect_headers_data: bytes = b""
        for i, sect_header in new_header.sections.items():
            sect_headers_data += struct.pack(
                GIN_SECTION_HEADER_FORMAT_STRING,
                sect_header.name,
                sect_header.offset,
                sect_header.size,
                sect_header.compressed_size,
                sect_header.flags,
                sect_header.params,
                sect_header.section_version,
                sect_header.section_id,
                sect_header.checksum,
            )
        final_data += sect_headers_data

        final_checksum: bytes = mmh3.hash_bytes(
            key=final_data,
            x64arch=True,
        )  # take checksum of main header and section headers

        print(f"Computed file checksum: {final_checksum}")

        final_data += sections_data

        new_header.main.checksum: bytes = final_checksum
        final_data_with_checksum: bytes = self.__overwrite_bytes(
            final_data, final_checksum, 294
        )

        print(final_data_with_checksum[:310])

        output_file.write_bytes(final_data_with_checksum)
