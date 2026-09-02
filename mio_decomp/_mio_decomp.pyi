from os import PathLike
from pathlib import Path

_AnyPath = str | PathLike[str] | Path

class GinDecompiler:
    def __init__(self, silent: bool = True) -> None: ...
    def check_if_gin_file(self, file_path: _AnyPath) -> bool: ...
    def decompile_file(
        self,
        file_path: _AnyPath,
        output_dir: _AnyPath,
        file_count_offset: int = 0,
        include_number_prefix: bool = True,
    ) -> list[str]: ...
    def decompile_multi(
        self,
        input_paths: list[_AnyPath],
        output_dir: _AnyPath,
        include_number_prefix: bool = True,
    ) -> list[str]: ...
    def decompile_to_structure(
        self,
        input_paths: list[_AnyPath],
        output_dir: _AnyPath,
    ) -> None: ...
