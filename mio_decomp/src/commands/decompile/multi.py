import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich import print

from mio_decomp.src.config import config
from mio_decomp.src.libraries.decompiler.decompiler import GinDecompiler

app = typer.Typer()


@app.command()
def decompile_multi(
    output_dir: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            "--output-dir",
            help="The directory to output the decompiled .gin files to. Any files or directories inside the provided directory will be deleted.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ],
    input_paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="The paths to the input .gin files. If omitted, will decompile all of the .gin files in the flamby folder inside of your install of MIO.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    debug: Annotated[
        bool, typer.Option(help="Enables print statements used in debugging.")
    ] = False,
):
    if input_paths is None:
        target_path: Path = config.config_model.game_dir / "flamby"
        target_path: Path = target_path.resolve()
        if not target_path.exists():
            print(
                f'"{target_path.parent}" not found! Please make sure you have MIO: Memories in Orbit installed locally, and that the "game_dir" value in your configuration is pointing to it.'
            )
            raise typer.Abort()
        input_paths: list[Path] = [path for path in target_path.iterdir()]

    final_input_paths: list[Path] = []

    for path in input_paths:
        if path.is_file():
            if path not in final_input_paths:
                final_input_paths.append(path)
        else:
            for p in path.iterdir():
                if p.is_file():
                    if p not in final_input_paths:
                        final_input_paths.append(p)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        contents: list[Path] = list(output_dir.iterdir())
        if len(contents) > 0:
            delete: bool = typer.confirm(
                f"Are you sure you want to delete {len(contents)} files?"
            )
            if not delete:
                raise typer.Abort()
            for file in contents:
                print(f'Deleting "{file}".')
                if file.is_dir():
                    shutil.rmtree(file)
                else:
                    file.unlink()

    print(final_input_paths)
    print(output_dir)

    decompiler: GinDecompiler = GinDecompiler(silent=not debug)
    decompiler.decompile_multi(input_paths=final_input_paths, output_dir=output_dir)
