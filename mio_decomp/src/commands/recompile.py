from pathlib import Path
from typing import Annotated
import typer
from rich import print
from mio_decomp.src.config import config
from mio_decomp.src.libraries.decompiler.recompiler import GinRecompiler

app = typer.Typer()


@app.command()
def recompile(
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            "--output-dir",
            help="The directory to output the recompiled .gin files to.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
    input_paths: Annotated[
        list[Path] | None,
        typer.Argument(
            help="The paths to the original .gin files to recompile. If omitted, will recompile all of the .gin files in the flamby folder inside of your install of MIO.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    decompiled_dir: Annotated[
        Path | None,
        typer.Option(
            "-d",
            "--decompiled",
            "--decompiled-dir",
            help="The root directory containing the decompiled section folders. Each .gin must have a corresponding subfolder named after its stem. Defaults to ./extracted.",
            file_okay=False,
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
    """Recompiles decompiled .gin folders back into .gin files."""
    if input_paths is None:
        target_path: Path = config.config_model.game_dir / "flamby"
        target_path: Path = target_path.resolve()
        if not target_path.exists():
            print(
                f'"{target_path.parent}" not found! Please make sure you have MIO: Memories in Orbit installed locally, and that the "game_dir" value in your configuration is pointing to it.'
            )
            raise typer.Abort()
        input_paths: list[Path] = [path for path in target_path.iterdir()]

    if decompiled_dir is None:
        decompiled_dir: Path = Path("./extracted")
        decompiled_dir = decompiled_dir.resolve()

    if not decompiled_dir.exists() or not decompiled_dir.is_dir():
        print(
            f'Decompiled directory "{decompiled_dir}" not found! Please make sure you have decompiled your .gin files first, or provide a path with "--decompiled-dir".'
        )
        raise typer.Abort()

    if output_dir is None:
        output_dir: Path = Path("./recompiled")
        output_dir = output_dir.resolve()

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
            overwrite: bool = typer.confirm(
                f"Output directory already contains {len(contents)} file(s)/director(ies). Overwrite?"
            )
            if not overwrite:
                raise typer.Abort()

    recompiler: GinRecompiler = GinRecompiler(silent=not debug)
    recompiler.recompile_multi(
        input_paths=final_input_paths,
        decompiled_dir=decompiled_dir,
        output_dir=output_dir,
    )
    print("Done!")