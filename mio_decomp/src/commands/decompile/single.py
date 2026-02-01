from pathlib import Path
from typing import Annotated

import typer
from rich import print

from mio_decomp.src.libraries.decompiler.decompiler import GinDecompiler

app = typer.Typer()


@app.command()
def decompile(
    target_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the input .gin file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            "--output-dir",
            help="The directory to output the decompiled .gin file to. Any files or directories inside the provided directory will be deleted.",
            file_okay=False,
            dir_okay=True,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
    debug: Annotated[
        bool, typer.Option(help="Enables print statements used in debugging.")
    ] = False,
):
    """Decompile a .gin file."""
    if output_dir is None:
        output_dir: Path = target_file.parent / "extracted_gin"
        output_dir.resolve()

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
                file.unlink()

    compiler: GinDecompiler = GinDecompiler(silent=not debug)
    compiler.decompile_file(file_path=target_file, output_dir=output_dir)
