from pathlib import Path
from typing import Annotated

import typer
from rich import print

from ..._mio_decomp import GinDecompiler

app = typer.Typer()


@app.command()
def check(
    target_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
):
    """Checks the magic number of a file to determine if it is a .gin file."""
    decompiler: GinDecompiler = GinDecompiler()
    if decompiler.check_if_gin_file(target_file):
        print(
            f'File [white]"{target_file}"[/white] [bold][green]is[/green][/bold] a .gin file.'
        )
    else:
        print(
            f'File [white]"{target_file}"[/white] [bold][red]is not[/red][/bold] a .gin file.'
        )
