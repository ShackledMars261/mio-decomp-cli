from pathlib import Path
from typing import Annotated

import typer
from rich import print

from mio_decomp.src.libraries.decompiler.parser import SaveParser

app = typer.Typer()


@app.command()
def parse(
    save_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the .save file to parse.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
            resolve_path=True,
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            "--output-dir",
            help="The file to output the JSON to. Will be overwritten if it exists. Will be printed to console if omitted.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
):
    """Parses a MIO .save file into JSON."""
    parser: SaveParser = SaveParser()
    if output_file is None:
        print(parser.parse_save(save_file))
    else:
        output_file.write_text(parser.parse_save(save_file))

@app.command()
def compile_save(
    json_file: Annotated[
        Path,
        typer.Argument(
            help="The path to the JSON file to parse.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
            resolve_path=True,
        ),
    ],
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            "--output-dir",
            help="The file to output the MIO .save to. Will be overwritten if it exists. Will be printed to console if omitted.",
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
        ),
    ] = None,
):
    """Parses a MIO JSON file into MIO .save."""
    parser: SaveParser = SaveParser()
    if output_file is None:
        print(parser.compile_save(json_file))
    else:
        output_file.write_text(parser.compile_save(json_file))