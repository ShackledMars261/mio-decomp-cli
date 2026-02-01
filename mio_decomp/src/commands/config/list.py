from typing import Annotated

import typer
from rich import print

from ...config import config

app = typer.Typer()


@app.command()
def list(
    json: Annotated[
        bool,
        typer.Option(help="Returns the pairs as JSON."),
    ] = False,
):
    """Lists all key-value pairs in the current configuration."""
    if json:
        print(config.config_model.model_dump_json(indent=4))
    else:
        for key, value in config.config_model:
            print(f'"{key}": "{value}"')
