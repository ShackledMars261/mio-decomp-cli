from typing import Annotated

import typer
from rich import print

from ...config import config

app = typer.Typer()


@app.command()
def get(
    key: Annotated[
        str, typer.Argument(help="The key to get the value of. Must be a valid key.")
    ],
):
    """Gets the value of a key in the configuration."""
    try:
        value: str = config.get_value_from_key(key)
        print(f'"{key}": "{value}"')
    except AttributeError:
        print(f'Key "{key}" not found.')
