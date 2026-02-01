from typing import Annotated

import typer
from rich import print

from ...config import config

app = typer.Typer()


@app.command()
def set(
    key: Annotated[str, typer.Argument(help="The key to set. Must be a valid key.")],
    value: Annotated[str, typer.Argument(help="The value to set the key to.")],
):
    """Set the value of a key in the configuration."""
    try:
        config.set_value_from_key(key, value)
        print(f'"{key}" set successfully.')
    except AttributeError:
        print(f'Key "{key}" not found.')
