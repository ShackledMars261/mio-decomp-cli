from typing import Annotated

import typer
from rich import print

from mio_decomp.src.config import config

app = typer.Typer()


@app.command()
def reset(
    confirm: Annotated[
        bool,
        typer.Option(
            help="Confirm that you want to reset your configuration.",
        ),
    ] = False,
):
    """Resets your configuration to the default."""
    if confirm:
        config.reset_config()
        print("Finished resetting your configuration.")
    else:
        print(
            'Please repeat the command with the "--confirm" flag if you\'re sure you want to reset your configuration.'
        )
