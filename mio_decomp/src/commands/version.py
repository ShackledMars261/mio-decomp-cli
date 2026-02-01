import importlib.metadata

import typer
from rich import print

app = typer.Typer()


@app.command()
def version():
    """Prints the currently installed version of the package."""
    print(f"MIO-Decomp CLI Version {importlib.metadata.version('mio_decomp')}")


def print_version_basic():
    print(f"{importlib.metadata.version('mio_decomp')}")
