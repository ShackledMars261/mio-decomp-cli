import typer

from .commands.config import app as config_app
from .commands.decompile import app as decompile_app
from .commands.version import app as version_app
from .commands.version import print_version_basic
from .config import config  # noqa: F401 # Import here so the config file is created

app = typer.Typer(
    help="MIO-Decomp: A CLI for decompiling the .gin files from MIO: Memories in Orbit.",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(config_app, name="config")
app.add_typer(decompile_app, name="decompile")
app.add_typer(version_app)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Prints the version of the package.",
        is_eager=True,
    ),
):
    if version:
        print_version_basic()
        raise typer.Exit()

    if ctx.invoked_subcommand is None and not ctx.params["version"]:
        typer.echo(ctx.get_help())
        raise typer.Exit()
