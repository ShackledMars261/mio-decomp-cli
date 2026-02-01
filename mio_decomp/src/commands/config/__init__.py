import typer

from .get import app as get_app
from .list import app as list_app
from .reset import app as reset_app
from .set import app as set_app

app = typer.Typer(
    help="View and edit the extension's configuration.",
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(get_app)
app.add_typer(list_app)
app.add_typer(reset_app)
app.add_typer(set_app)
