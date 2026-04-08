import typer

from dhron_finetune.cli.preprocess import preprocess
from dhron_finetune.cli.train import train
from dhron_finetune.cli.chat import chat

app = typer.Typer()

app.command()(preprocess)
app.command()(train)
app.command()(chat)

if __name__ == "__main__":
    app()