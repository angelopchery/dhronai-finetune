import typer

app = typer.Typer()


@app.command()
def preprocess(*args, **kwargs):
    from dhron_finetune.cli.preprocess import preprocess as _preprocess
    _preprocess(*args, **kwargs)


@app.command()
def train(*args, **kwargs):
    try:
        from dhron_finetune.cli.train import train as _train
    except ImportError:
        print(
            "\n[ERROR] Training dependencies not installed.\n"
            "Run: pip install dhronai[train]\n"
        )
        raise
    _train(*args, **kwargs)


@app.command()
def chat(*args, **kwargs):
    try:
        from dhron_finetune.cli.chat import chat as _chat
    except ImportError:
        print(
            "\n[ERROR] Chat dependencies missing.\n"
            "Run: pip install dhronai[train]\n"
        )
        raise
    _chat(*args, **kwargs)


if __name__ == "__main__":
    app()