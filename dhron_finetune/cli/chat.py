import os
import time
import typer
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text

from dhron_finetune.inference.loader import load_model_with_adapter
from dhron_finetune.inference.chat_loop import chat_loop
from dhron_finetune.utils.hardware import get_device_info


def show_splash_screen(console: Console):
    """
    Display clean ASCII splash screen for DhronAI branding
    """

    banner = r"""
██████╗ ██╗  ██╗██████╗  ██████╗ ███╗   ██╗ █████╗ ██╗
██╔══██╗██║  ██║██╔══██╗██╔═══██╗████╗  ██║██╔══██╗██║
██║  ██║███████║██████╔╝██║   ██║██╔██╗ ██║███████║██║
██║  ██║██╔══██║██╔══██╗██║   ██║██║╚██╗██║██╔══██║██║
██████╔╝██║  ██║██║  ██║╚██████╔╝██║ ╚████║██║  ██║██║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝
"""

    console.print(Align.center(Text(banner, style="bold green")))
    console.print(Align.center("[bold white]DhronAI[/bold white]"))
    console.print(Align.center("[dim]AI Fine-Tuning CLI[/dim]\n"))

    time.sleep(0.4)


def show_init_panel(console: Console, model: str, device: str):
    """
    Unified initialization panel (single box as requested)
    """

    content = (
        f"[bold cyan]Base Model:[/bold cyan] gpt2\n"
        f"[bold cyan]Adapter Path:[/bold cyan] {model}\n"
        f"[bold cyan]Device:[/bold cyan] {device}\n"
        f"[bold cyan]Status:[/bold cyan] [green]Ready[/green]"
    )

    panel = Panel(
        content,
        title="[bold white]🚀 DhronAI Initialization[/bold white]",
        border_style="green",
        padding=(1, 2),
    )

    console.print(panel)


def show_section_divider(console: Console):
    """
    Clean section separator before chat
    """

    divider = "[bold green]" + "━" * 60 + "[/bold green]"

    console.print()
    console.print(Align.center(divider))
    console.print(Align.center("[bold green]💬 Chat with Assistant[/bold green]"))
    console.print(Align.center(divider))
    console.print()


def chat(
    model: str = typer.Option(..., "--model", help="Path to trained model"),
):
    console = Console()

    # Splash screen
    show_splash_screen(console)

    hw = get_device_info()
    device = hw["device"]

    # Validate model path exists
    if not os.path.exists(model):
        console.print(f"[red][ERROR] Model path not found: {model}[/red]")
        console.print("[yellow]Please provide a valid local path to a trained model directory.[/yellow]")
        raise typer.Exit(code=1)

    if not os.path.isdir(model):
        console.print(f"[red][ERROR] Path is not a directory: {model}[/red]")
        raise typer.Exit(code=1)

    # Loading (this stays OUTSIDE panel for clean UX)
    with console.status("[bold green]⏳ Loading model + adapter...[/bold green]"):
        model_obj, tokenizer = load_model_with_adapter(model, device)

    # Unified panel
    show_init_panel(console, model, device)

    time.sleep(0.3)

    # Chat section
    show_section_divider(console)

    # Start chat
    chat_loop(model_obj, tokenizer, device, console)