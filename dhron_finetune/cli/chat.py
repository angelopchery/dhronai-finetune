import typer
from rich.console import Console

from dhron_finetune.inference.loader import load_model_with_adapter
from dhron_finetune.inference.chat_loop import chat_loop
import dhron_finetune.utils.hardware as hardware


app = typer.Typer()
console = Console()


@app.command()
def chat(
    model_path: str = typer.Option(..., "--model", help="Path to trained model"),
):
    console.print("🚀 Starting chat...")

    # Detect hardware
    hw = hardware.get_device_info()
    device = hw["device"]

    console.print(f"🖥 Device: {device}")

    # Load model
    console.print("📦 Loading model + adapter...")
    model, tokenizer = load_model_with_adapter(model_path, device)

    console.print("[green]✅ Model ready[/green]")

    # Start chat
    chat_loop(model, tokenizer, device)