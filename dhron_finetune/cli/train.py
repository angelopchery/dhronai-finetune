import typer
from rich.console import Console
from rich.table import Table

from dhron_finetune.data.checker import is_preprocessed
import dhron_finetune.utils.hardware as hardware

from dhron_finetune.model.registry import MODEL_REGISTRY
from dhron_finetune.model.loader import load_model

from dhron_finetune.data.dataset import load_jsonl_dataset
from dhron_finetune.data.formatter import format_prompt
from dhron_finetune.data.tokenizer import tokenize_function

from dhron_finetune.model.lora import get_lora_config
from dhron_finetune.training.trainer import train_model


app = typer.Typer()
console = Console()


@app.command()
def train(
    data_path: str = typer.Option(..., "--data"),

    # 🔥 Configurable params
    epochs: float = typer.Option(1.0, help="Training epochs"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
    batch_size: int = typer.Option(1, help="Batch size"),
    max_length: int = typer.Option(512, help="Max sequence length"),

    lora_r: int = typer.Option(8, help="LoRA rank"),
    qlora: bool = typer.Option(False, help="Enable QLoRA (4-bit)")
):
    console.print("🚀 Starting training pipeline...")

    # ---------------- PREPROCESS CHECK ----------------
    if not is_preprocessed(data_path):
        console.print("[red]Dataset not preprocessed![/red]")
        raise typer.Exit()

    console.print("[green]Dataset validation passed ✔[/green]")

    # ---------------- HARDWARE ----------------
    hw = hardware.get_device_info()

    console.print(f"\n🖥 Device: {hw['device']}")
    if hw["device"] == "cuda":
        console.print("[green]🚀 GPU detected — fast training enabled[/green]")

    # ---------------- MODEL DISPLAY ----------------
table = Table(title="Available Models")
table.add_column("Index")
table.add_column("Model")
table.add_column("Min VRAM")

for i, model in enumerate(MODEL_REGISTRY):

    lora = "✔" if model.get("supports_lora", False) else "✖"
    qlora_flag = "✔" if model.get("supports_qlora", False) else "✖"

    label = f"{model['label']} | LoRA:{lora} QLoRA:{qlora_flag}"

    if not model.get("safe", True):
        label += " ⚠"

    table.add_row(str(i), label, f"{model['min_vram']} GB")

console.print(table)


# ---------------- MODEL SELECTION ----------------
choice = int(typer.prompt("Select model index"))
selected_model_info = MODEL_REGISTRY[choice]

# 🚨 BLOCK UNSAFE MODELS
if not selected_model_info.get("safe", True):
    console.print("[red]❌ This model is not compatible with current environment[/red]")
    raise typer.Exit()

# 🚨 QLoRA VALIDATION
if qlora and not selected_model_info.get("supports_qlora", False):
    console.print("[yellow]⚠ QLoRA not supported for this model[/yellow]")
    console.print("[cyan]➡ Falling back to LoRA[/cyan]")
    qlora = False

selected_model = selected_model_info["name"]