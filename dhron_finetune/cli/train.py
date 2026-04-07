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
        label = model["label"]

        # 🔥 Show warning in UI
        if not model.get("safe", True):
            label += " ⚠ (Limited compatibility)"

        table.add_row(str(i), label, f"{model['min_vram']} GB")

    console.print(table)

    # ---------------- MODEL SELECTION ----------------
    choice = int(typer.prompt("Select model index"))
    selected_model_info = MODEL_REGISTRY[choice]

    # 🔥 HARD BLOCK (VERY IMPORTANT)
    if not selected_model_info.get("safe", True):
        console.print("[red]❌ This model is not compatible with current torch version.[/red]")
        console.print("👉 Please select a supported model (GPT-2 / DistilGPT-2)")
        raise typer.Exit()

    selected_model = selected_model_info["name"]

    console.print(f"\n📦 Loading model: {selected_model}")

    model, tokenizer = load_model(selected_model, hw["device"], qlora=qlora)

    console.print("[green]✅ Model loaded successfully[/green]")

    # ---------------- DATASET ----------------
    console.print("\n📊 Preparing dataset...")

    dataset = load_jsonl_dataset(data_path)

    # GPU-friendly sampling
    MAX_SAMPLES = 1000
    dataset = dataset.select(range(min(len(dataset), MAX_SAMPLES)))

    dataset = dataset.map(format_prompt)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )

    console.print("[green]✅ Dataset ready[/green]")

    # ---------------- LORA ----------------
    console.print("\n⚙ Applying LoRA configuration...")
    lora_config = get_lora_config(selected_model)
    lora_config.r = lora_r

    # ---------------- TRAIN ----------------
    console.print("\n🚀 Starting training...")

    output_dir = train_model(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        lora_config=lora_config,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        qlora=qlora
    )

    console.print(f"\n[green]✅ Training complete![/green]")
    console.print(f"📁 Saved to: {output_dir}")


if __name__ == "__main__":
    app()