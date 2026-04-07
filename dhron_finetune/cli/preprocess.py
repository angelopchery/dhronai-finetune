import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

import dhron_finetune.data.cleaner as cleaner
import dhron_finetune.data.validator as validator
import dhron_finetune.data.stats as stats_module

app = typer.Typer()
console = Console()


@app.command()
def preprocess(
    input_path: str = typer.Option(..., "--input", help="Path to raw dataset"),
    output_path: str = typer.Option(..., "--output", help="Path to save processed dataset"),
):
    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        console.print("[red]Input file does not exist[/red]")
        raise typer.Exit()

    console.print("🔄 Loading dataset...")

    processed = []
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)

            if cleaner.is_garbage(sample):
                skipped += 1
                continue

            normalized = cleaner.normalize(sample)

            if cleaner.is_valid(normalized):
                processed.append(normalized)
            else:
                skipped += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # 🔥 IMPORTANT: dataset marker (Phase 1.5)
        f.write(json.dumps({
            "__dhron_processed__": True,
            "version": "1.0"
        }) + "\n")

        for row in processed:
            f.write(json.dumps(row) + "\n")

    console.print("[green]✅ Preprocessing complete[/green]")

    # Validate
    total, errors = validator.validate_jsonl(output_path)

    # Stats
    dataset_stats = stats_module.compute_stats(output_path)

    # Display
    table = Table(title="Dataset Summary")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Total Samples", str(dataset_stats["total"]))
    table.add_row("Avg Tokens", f"{dataset_stats['avg_tokens']:.2f}")
    table.add_row("Max Tokens", str(dataset_stats["max_tokens"]))
    table.add_row("Skipped Samples", str(skipped))
    table.add_row("Validation Errors", str(errors))

    console.print(table)


if __name__ == "__main__":
    app()