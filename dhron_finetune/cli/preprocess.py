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


def is_bad_sample(sample):
    """
    Extra filtering beyond cleaner
    """

    # ❌ remove generated garbage
    if str(sample.get("input", "")).lower().strip() == "generated question":
        return True

    # ❌ remove nested JSON outputs
    output = sample.get("output", "")
    if isinstance(output, str) and "qa_pairs" in output:
        return True

    return False


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
    seen = set()  # 🔥 dedup

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
            except Exception:
                skipped += 1
                continue

            # 🔥 strict filtering
            if is_bad_sample(sample):
                skipped += 1
                continue

            if cleaner.is_garbage(sample):
                skipped += 1
                continue

            normalized = cleaner.normalize(sample)

            if not cleaner.is_valid(normalized):
                skipped += 1
                continue

            user = str(normalized.get("input", "")).strip()
            assistant = str(normalized.get("output", "")).strip()

            # ❌ empty guard
            if not user or not assistant:
                skipped += 1
                continue

            # ❌ ultra short guard (important for GPT2 stability)
            if len(user) < 3 or len(assistant) < 3:
                skipped += 1
                continue

            # 🔥 dedup
            key = (user, assistant)
            if key in seen:
                continue
            seen.add(key)

            # 🔥 FINAL FORMAT (CRITICAL FOR TRAINING)
            formatted = {
                "text": f"User: {user}\nAssistant: {assistant}"
            }

            processed.append(formatted)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # 🔥 dataset marker
        f.write(json.dumps({
            "__dhron_processed__": True,
            "version": "2.0"
        }) + "\n")

        for row in processed:
            f.write(json.dumps(row) + "\n")

    console.print("[green]✅ Preprocessing complete[/green]")

    # 🔍 sanity preview (VERY IMPORTANT)
    console.print("\n[blue]🔎 Sample Output Preview:[/blue]\n")
    for i in range(min(3, len(processed))):
        console.print(processed[i]["text"])
        console.print("-" * 50)

    # Validate
    total, errors = validator.validate_jsonl(output_path)

    # Stats
    dataset_stats = stats_module.compute_stats(output_path)

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