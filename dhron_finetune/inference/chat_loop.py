import time
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


def stream_response(console: Console, response: str):
    """Simulate streaming delay without printing to console."""
    # Internal delay only - simulate typing time
    word_count = len(response.split())
    time.sleep(word_count * 0.02)
    return response


def display_message(console: Console, role: str, message: str):
    """Display a message in a styled panel."""
    # Clean up artifacts before display
    message = message.replace("User:", "").replace("Assistant:", "").replace("###", "").strip()

    if role == "user":
        # Right-aligned user panel
        terminal_width = console.width
        panel_content_width = len(message) + 10  # Approximate panel overhead
        panel_width = min(panel_content_width, terminal_width - 20)

        panel = Panel(
            message,
            title="(You)",
            border_style="cyan",
            padding=(0, 1),
        )

        # Calculate padding for right alignment
        padding = max(0, terminal_width - panel_width - 5)
        console.print(" " * padding, end="")
        console.print(panel)

    elif role == "assistant":
        # Left-aligned assistant panel
        panel = Panel(
            message,
            title="(Assistant)",
            border_style="green",
            padding=(0, 1),
        )
        console.print(panel)

    console.print()  # Spacing between turns


def display_error(console: Console, message: str):
    """Display error message in a styled panel."""
    panel = Panel(
        f"[yellow]{message}[/yellow]",
        title="[bold red]⚠️ Warning[/bold red]",
        border_style="yellow",
        padding=(0, 1),
    )
    console.print(panel)
    console.print()


def generate_response(model, tokenizer, prompt, device, temperature=0.7):
    """Generate response from model - UI-agnostic core logic."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant response
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    # HARD STOP: prevent multi-turn leakage
    if "User:" in response:
        response = response.split("User:")[0]
    if "Assistant:" in response:
        response = response.split("Assistant:")[0]

    response = response.strip()

    return response


def build_prompt(user_input, conversation_buffer):
    """Build conversational prompt with history buffer."""
    prompt = f"User: {user_input}\nAssistant:"

    if conversation_buffer:
        history_prompt = ""
        for turn in conversation_buffer:
            history_prompt += f"User: {turn['input']}\nAssistant: {turn['response']}\n"

        prompt = history_prompt + prompt

    return prompt


def chat_loop(model, tokenizer, device, console: Console):
    """Main chat loop with rich UI."""
    conversation_buffer = []
    MAX_BUFFER_SIZE = 3

    while True:
        # Styled input prompt
        user_input = Prompt.ask("[bold cyan]💬 You >[/bold cyan]")

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]👋 Exiting chat...[/yellow]")
            break

        # Build prompt with context
        prompt = build_prompt(user_input, conversation_buffer)

        # First generation attempt with spinner
        with console.status("[cyan]⏳ Thinking...[/cyan]"):
            response = generate_response(
                model, tokenizer, prompt, device, temperature=0.7
            )

        # Retry with higher randomness if weak output
        if len(response) < 5:
            with console.status("[cyan]⏳ Retrying...[/cyan]"):
                response = generate_response(
                    model, tokenizer, prompt, device, temperature=0.8
                )

        # Handle generation failure
        if len(response) < 5:
            display_error(console, "Model could not generate a response")
            response = "[Model could not generate a response]"

        # Display user message (right-aligned panel)
        display_message(console, "user", user_input)

        # Stream response (internal delay only)
        stream_response(console, response)

        # Display assistant message (left panel)
        display_message(console, "assistant", response)

        # Add to conversation buffer
        conversation_buffer.append({
            "input": user_input,
            "response": response
        })

        # Maintain buffer size
        if len(conversation_buffer) > MAX_BUFFER_SIZE:
            conversation_buffer.pop(0)
