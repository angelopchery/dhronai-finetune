import torch


def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ safer extraction (prevents empty responses)
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    return response


def chat_loop(model, tokenizer, device):
    print("\n💬 DhronAI Chat (type 'exit' to quit)\n")

    # ✅ Strong system prompt
    history = "You are a helpful AI assistant."

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat...")
            break

        # ✅ Structured conversation format (VERY IMPORTANT)
        history += f"\n### User:\n{user_input}\n### Assistant:\n"

        response = generate_response(model, tokenizer, history, device)

        # ✅ Avoid empty responses breaking loop
        if not response.strip():
            response = "[No response generated]"

        history += response

        print(f"\nAssistant: {response}\n")