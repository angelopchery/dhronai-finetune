from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(model_name: str, device: str, qlora: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = device == "cuda"

    # ---------------- QLoRA ----------------
    if qlora and use_cuda:
        try:
            from transformers import BitsAndBytesConfig

            print("⚡ Loading model with QLoRA (4-bit)...")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )

            # 🔥 CRITICAL FIX FOR TRAINING STABILITY
            model.config.use_cache = False

            return model, tokenizer

        except Exception as e:
            print(f"⚠ QLoRA failed: {e}")
            print("➡ Falling back to standard LoRA")

    # ---------------- STANDARD LOAD ----------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if use_cuda else None
    )

    model.config.use_cache = False

    return model, tokenizer