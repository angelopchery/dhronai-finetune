from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def load_model_with_adapter(model_path: str, device: str):
    """
    Load base model + LoRA adapter safely
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ✅ Ensure pad token exists (prevents generation warnings/issues)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ Load base model safely
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True,  # 🔥 fixes LoRA mismatch warnings
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    # ✅ Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)

    model.eval()

    return model, tokenizer