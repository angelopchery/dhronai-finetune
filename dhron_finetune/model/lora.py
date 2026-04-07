from peft import LoraConfig


def get_lora_config(model_name: str):
    """
    Model-aware LoRA config
    """

    model_name = model_name.lower()

    # 🔥 GPT-2
    if "gpt2" in model_name:
        target_modules = ["c_attn"]

    # 🔥 OPT
    elif "opt" in model_name:
        target_modules = ["q_proj", "v_proj"]

    # 🔥 BLOOM
    elif "bloom" in model_name:
        target_modules = ["query_key_value"]

    # 🔥 fallback (safe default)
    else:
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )