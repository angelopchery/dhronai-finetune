MODEL_REGISTRY = [
    {
        "label": "GPT-2 (124M)",
        "name": "gpt2",
        "min_vram": 2,
        "supports_lora": True,
        "supports_qlora": True,
        "safe": True
    },
    {
        "label": "DistilGPT-2 (82M)",
        "name": "distilgpt2",
        "min_vram": 2,
        "supports_lora": True,
        "supports_qlora": True,
        "safe": True
    },
    {
        "label": "OPT-125M",
        "name": "facebook/opt-125m",
        "min_vram": 4,
        "supports_lora": True,
        "supports_qlora": False,
        "safe": False  # 🚨 blocked
    },
]