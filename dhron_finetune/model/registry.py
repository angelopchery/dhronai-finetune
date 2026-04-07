MODEL_REGISTRY = [
    {
        "name": "gpt2",
        "label": "GPT-2 (124M)",
        "min_vram": 2,
        "supports_lora": True,
        "supports_qlora": True,
        "safe": True
    },
    {
        "name": "distilgpt2",
        "label": "DistilGPT-2 (82M)",
        "min_vram": 2,
        "supports_lora": True,
        "supports_qlora": True,
        "safe": True
    },
    {
        "name": "facebook/opt-125m",
        "label": "OPT-125M",
        "min_vram": 4,
        "supports_lora": True,
        "supports_qlora": False,  # 🚨 IMPORTANT
        "safe": False
    }
]