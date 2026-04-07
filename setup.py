from setuptools import setup, find_packages

setup(
    name="dhronai",
    version="0.1.0",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "typer",
        "rich",
        "transformers",
        "peft",
        "bitsandbytes",
        "trl",
        "torch",
        "datasets",
    ],
    entry_points={
        "console_scripts": [
            "dhronai=dhron_finetune.main:app",
        ],
    },
)