from setuptools import setup, find_packages

setup(
    name="dhronai",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "train": [
            "torch>=2.5.1",
            "transformers>=5.5.0",
            "datasets>=4.7.0",
            "peft>=0.18.1",
            "trl>=1.0.0",
            "bitsandbytes",
        ]
    },
    entry_points={
        "console_scripts": [
            "dhronai=dhron_finetune.main:app",
        ],
    },
)