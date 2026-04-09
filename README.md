# 🚀 DhronAI

A powerful CLI-based tool for fine-tuning language models using **LoRA** and **QLoRA**, designed for efficiency, flexibility, and real-world usability.

---

## 🖼️ Demo

<p align="center">
  <img src="assets/image.png" width="700"/>
</p>

---

## ⚙️ Features

* ⚡ CLI-based workflow using **Typer + Rich**
* 🧠 LoRA fine-tuning support
* 🚀 QLoRA (4-bit quantized training)
* 💻 Hardware-aware model selection (CPU/GPU)
* 🧹 Dataset preprocessing and validation
* 🧩 Modular and extensible pipeline

---

## 📦 Installation

### 🔹 Install from Wheel

```bash
pip install dist/dhronai-0.1.0-py3-none-any.whl
```

### 🔹 CPU Support

```bash
pip install "dhronai[cpu]"
```

### 🔹 GPU Support (Recommended)

```bash
pip install "dhronai[gpu]"
```

> ⚡ GPU mode enables faster training and efficient QLoRA (4-bit quantization)

---

## 🚀 Usage

### 🔹 Preprocess Dataset

```bash
dhronai preprocess preprocess \
  --input-path path/to/raw_data.json \
  --output-path data/processed/clean.jsonl
```

---

### 🔹 Train Model (LoRA)

```bash
dhronai train \
  --data data/processed/clean.jsonl
```

---

### 🔹 Train with QLoRA (4-bit)

```bash
dhronai train \
  --data data/processed/clean.jsonl \
  --qlora
```

---

### 🔹 Customize Training

```bash
dhronai train \
  --data data/processed/clean.jsonl \
  --epochs 2 \
  --batch-size 2 \
  --lr 2e-4
```

---

### 🔹 View All Commands

```bash
dhronai --help
dhronai train --help
dhronai preprocess --help
```

---

## 🤖 Supported Models

* GPT-2 (124M)
* DistilGPT-2 (82M)
* OPT-125M *(limited compatibility)*

---

## 📁 Output

Training outputs include:

* Adapter weights
* Tokenizer files
* Training artifacts

Saved in:

```bash
outputs/
```

---

## ⚙️ Notes

* ⚠️ GPU is recommended for QLoRA training
* 🖥️ CPU mode works for smaller models
* 📌 Dataset must be preprocessed before training

---

## 🧪 Example Workflow

```bash
# Step 1: Preprocess
dhronai preprocess preprocess \
  --input-path raw.json \
  --output-path clean.jsonl

# Step 2: Train
dhronai train --data clean.jsonl --qlora
```

---

## 🏁 Summary

DhronAI provides an end-to-end pipeline for:

* Data preprocessing
* Model selection
* Efficient fine-tuning (LoRA / QLoRA)
* Local execution

Built for **simplicity, extensibility, and practical AI workflows**.
