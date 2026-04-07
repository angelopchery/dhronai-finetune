from transformers import TrainingArguments
from trl import SFTTrainer
from peft import get_peft_model
import torch


def train_model(
    model,
    tokenizer,
    dataset,
    lora_config,
    epochs=1.0,
    lr=2e-4,
    batch_size=1,
    qlora=False,
    output_dir="outputs"
):

    model = get_peft_model(model, lora_config)

    use_cuda = torch.cuda.is_available()

    if qlora:
        fp16 = False
        bf16 = use_cuda
    else:
        fp16 = use_cuda
        bf16 = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=5,
        save_steps=50,
        fp16=fp16,
        bf16=bf16,

        # 🔥 disable problematic scaler behavior
        optim="adamw_torch",

        gradient_accumulation_steps=1,
        max_grad_norm=1.0,

        # 🔥 prevents AMP crash
        fp16_full_eval=False,

        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted. Saving model...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir