import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import bitsandbytes as bnb

def load_squad_finetune(path):
    with open(path) as f:
        squad = json.load(f)["data"]

    samples = []
    for article in squad:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa.get("is_impossible", False):
                    continue

                question = qa["question"]
                answers = qa.get("answers", [])
                answer = answers[0]["text"] if answers else ""

                text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
                samples.append({"text": text})
    return samples

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config
    )

    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = load_squad_finetune("data/train-v2.0.json")
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    output_dir = "models/gpt2-qlora-squad"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=100,
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"QLoRA fine-tuned model saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
