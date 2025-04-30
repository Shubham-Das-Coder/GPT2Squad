import os
import json
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import Dataset

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

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_dir="metrics_logs"):
        self.epoch_metrics = {}
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, "gpt2_vanilla_finetuning_squad.txt")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        log_history = state.log_history[-1]

        clm_loss = log_history.get("loss", None)
        learning_rate = log_history.get("learning_rate", None)

        perplexity = math.exp(clm_loss) if clm_loss is not None else None

        total_tokens = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.logging_steps
            * 512
        )

        self.epoch_metrics[epoch] = {
            "loss": clm_loss,
            "learning_rate": learning_rate,
            "perplexity": perplexity,
            "token_count": total_tokens,
        }

        print(f">>> Epoch {epoch} Summary:")
        print(f"   Loss: {clm_loss:.4f}" if clm_loss else "   Loss: N/A")
        print(f"   Perplexity: {perplexity:.2f}" if perplexity else "   Perplexity: N/A")
        print(f"   Learning Rate: {learning_rate:.6f}" if learning_rate else "   Learning Rate: N/A")
        print(f"   Token Count (approx.): {total_tokens}")

        with open(self.log_file_path, "a") as f:
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Loss: {clm_loss:.4f}\n" if clm_loss else "  Loss: N/A\n")
            f.write(f"  Perplexity: {perplexity:.2f}\n" if perplexity else "  Perplexity: N/A\n")
            f.write(f"  Learning Rate: {learning_rate:.6f}\n" if learning_rate else "  Learning Rate: N/A\n")
            f.write(f"  Token Count: {total_tokens}\n\n")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    samples = load_squad_finetune("data/train-v2.0.json")
    dataset = Dataset.from_list(samples)
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    output_dir = "models/gpt2-vanilla-finetune-squad"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=100,
        num_train_epochs=5,
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

    loss_logger = LossLoggerCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[loss_logger]
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Vanilla fine-tuned model saved to '{output_dir}/'")
    print("Metrics per epoch:", loss_logger.epoch_metrics)

if __name__ == "__main__":
    main()
