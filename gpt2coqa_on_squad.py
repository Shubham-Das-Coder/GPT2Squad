from datasets import load_dataset
import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from evaluate import load as load_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "./gpt2coqa"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("squad")
val_data = dataset["validation"]

def preprocess_squad(example):
    question = example["question"]
    context = example["context"]
    answers = example["answers"]

    answer = answers["text"][0]
    start_char = answers["answer_start"][0]
    end_char = start_char + len(answer)

    encoding = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_offsets_mapping=True,
        padding="max_length"
    )

    offset_mapping = encoding.pop("offset_mapping")
    input_ids = encoding["input_ids"]

    start_token = 0
    end_token = 0
    for idx, (start, end) in enumerate(offset_mapping):
        if start <= start_char < end:
            start_token = idx
        if start < end_char <= end:
            end_token = idx
            break

    encoding["start_positions"] = start_token
    encoding["end_positions"] = end_token
    encoding["id"] = example["id"]
    encoding["context"] = context
    return encoding

val_data = val_data.map(preprocess_squad, batched=False)

val_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

metric = load_metric("squad")

def compute_metrics(eval_pred):
    start_logits, end_logits = eval_pred.predictions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)

    references = []
    predictions = []

    for i in range(len(start_preds)):
        input_ids = val_data[i]["input_ids"].tolist()
        ref_start = val_data[i]["start_positions"].item()
        ref_end = val_data[i]["end_positions"].item()

        ref_answer = tokenizer.decode(input_ids[ref_start:ref_end+1], skip_special_tokens=True)
        pred_answer = tokenizer.decode(input_ids[start_preds[i]:end_preds[i]+1], skip_special_tokens=True)

        references.append({
            "id": str(i),
            "answers": {
                "text": [ref_answer],
                "answer_start": [ref_start]
            }
        })
        predictions.append({
            "id": str(i),
            "prediction_text": pred_answer
        })

    return metric.compute(predictions=predictions, references=references)

training_args = TrainingArguments(
    output_dir="./eval_gpt2coqa_on_squad",
    per_device_eval_batch_size=50,
    do_train=False,
    do_eval=True,
    logging_dir="./logs_eval",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

metrics = trainer.evaluate()

print(f"\nEvaluation on SQuAD val split - F1: {metrics['eval_f1']:.4f}, EM: {metrics['eval_exact_match']:.4f}")

import os
os.makedirs("logs", exist_ok=True)
with open("logs/gpt2coqa_on_squad.txt", "w") as f:
    f.write(f"F1 Score: {metrics['eval_f1']:.4f}\n")
    f.write(f"Exact Match: {metrics['eval_exact_match']:.4f}\n")
