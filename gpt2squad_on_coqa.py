from datasets import load_dataset
import numpy as np
import os
import torch
from evaluate import load as load_metric
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "./gpt2squad"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.sep_token is None:
    tokenizer.sep_token = tokenizer.eos_token

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.to(device)

dataset = load_dataset("coqa")
val_data = dataset["validation"]

def flatten_coqa(data):
    flattened = {
        "story": [],
        "question": [],
        "answer": [],
        "answer_start": []
    }
    for example in data:
        context = example["story"]
        questions = example["questions"]
        answers = example["answers"]
        for i in range(len(questions)):
            flattened["story"].append(context)
            flattened["question"].append(questions[i])
            flattened["answer"].append(answers["input_text"][i])
            flattened["answer_start"].append(answers["answer_start"][i])
    return flattened

flat_val_data = flatten_coqa(val_data)

from datasets import Dataset
flat_val_dataset = Dataset.from_dict(flat_val_data)

def preprocess_coqa(example):
    question = example["question"]
    context = example["story"]
    answer = example["answer"]
    start_char = example["answer_start"]
    end_char = start_char + len(answer)

    encoding = tokenizer(
        question,
        context,
        truncation=True,
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

    if start_token == 0 and end_token == 0:
        start_token = tokenizer.pad_token_id
        end_token = tokenizer.pad_token_id

    encoding["start_positions"] = start_token
    encoding["end_positions"] = end_token

    return encoding

flat_val_dataset = flat_val_dataset.map(preprocess_coqa, batched=False, remove_columns=flat_val_dataset.column_names)

flat_val_dataset.set_format(
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
        input_ids = flat_val_dataset[i]["input_ids"].tolist()
        ref_start = flat_val_dataset[i]["start_positions"].item()
        ref_end = flat_val_dataset[i]["end_positions"].item()

        ref_answer = tokenizer.decode(input_ids[ref_start:ref_end+1], skip_special_tokens=True)
        pred_answer = tokenizer.decode(input_ids[start_preds[i]:end_preds[i]+1], skip_special_tokens=True)

        references.append({"id": str(i), "answers": {"text": [ref_answer], "answer_start": [ref_start]}})
        predictions.append({"id": str(i), "prediction_text": pred_answer})

    return metric.compute(predictions=predictions, references=references)

training_args = TrainingArguments(
    output_dir="./eval_gpt2squad_on_coqa",
    per_device_eval_batch_size=50,
    do_train=False,
    do_eval=True,
    logging_dir="./logs_eval",
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=flat_val_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

metrics = trainer.evaluate()

print(f"\nEvaluation on CoQA val split - F1: {metrics['eval_f1']:.4f}, EM: {metrics['eval_exact_match']:.4f}")

os.makedirs("logs", exist_ok=True)
with open("logs/gpt2squad_on_coqa.txt", "w") as f:
    f.write(f"F1 Score: {metrics['eval_f1']:.4f}\n")
    f.write(f"Exact Match: {metrics['eval_exact_match']:.4f}\n")
