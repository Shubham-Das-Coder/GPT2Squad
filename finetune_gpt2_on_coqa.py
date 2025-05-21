from datasets import load_dataset
import numpy as np
import os
import torch
from evaluate import load as load_metric
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = "[SEP]"

model.to(device)

dataset = load_dataset("coqa")
train_data = dataset["train"]
val_data = dataset["validation"]

def coqa_to_squad_format(example, idx):
    ids = []
    contexts = []
    questions = []
    answers = []
    context = example["story"]
    question_list = example["questions"]
    answer_list = example["answers"]["input_text"]
    for i in range(len(question_list)):
        answer_text = answer_list[i]
        answer_start = context.find(answer_text)
        if answer_start == -1:
            continue
        ids.append(f"{idx}_{i}")
        contexts.append(context)
        questions.append(question_list[i])
        answers.append({"text": [answer_text], "answer_start": [answer_start]})
    return {"id": ids, "context": contexts, "question": questions, "answers": answers}

train_data = train_data.map(lambda example, idx: coqa_to_squad_format(example, idx), with_indices=True, batched=False, remove_columns=train_data.column_names)
val_data = val_data.map(lambda example, idx: coqa_to_squad_format(example, idx), with_indices=True, batched=False, remove_columns=val_data.column_names)

def preprocess_function(examples):
    questions = []
    contexts = []
    answers = []

    for q_list, c_list, a_list in zip(examples["question"], examples["context"], examples["answers"]):
        for q, c, a in zip(q_list, c_list, a_list):
            questions.append(q.strip())
            contexts.append(c.strip())
            answers.append(a)

    encoded = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = encoded["input_ids"][i]
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        start_token = 0
        end_token = 0

        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start < end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token)
        end_positions.append(end_token)

    encoded["start_positions"] = start_positions
    encoded["end_positions"] = end_positions

    return encoded

train_data = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
val_data = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)

metric = load_metric("squad")

def compute_metrics(eval_pred):
    start_logits, end_logits = eval_pred.predictions
    start_preds = np.argmax(start_logits, axis=1)
    end_preds = np.argmax(end_logits, axis=1)
    references = []
    predictions = []
    for i in range(len(start_preds)):
        input_ids = val_data[i]["input_ids"]
        ref_start = val_data[i]["start_positions"]
        ref_end = val_data[i]["end_positions"]
        ref_answer = tokenizer.decode(input_ids[ref_start:ref_end+1], skip_special_tokens=True)
        pred_answer = tokenizer.decode(input_ids[start_preds[i]:end_preds[i]+1], skip_special_tokens=True)
        references.append({"id": str(i), "answers": {"text": [ref_answer], "answer_start": [ref_start]}})
        predictions.append({"id": str(i), "prediction_text": pred_answer})
    return metric.compute(predictions=predictions, references=references)

output_dir = "./finetune_gpt2_on_coqa"
log_path = os.path.join("logs", "finetune_gpt2_on_coqa.txt")

class UnifiedLoggerCallback(TrainerCallback):
    def __init__(self):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write("====== GPT-2 Fine-Tuning Log ======\n\n")
            f.write("Step\tEpoch\tMetrics\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = logs.get("epoch", "N/A")
            with open(log_path, "a") as f:
                f.write(f"{step}\t{epoch}\t{logs}\n")

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[UnifiedLoggerCallback()]
)

trainer.train()

os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

metrics = trainer.evaluate()
f1 = metrics.get("eval_f1", 0.0)
em = metrics.get("eval_exact_match", 0.0)

with open(log_path, "a") as f:
    f.write("\n====== Final Evaluation Results ======\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Exact Match: {em:.4f}\n")

print(f"\nTraining complete. Model saved to '{output_dir}'")
print(f"Final Evaluation - F1: {f1:.4f}, EM: {em:.4f}")
print(f"All logs saved to '{log_path}'")
