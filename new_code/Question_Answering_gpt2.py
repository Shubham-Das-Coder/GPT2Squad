from datasets import load_dataset
import numpy as np
import os
from evaluate import load
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer,TrainingArguments, Trainer,TrainerCallback

model_name = "gpt2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained("/home/ubuntu/gpt2_qa_best/best_model_63.3825/")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.sep_token = "[SEP]"

dataset = load_dataset("squad")

train_data = dataset["train"]
val_data = dataset["validation"]

print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")

sample = train_data[0]
print("样本结构:", sample.keys())
for key, value in sample.items():
    print(f"{key} shape:", value.shape if hasattr(value, 'shape') else len(value))

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]
    
    encoded = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors=None
    )
    sample_mapping = encoded.pop("overflow_to_sample_mapping")
    offset_mapping = encoded.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    for i, offset in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = examples["answers"][sample_idx]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
    
            start_token = None
            end_token = None
            
            for idx, (start, end) in enumerate(offset):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break
            
            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = 0
                
            start_positions.append(start_token)
            end_positions.append(end_token)
        
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "start_positions": torch.tensor(start_positions),
        "end_positions": torch.tensor(end_positions),
        "example_id": [str(i) for i in range(len(encoded["input_ids"]))],  # 添加唯一ID
        "context": [examples["context"][sample_mapping[i]] for i in range(len(encoded["input_ids"]))],  # 保存上下文
        "question": [examples["question"][sample_mapping[i]] for i in range(len(encoded["input_ids"]))],  # 保存问题
    }
train_data = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names
)

val_data = val_data.map(
    preprocess_function,
    batched=True,
    remove_columns=val_data.column_names
)

metric = load("squad")

def compute_metrics(eval_pred):
    start_logits, end_logits = eval_pred.predictions
    _, inputs = eval_pred.label_ids
    
    start_pred = np.argmax(start_logits, axis=1)
    end_pred = np.argmax(end_logits, axis=1)
    
    predictions = []
    references = []
    
    for idx in range(len(start_pred)):
        try:
            input_ids = inputs["input_ids"][idx]
        
            if start_pred[idx] <= end_pred[idx]:
                pred_text = tokenizer.decode(
                    input_ids[start_pred[idx]:end_pred[idx] + 1],
                    skip_special_tokens=True
                )
            else:
                pred_text = ""
                
            start_pos = inputs["start_positions"][idx]
            end_pos = inputs["end_positions"][idx]
            ref_text = tokenizer.decode(
                input_ids[start_pos:end_pos + 1],
                skip_special_tokens=True
            )
            
            predictions.append({
                'id': str(idx),
                'prediction_text': pred_text
            })
            
            references.append({
                'id': str(idx),
                'answers': {
                    'text': [ref_text],
                    'answer_start': [start_pos.item()]
                }
            })
            
        except Exception as e:
            print(f"Error processing prediction {idx}: {e}")
            continue
    
    result = metric.compute(predictions=predictions, references=references)
    return {
        "f1": result["f1"],
        "exact_match": result["exact_match"]
    }
    
training_args = TrainingArguments(
    output_dir="./gpt2_qa",
    num_train_epochs=5,
    save_total_limit=2,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    warmup_steps=500,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to=["tensorboard"],
    lr_scheduler_type="cosine",
)

class SaveBestModelCallback(TrainerCallback):
    def __init__(self, save_path, tokenizer):
        self.save_path = save_path
        self.best_metric = None
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get("eval_f1")
        if metric_value is not None:
            if self.best_metric is None or metric_value > self.best_metric:
                self.best_metric = metric_value
                model_path = os.path.join(self.save_path, f"best_model_{metric_value:.4f}")
                kwargs['model'].save_pretrained(model_path)
                self.tokenizer.save_pretrained(model_path)
                print(f"Saved new best model with F1 score: {metric_value:.4f}")

class QuestionAnsweringTrainer(Trainer):
     def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys = None):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        return (loss, logits, (labels, inputs))

trainer = QuestionAnsweringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestModelCallback(save_path="./gpt2_qa_best", tokenizer=tokenizer)] 
)

trainer.train()

model.save_pretrained("./gpt2_qa")
tokenizer.save_pretrained("./gpt2_qa")
print("Fine-tuned model saved to ./gpt2_qa.")
