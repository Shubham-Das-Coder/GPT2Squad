import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path):
    with open(path, 'r') as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data:
        story = item["story"].strip()
        question = item["question"].strip()
        answer = item["answer"].strip()

        prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
        target = f" {answer}"

        processed.append((prompt, target))

    return processed

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.labels = []

        for prompt, answer in data:
            full_text = prompt + answer
            encodings = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

            labels = encodings.input_ids.clone()

            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)['input_ids']
            prompt_len = len(prompt_ids)
            labels[0][:prompt_len] = -100

            self.inputs.append(encodings)
            self.labels.append(labels)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(0),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(0),
            'labels': self.labels[idx].squeeze(0)
        }

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.resize_token_embeddings(len(tokenizer))

data_path = "data/modified/coqa_train.json"
full_data = load_dataset(data_path)
train_dataset = QADataset(full_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 5
output_log = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_answer_tokens = 0

    print(f"\nEpoch {epoch+1}/{epochs}")
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=-1)
        labels = batch['labels']
        mask = labels != -100
        correct = (predictions == labels) & mask
        total_correct += correct.sum().item()
        total_answer_tokens += mask.sum().item()

        progress_bar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    perplexity = torch.exp(torch.tensor(avg_train_loss))
    accuracy = total_correct / total_answer_tokens if total_answer_tokens > 0 else 0

    log_str = (f"Epoch {epoch+1}: "
               f"Train Loss = {avg_train_loss:.4f} | "
               f"Perplexity = {perplexity:.2f} | "
               f"Accuracy = {accuracy*100:.2f}%")
    print(log_str)
    output_log.append(log_str)

with open("gpt2coqa.txt", "w") as f:
    for line in output_log:
        f.write(line + "\n")

model_save_path = "gpt2coqa_model"
model.save_pretrained(model_save_path)

tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")