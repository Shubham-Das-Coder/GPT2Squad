import json
import torch
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import evaluate

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def load_squad(path):
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

                samples.append({
                    "input": f"Question: {question}\nContext: {context}\nAnswer: ",
                    "output": answer
                })
    return samples

def evaluate_model(model_path="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)

    dev = load_squad("data/dev-v2.0.json")

    predictions = []
    for idx, item in enumerate(dev):
        input_text = item["input"]
        expected_answer = item["output"]

        print(f"Sample {idx+1}/{len(dev)}")
        print(f"Context: {input_text.split('Answer:')[0].strip()}\n")
        print(f"Question: {input_text.split('Answer:')[0].strip().split('Context:')[1]}\n")

        output = gen(input_text, max_new_tokens=50, do_sample=False)[0]["generated_text"]
        predicted = output.split("Answer:")[-1].strip().split("\n")[0]
        predictions.append((expected_answer, predicted))

        print(f"Actual Answer: {expected_answer}\n")
        print(f"Predicted Answer: {predicted}\n")
        print("-" * 50 + "\n")

    metric = evaluate.load("squad")
    references = [{"id": str(i), "answers": {"text": [gt], "answer_start": [0]}} for i, (gt, _) in enumerate(predictions)]
    pred_list = [{"id": str(i), "prediction_text": pred} for i, (_, pred) in enumerate(predictions)]

    results = metric.compute(predictions=pred_list, references=references)
    print(f"Evaluation Results for {model_path}:\n", results)

if __name__ == "__main__":
    with open("evaluate_before_finetuning.txt", "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        try:
            evaluate_model("gpt2")
        finally:
            sys.stdout = original_stdout
