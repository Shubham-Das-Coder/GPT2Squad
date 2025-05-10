import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GenerationConfig
import json
import os
import nltk
import evaluate

nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = "gpt2coqa_model"
json_path = "data/modified/coqa_dev.json"

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()

gen_config_path = os.path.join(model_dir, "generation_config.json")
if os.path.exists(gen_config_path):
    with open(gen_config_path, "r") as f:
        gen_config = GenerationConfig.from_dict(json.load(f))
else:
    # Default config in case generation_config.json is missing
    gen_config = GenerationConfig(max_new_tokens=50, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

def answer_question(story: str, question: str) -> str:
    prompt = f"Story: {story.strip()}\nQuestion: {question.strip()}\nAnswer:"
    input_ids = tokenizer.encode(prompt, truncation=True, max_length=1024)
    input_ids = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=gen_config.max_new_tokens,
            do_sample=gen_config.do_sample,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted_answer = decoded_output[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
    return predicted_answer

def load_all_samples(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found at: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []
    if isinstance(data, list):
        for item in data:
            story = item.get("story", "")
            question = item.get("question", "")
            answer = item.get("answer", "")
            if story and question and answer:
                samples.append((story, question, answer))
    else:
        raise ValueError("Unsupported JSON structure. Expected a list of items.")
    return samples

if __name__ == "__main__":
    samples = load_all_samples(json_path)

    if len(samples) == 0:
        print(f"No samples found in {json_path}. Please check your dataset.")
        exit(1)

    metric = evaluate.load("squad")
    predictions = []

    for idx, (story, question, ground_truth) in enumerate(samples):
        try:
            pred = answer_question(story, question)
        except Exception as e:
            print(f"[{idx+1}] Error: {e}")
            pred = ""

        print(f"\nExample {idx + 1}")
        print("Question:", question)
        print("Predicted Answer:", pred)
        print("Ground Truth Answer:", ground_truth)

        predictions.append((
            {"id": str(idx), "answers": {"text": [ground_truth], "answer_start": [0]}},
            {"id": str(idx), "prediction_text": pred}
        ))

    references = [ref for ref, _ in predictions]
    pred_list = [pred for _, pred in predictions]

    results = metric.compute(predictions=pred_list, references=references)

    print(f"\nEvaluation Results for {model_dir}:\n", results)

    os.makedirs("logs", exist_ok=True)
    output_path = os.path.join("logs", "gpt2coqa_evaluation.txt")

    with open(output_path, "w") as f:
        f.write(f"Evaluation Results for {model_dir}:\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    print(f"\nResults written to {output_path}")
