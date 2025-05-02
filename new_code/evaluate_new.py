from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from datasets import load_dataset
import json
import torch
import sys
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, AutoModelForQuestionAnswering
import evaluate
from datasets import load_dataset


# Load the fine-tuned model
model_path = "/home/ubuntu/gpt2_qa_best/best_model_63.3825/"  # Replace with your actual path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.eval()
predictions=[]
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load SQuAD dataset
squad = load_dataset("squad", split="validation")

# QA prediction function
def extract_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get most probable start and end of answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_index:end_index]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Run on a few examples
for i in range(len(squad)):
    question = squad[i]["question"]
    context = squad[i]["context"]
    true_answer = squad[i]["answers"]["text"]
    
    predicted_answer = extract_answer(question, context)

    print(f"\nExample {i+1}")
    print("Question:", question)
    print("Predicted Answer:", predicted_answer)
    print("Ground Truth Answer(s):", true_answer)
    try:
        predictions.append((true_answer[0], predicted_answer))
    except:
        predictions.append(('', predicted_answer))
        print("exception")
        #print(f"Actual Answer: {expected_answer}\n")
        #print(f"Predicted Answer: {predicted}\n")
        #print("-" * 50 + "\n")

metric = evaluate.load("squad")
references = [{"id": str(i), "answers": {"text": [gt], "answer_start": [0]}} for i, (gt, _) in enumerate(predictions)]
pred_list = [{"id": str(i), "prediction_text": pred} for i, (_, pred) in enumerate(predictions)]
results = metric.compute(predictions=pred_list, references=references)
print(f"Evaluation Results for {model_path}:\n", results)

