import json

with open("data/original/CoQA/coqa-train-v1.0.json", "r") as f:
    data = json.load(f)

new_data = []

for entry in data["data"]:
    story = entry["story"]
    questions = entry["questions"]
    answers = entry["answers"]

    for i in range(min(2, len(questions))):
        qa_entry = {
            "id": f"{entry['id']}_q{i+1}",
            "story": story,
            "question": questions[i]["input_text"],
            "answer": answers[i]["input_text"]
        }
        new_data.append(qa_entry)

with open("data/modified/coqa_train.json", "w") as f:
    json.dump(new_data, f, indent=2)
