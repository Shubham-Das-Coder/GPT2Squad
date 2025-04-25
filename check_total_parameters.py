import torch
from transformers import GPT2LMHeadModel

def get_number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

def check_gpt2_parameters(model_path="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

    num_params = get_number_of_parameters(model)
    print(f"Number of parameters in {model_path}: {num_params:,}")

if __name__ == "__main__":
    check_gpt2_parameters("gpt2")
