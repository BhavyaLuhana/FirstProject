import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed 

MODEL_NAME = "gpt2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 

def load_model(model_name=MODEL_NAME, device=DEVICE):
    tokenizer = AutoTokenizer.from_pretrained (model_name) #tokenizer for the model
    # For GPT-2 ensure tokenizer has pad token (some older tokenizers don't)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained (model_name) #load the model
    model.to(device) #move the model to GPU or CPU
    model.eval() #set the model to evaluation mode
    return tokenizer, model

def generate(prompt: str, tokenizer=None, model=None, max_length=100, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95, seed=42):
    set_seed(seed) # set the random seed so results for no same search
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) #  prompt into numbers
    with torch.no_grad(): 
        out = model.generate( # ask the model to write text
        **inputs, # use the prompt as input
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature, # control how random the answer is
        num_return_sequences=num_return_sequences, 
        pad_token_id=tokenizer.eos_token_id # use end token for padding
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in out] # turn numbers back to text

if __name__ == "__main__": 
    tokenizer, model = load_model() 
    prompt = "In the near future, Artifical Intelligence will " 
    output = generate(prompt, tokenizer, model, max_length=100, num_return_sequences=3, temperature=0.9) #models write 3 answers
    for i, text in enumerate(output, 1): 
        print(f"Generated {i} ---\n{text}\n") 