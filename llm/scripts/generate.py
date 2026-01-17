import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace('/scripts', ''))

import torch
import tiktoken
from model.transformer import LLM
from config.config import *

def generate(prompt, max_length=100):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_length):
            if tokens.size(1) >= MAX_SEQ_LEN:
                break
            logits = model(tokens)
            next_token = torch.argmax(logits[0, -1, :])
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())

if __name__ == "__main__":
    prompt = input("Enter prompt: ")
    result = generate(prompt)
    print(f"\nGenerated text:\n{result}")
