import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace('/scripts', ''))

import torch
import tiktoken
from model.transformer import LLM
from config.config import *

def interactive_chat():
    print("🤖 LLM Interactive Chat")
    print("Type 'quit' to exit")
    print("=" * 40)
    
    if not os.path.exists('model/model.pth'):
        print("❌ Model not found! Train first.")
        return
    
    tokenizer = tiktoken.get_encoding("gpt2")
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    
    print("✅ Model loaded! Start chatting...")
    
    while True:
        prompt = input("\n👤 You: ")
        if prompt.lower() == 'quit':
            break
            
        tokens = tokenizer.encode(prompt)
        tokens = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(50):  # Generate 50 tokens
                if tokens.size(1) >= MAX_SEQ_LEN:
                    break
                logits = model(tokens)
                # Use temperature for more creative responses
                probs = torch.softmax(logits[0, -1, :] / 0.8, dim=-1)
                next_token = torch.multinomial(probs, 1)
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        
        response = tokenizer.decode(tokens[0].tolist())
        print(f"🤖 Bot: {response}")

if __name__ == "__main__":
    interactive_chat()
