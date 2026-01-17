import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)).replace('/scripts', ''))

import torch
import tiktoken
from model.transformer import LLM
from config.config import *

def test_model():
    print("🧪 Testing trained model...")
    
    # Check if model exists
    if not os.path.exists('model/model.pth'):
        print("❌ Model not found! Train first with: python scripts/train.py")
        return
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "Deep neural networks",
        "Natural language processing",
        "Computer vision can"
    ]
    
    print("\n🎯 Testing text generation:")
    print("=" * 50)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: '{prompt}'")
            
            tokens = tokenizer.encode(prompt)
            tokens = torch.tensor(tokens).unsqueeze(0)
            
            # Generate 30 tokens
            for _ in range(30):
                if tokens.size(1) >= MAX_SEQ_LEN:
                    break
                logits = model(tokens)
                next_token = torch.argmax(logits[0, -1, :])
                tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            generated = tokenizer.decode(tokens[0].tolist())
            print(f"Generated: {generated}")
            print("-" * 30)
    
    print("\n✅ Model testing completed!")
    print("\n💡 To generate custom text:")
    print("   python scripts/generate.py")

if __name__ == "__main__":
    test_model()
