# Simple training script for Google Colab
# Run this directly in a Colab cell

import sys
sys.path.append('/content/Hisu/llm')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.transformer import LLM
from model.dataset import TextDataset
from config.config import *
import time
import os

def train_model():
    print("🚀 Starting Hisu LLM Training...")
    
    # Use absolute path for data directory
    data_path = "/content/Hisu/llm/data"
    
    # Check if data files exist
    if not os.path.exists(f"{data_path}/1.txt") or not os.path.exists(f"{data_path}/2.txt"):
        print("❌ Data files missing!")
        print("Make sure 1.txt and 2.txt are in /content/Hisu/llm/data/")
        return
    
    dataset = TextDataset(data_path, MAX_SEQ_LEN)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Dataset size: {len(dataset):,} samples")
    print(f"✅ Total batches per epoch: {len(dataloader):,}")
    print("-" * 50)
    
    start_time = time.time()
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / GRADIENT_ACCUMULATION
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                progress = ((epoch * len(dataloader) + batch_idx) / (EPOCHS * len(dataloader))) * 100
                print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * GRADIENT_ACCUMULATION:.4f} | Progress: {progress:.1f}%')
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        
        print(f'✅ Epoch {epoch+1} completed in {epoch_time/60:.1f} mins | Avg Loss: {avg_loss:.4f}')
    
    total_time = time.time() - start_time
    print(f'🎉 Training completed in {total_time/60:.1f} minutes!')
    
    # Save model
    os.makedirs('/content/Hisu/llm/model', exist_ok=True)
    torch.save(model.state_dict(), '/content/Hisu/llm/model/model.pth')
    print("💾 Model saved to /content/Hisu/llm/model/model.pth")

# Run training
train_model()
