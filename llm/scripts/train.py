import sys
import os
# Add llm directory to path for imports
sys.path.append('/content/Hisu/llm')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.transformer import LLM
from model.dataset import TextDataset
from config.config import *
import time

def train():
    dataset = TextDataset("data", MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset):,} samples")
    print(f"Total batches per epoch: {len(dataloader):,}")
    print(f"Training will process: {len(dataloader) * EPOCHS:,} total batches")
    print("-" * 50)
    
    start_time = time.time()
    total_batches = len(dataloader) * EPOCHS
    processed_batches = 0
    
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
            processed_batches += 1
            
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                progress = (processed_batches / total_batches) * 100
                batches_per_sec = processed_batches / elapsed if elapsed > 0 else 0
                remaining_batches = total_batches - processed_batches
                eta_seconds = remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                eta_mins = eta_seconds / 60
                
                print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * GRADIENT_ACCUMULATION:.4f}')
                print(f'Progress: {progress:.1f}% | Speed: {batches_per_sec:.1f} batch/s | ETA: {eta_mins:.1f} mins')
                print("-" * 30)
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        elapsed_total = time.time() - start_time
        
        print(f'✅ Epoch {epoch+1} completed in {epoch_time/60:.1f} mins')
        print(f'Average loss: {avg_loss:.4f}')
        print(f'Total elapsed: {elapsed_total/60:.1f} mins')
        print("=" * 50)
    
    total_time = time.time() - start_time
    print(f'🎉 Training completed in {total_time/60:.1f} minutes!')
    
    torch.save(model.state_dict(), 'model/model.pth')
    print("Model saved to model/model.pth")

if __name__ == "__main__":
    train()
