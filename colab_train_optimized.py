# Optimized training script for T4 GPU
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

def train_optimized():
    print("🚀 Starting OPTIMIZED Hisu LLM Training for T4 GPU...")
    print(f"⚡ Config: {EPOCHS} epochs, {DATASET_LIMIT:,} tokens, effective batch={BATCH_SIZE * GRADIENT_ACCUMULATION}")
    
    # Enable mixed precision for speed
    scaler = torch.cuda.amp.GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {device}")
    
    data_path = "/content/Hisu/llm/data"
    dataset = TextDataset(data_path, MAX_SEQ_LEN)
    
    if len(dataset) == 0:
        print("❌ Dataset is empty!")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    model = LLM(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, MAX_SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Dataset samples: {len(dataset):,}")
    print(f"✅ Batches per epoch: {len(dataloader):,}")
    
    # Time estimation
    estimated_time_per_epoch = len(dataloader) / 25 / 60  # 25 batch/sec estimate
    total_estimated = estimated_time_per_epoch * EPOCHS
    print(f"⏱️ Estimated time per epoch: {estimated_time_per_epoch:.1f} mins")
    print(f"⏱️ Total estimated time: {total_estimated:.1f} mins ({total_estimated/60:.1f} hours)")
    print("-" * 50)
    
    start_time = time.time()
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        total_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / GRADIENT_ACCUMULATION
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION
            
            if batch_idx % 500 == 0:  # Less frequent logging
                elapsed = time.time() - start_time
                progress = ((epoch * len(dataloader) + batch_idx) / (EPOCHS * len(dataloader))) * 100
                batches_per_sec = (epoch * len(dataloader) + batch_idx) / elapsed if elapsed > 0 else 0
                remaining_time = ((EPOCHS * len(dataloader)) - (epoch * len(dataloader) + batch_idx)) / batches_per_sec / 60 if batches_per_sec > 0 else 0
                
                print(f'Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item() * GRADIENT_ACCUMULATION:.4f}')
                print(f'Progress: {progress:.1f}% | Speed: {batches_per_sec:.1f} batch/s | ETA: {remaining_time:.1f} mins')
                print("-" * 30)
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        
        print(f'✅ Epoch {epoch+1} completed in {epoch_time/60:.1f} mins | Avg Loss: {avg_loss:.4f}')
        print("=" * 50)
    
    total_time = time.time() - start_time
    print(f'🎉 Training completed in {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)!')
    
    # Save model
    os.makedirs('/content/Hisu/llm/model', exist_ok=True)
    torch.save(model.state_dict(), '/content/Hisu/llm/model/model.pth')
    print("💾 Model saved to /content/Hisu/llm/model/model.pth")

# Run optimized training
train_optimized()
