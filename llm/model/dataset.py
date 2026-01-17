import torch
from torch.utils.data import Dataset
import tiktoken
import os

class TextDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.seq_len = seq_len
        
        # Load all text files
        all_text = ""
        files_loaded = 0
        total_chars = 0
        
        print("Loading datasets...")
        for filename in ["1.txt", "2.txt"]:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    file_size = len(file_content)
                    all_text += file_content + "\n"
                    files_loaded += 1
                    total_chars += file_size
                    print(f"✅ Loaded {filename}: {file_size:,} characters")
            else:
                print(f"❌ Missing {filename}")
        
        print(f"📊 Total files loaded: {files_loaded}/2")
        print(f"📊 Total characters: {total_chars:,}")
        
        self.tokens = self.tokenizer.encode(all_text)
        print(f"📊 Total tokens: {len(self.tokens):,}")
        print(f"📊 Vocab size: {self.tokenizer.n_vocab:,}")
        print("-" * 40)
        
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
