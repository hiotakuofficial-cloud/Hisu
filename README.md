# Hisu - Large Language Model

A transformer-based language model trained on anime dataset.

## Structure
```
llm/
├── config/          # Model configuration
├── model/           # Model architecture & dataset
├── data/            # Training datasets (1.txt, 2.txt)
└── scripts/         # Training & testing scripts
```

## Setup
```bash
python3 -m venv llm_env
source llm_env/bin/activate
pip install torch tiktoken numpy
```

## Usage
```bash
cd llm
python scripts/train.py    # Train model
python scripts/test.py     # Test model
python scripts/chat.py     # Interactive chat
```

## Model Specs
- Parameters: ~70M
- Architecture: Transformer decoder
- Context length: 512 tokens
- Training data: Anime descriptions
