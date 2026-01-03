# Analazy - Anime Knowledge AI Chatbot

A sophisticated AI chatbot specialized in anime knowledge, supporting both Hindi and English conversations with natural, contextual responses.

## Project Structure

```
analazy/
├── src/                          # Core source code
│   ├── models/                   # Neural network architectures
│   ├── data/                     # Data processing utilities
│   ├── preprocessing/            # Text preprocessing & tokenization
│   ├── training/                 # Training pipeline
│   ├── evaluation/               # Model evaluation metrics
│   ├── utils/                    # Helper utilities
│   └── visualization/            # Training visualization
├── training_data/                # High-quality training datasets
│   ├── conversational_training.csv    # Natural conversation patterns
│   ├── anime_knowledge.csv            # Anime facts and information
│   ├── reasoning_responses.csv        # Complex analytical responses
│   ├── hindi_english_mix.csv          # Hinglish conversational data
│   ├── instruction_following.csv      # Task-specific instructions
│   └── advanced_queries.csv           # Expert-level discussions
├── models/                       # Trained model checkpoints
│   └── checkpoints/              # Training checkpoints
├── anime_hindi_chatbot.py        # Main chatbot interface
├── main.py                       # Entry point
└── setup.py                      # Package configuration

```

## Dataset Information

All datasets are manually curated for quality and natural language patterns:

- **conversational_training.csv**: 20 entries of greeting, recommendation, and general chat patterns
- **anime_knowledge.csv**: 20 entries covering anime facts, characters, and series information
- **reasoning_responses.csv**: 20 entries with detailed analytical responses
- **hindi_english_mix.csv**: 20 entries demonstrating natural Hinglish conversation style
- **instruction_following.csv**: 20 entries for structured task responses
- **advanced_queries.csv**: 19 entries for expert-level philosophical and technical discussions

Total: **119 high-quality training examples** focusing on natural, non-hallucinated responses.

## Features

- **Bilingual Support**: Natural Hindi and English understanding
- **Hinglish Fluency**: Seamless code-switching in conversations
- **Anime Expertise**: Deep knowledge across genres, series, characters
- **Natural Responses**: Non-robotic, contextual conversation flow
- **Low Hallucination**: Trained on verified, accurate information
- **Reasoning Capability**: Analytical and comparative discussions

## Training Philosophy

- **Quality over Quantity**: Manually written datasets ensure accuracy
- **Natural Language**: Responses mimic human conversation patterns
- **Context Awareness**: Understanding user intent and conversation flow
- **Fact-Based**: Grounded in actual anime knowledge, no fabrication
- **Cultural Sensitivity**: Respects both Indian and Japanese cultural contexts

## Usage

```python
python main.py
```

## Tech Stack

- Pure Python implementation
- Custom neural network architecture
- Efficient tokenization for multilingual support
- Scalable training pipeline
