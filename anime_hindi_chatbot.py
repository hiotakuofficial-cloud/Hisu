"""
Anime Hindi Conversational Chatbot
Interactive interface for anime recommendations and discussions in Hindi/English
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_downloader import DatasetDownloader
from src.models.large_language_model import create_5b_model
from src.preprocessing.hindi_tokenizer import MultilingualTokenizer, ConversationalInterface
from src.training.anime_trainer import AnimeLanguageTrainer, DatasetSplitter


class AnimeHindiChatbot:
    """Conversational AI for anime recommendations with Hindi support"""

    def __init__(self):
        print("\n" + "="*70)
        print("  ANIME HINDI CHATBOT - 5B Parameter Multilingual Model")
        print("="*70 + "\n")

        self.dataset_downloader = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.conversational_interface = None

        # Conversation templates
        self.templates = {
            'greeting_hindi': 'नमस्ते! मैं एनीमे सहायक हूं। मैं आपकी कैसे मदद कर सकता हूं?',
            'greeting_english': 'Hello! I am your anime assistant. How can I help you?',
            'recommendation_hindi': 'मैं आपको {anime} देखने की सिफारिश करता हूं। यह {genre} शैली का है।',
            'recommendation_english': 'I recommend you watch {anime}. It is a {genre} anime.',
        }

    def initialize(self):
        """Initialize all components"""
        print("Step 1: Loading Datasets...")
        print("-" * 70)

        # Initialize dataset downloader
        self.dataset_downloader = DatasetDownloader(cache_dir="data/raw")

        # Create sample datasets
        anime_df, hindi_df = self.dataset_downloader.create_comprehensive_training_dataset()

        print(f"\n✓ Anime Dataset: {len(anime_df)} records")
        print(f"  Columns: {', '.join(anime_df.columns[:5])}...")
        print(f"\n✓ Hindi Translation Dataset: {len(hindi_df)} records")
        print(f"  Sample: {hindi_df.iloc[0]['english']} -> {hindi_df.iloc[0]['hindi']}")

        # Step 2: Build Tokenizer
        print("\n\nStep 2: Building Multilingual Tokenizer...")
        print("-" * 70)

        # Collect all texts for vocabulary
        all_texts = []
        all_texts.extend(anime_df['title'].tolist())
        all_texts.extend(anime_df['synopsis'].tolist())
        all_texts.extend(anime_df['synopsis_hindi'].tolist())
        all_texts.extend(hindi_df['english'].tolist())
        all_texts.extend(hindi_df['hindi'].tolist())

        self.tokenizer = MultilingualTokenizer(vocab_size=50000)
        self.tokenizer.build_vocab(all_texts, min_frequency=1)

        print(f"✓ Vocabulary size: {self.tokenizer.get_vocab_size()}")
        print(f"✓ Special tokens: {list(self.tokenizer.special_tokens.keys())}")

        # Test tokenization
        test_text = "मैं एनीमे देखना पसंद करता हूं। I love watching anime."
        tokens = self.tokenizer.tokenize(test_text)
        encoded = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(encoded)

        print(f"\nTokenization Test:")
        print(f"  Original: {test_text}")
        print(f"  Tokens: {tokens[:10]}...")
        print(f"  Token IDs: {encoded[:10]}...")
        print(f"  Decoded: {decoded}")

        # Step 3: Create Model
        print("\n\nStep 3: Initializing 5B Parameter Language Model...")
        print("-" * 70)

        self.model = create_5b_model()
        model_info = self.model.get_model_size()

        print(f"\nModel Architecture:")
        print(f"  Total Parameters: {model_info['total_parameters']:,}")
        print(f"  Parameters (B): {model_info['parameters_billions']:.2f}B")
        print(f"  Hidden Size: {model_info['d_model']}")
        print(f"  Layers: {model_info['n_layers']}")
        print(f"  Attention Heads: {model_info['n_heads']}")
        print(f"  Feed-forward Size: {model_info['d_ff']}")
        print(f"  Max Sequence Length: {model_info['max_seq_length']}")
        print(f"  Vocabulary Size: {model_info['vocab_size']}")

        # Step 4: Initialize Trainer
        print("\n\nStep 4: Preparing Training Pipeline...")
        print("-" * 70)

        self.trainer = AnimeLanguageTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            learning_rate=1e-4,
            batch_size=4,
            max_seq_length=256,
            warmup_steps=100,
        )

        # Prepare training data
        training_examples = self.trainer.prepare_anime_dataset(anime_df, hindi_df)

        # Split dataset
        train_examples, val_examples, test_examples = DatasetSplitter.split(
            training_examples,
            train_ratio=0.8,
            val_ratio=0.1
        )

        # Step 5: Train Model (demo with 1 epoch)
        print("\n\nStep 5: Training Model (Demo Mode)...")
        print("-" * 70)

        self.trainer.train(
            training_examples=train_examples[:40],  # Small subset for demo
            num_epochs=1,
            save_dir="models/checkpoints",
            log_interval=5,
        )

        # Get training stats
        stats = self.trainer.get_training_stats()
        print(f"\nTraining Statistics:")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Final Loss: {stats['final_loss']:.4f}")
        print(f"  Final Perplexity: {stats['final_perplexity']:.2f}")

        # Step 6: Initialize Conversational Interface
        print("\n\nStep 6: Initializing Conversational Interface...")
        print("-" * 70)

        self.conversational_interface = ConversationalInterface(self.tokenizer)

        print("✓ Chatbot ready for interaction!")

        return anime_df, hindi_df

    def process_query(self, query: str, include_hindi: bool = True) -> Dict:
        """Process user query and generate response"""
        # Analyze query
        result = self.conversational_interface.process_query(query)

        print(f"\nQuery Analysis:")
        print(f"  Detected Language: {result['detected_language']}")
        print(f"  Token Count: {result['token_count']}")
        print(f"  Language Hint: {result['language_hint']}")

        # Generate response (simplified demo)
        if 'नमस्ते' in query or 'hello' in query.lower():
            if result['detected_language'] == 'hindi':
                response = self.templates['greeting_hindi']
            else:
                response = self.templates['greeting_english']
        elif 'recommend' in query.lower() or 'सिफारिश' in query:
            response = "Based on your preferences, I recommend:\n"
            response += "1. Attack on Titan (進撃の巨人) - Action/Dark Fantasy\n"
            response += "2. Your Name (君の名は) - Romance/Drama\n"
            response += "3. Demon Slayer (鬼滅の刃) - Action/Supernatural\n"
            if include_hindi:
                response += "\n[हिंदी में] आपकी पसंद के आधार पर, मैं सिफारिश करता हूं..."
        else:
            response = "I understand your query. "
            if include_hindi:
                response += "(मैं आपकी बात समझता हूं।)"

        result['response'] = response
        return result

    def interactive_mode(self):
        """Run interactive chat mode"""
        print("\n" + "="*70)
        print("  INTERACTIVE MODE")
        print("="*70)
        print("\nCommands:")
        print("  'exit' or 'quit' - Exit chatbot")
        print("  'hindi' - Switch to Hindi mode")
        print("  'english' - Switch to English mode")
        print("  'help' - Show help")
        print("\nStart chatting! (Type in English or Hindi)\n")

        include_hindi = True

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nBot: Goodbye! अलविदा!")
                    break

                if user_input.lower() == 'hindi':
                    include_hindi = True
                    print("\nBot: Hindi mode enabled. हिंदी मोड सक्षम है।")
                    continue

                if user_input.lower() == 'english':
                    include_hindi = False
                    print("\nBot: English mode enabled.")
                    continue

                if user_input.lower() == 'help':
                    print("\nBot: I can help you with:")
                    print("  - Anime recommendations (एनीमे सिफारिशें)")
                    print("  - Anime information (एनीमे जानकारी)")
                    print("  - Genre-based search (शैली-आधारित खोज)")
                    print("  - Character discussions (चरित्र चर्चा)")
                    continue

                # Process query
                result = self.process_query(user_input, include_hindi)

                print(f"\nBot: {result['response']}")

            except KeyboardInterrupt:
                print("\n\nBot: Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    def demo_queries(self):
        """Run demo queries"""
        print("\n" + "="*70)
        print("  DEMO QUERIES")
        print("="*70 + "\n")

        demo_queries = [
            "नमस्ते! मुझे एक्शन एनीमे पसंद है।",
            "Hello! Can you recommend some anime?",
            "मुझे फंतासी एनीमे के बारे में बताएं",
            "What are the top rated anime?",
            "एनीमे की सिफारिश करें",
        ]

        for i, query in enumerate(demo_queries, 1):
            print(f"\nDemo Query {i}:")
            print(f"User: {query}")

            result = self.process_query(query)

            print(f"Bot: {result['response']}")
            print("-" * 70)


def main():
    """Main function"""
    # Create chatbot
    chatbot = AnimeHindiChatbot()

    # Initialize
    anime_df, hindi_df = chatbot.initialize()

    # Show dataset info
    print("\n" + "="*70)
    print("  DATASET INFORMATION")
    print("="*70 + "\n")

    print("Anime Dataset Sample:")
    print(anime_df[['title', 'title_hindi', 'genre', 'rating']].head(3))

    print("\n\nHindi Translation Dataset Sample:")
    print(hindi_df[['english', 'hindi']].head(3))

    # Run demo queries
    chatbot.demo_queries()

    # Interactive mode
    print("\n\nWould you like to try interactive mode? (yes/no)")
    response = input(">> ").strip().lower()

    if response in ['yes', 'y']:
        chatbot.interactive_mode()

    print("\n" + "="*70)
    print("  SESSION COMPLETE")
    print("="*70)
    print("\nDataset Sources:")
    print("  - Anime: MyAnimeList, Kaggle datasets")
    print("  - Hindi: IIT Bombay Parallel Corpus, AI4Bharat Samanantar")
    print("\nModel Architecture:")
    print("  - 5B parameters transformer model")
    print("  - Multilingual support (Hindi + English)")
    print("  - Anime domain specialization")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
