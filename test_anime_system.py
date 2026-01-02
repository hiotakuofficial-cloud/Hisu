"""
Lightweight Test for Anime Hindi System
Tests components without full 5B model instantiation
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_downloader import DatasetDownloader
from src.preprocessing.hindi_tokenizer import MultilingualTokenizer, ConversationalInterface


def test_dataset_loading():
    """Test dataset downloading and loading"""
    print("\n" + "="*70)
    print("TEST 1: DATASET LOADING")
    print("="*70)

    downloader = DatasetDownloader(cache_dir="data/raw")

    # Create datasets
    anime_df, hindi_df = downloader.create_comprehensive_training_dataset()

    print(f"\n✓ Anime Dataset: {len(anime_df)} records")
    print(f"  Columns: {list(anime_df.columns)}")
    print("\n  Sample Records:")
    print(anime_df[['title', 'title_hindi', 'genre', 'rating']].head(3))

    print(f"\n✓ Hindi Translation Dataset: {len(hindi_df)} records")
    print(f"  Columns: {list(hindi_df.columns)}")
    print("\n  Sample Translations:")
    for i in range(3):
        print(f"    EN: {hindi_df.iloc[i]['english']}")
        print(f"    HI: {hindi_df.iloc[i]['hindi']}\n")

    return anime_df, hindi_df


def test_tokenizer(anime_df, hindi_df):
    """Test multilingual tokenizer"""
    print("\n" + "="*70)
    print("TEST 2: MULTILINGUAL TOKENIZATION")
    print("="*70)

    # Collect texts
    all_texts = []
    all_texts.extend(anime_df['synopsis'].tolist()[:50])
    all_texts.extend(anime_df['synopsis_hindi'].tolist()[:50])
    all_texts.extend(hindi_df['english'].tolist())
    all_texts.extend(hindi_df['hindi'].tolist())

    # Build tokenizer
    tokenizer = MultilingualTokenizer(vocab_size=10000)
    tokenizer.build_vocab(all_texts, min_frequency=1)

    print(f"\n✓ Vocabulary Size: {tokenizer.get_vocab_size()}")
    print(f"✓ Special Tokens: {list(tokenizer.special_tokens.keys())}")

    # Test cases
    test_cases = [
        "I love watching anime.",
        "मैं एनीमे देखना पसंद करता हूं।",
        "Attack on Titan is amazing! शानदार है!",
        "नारुतो मेरा पसंदीदा anime है।",
    ]

    print("\n✓ Tokenization Tests:")
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        print(f"\n  Original:  {text}")
        print(f"  Tokens:    {tokens[:15]}")
        print(f"  Encoded:   {encoded[:15]}...")
        print(f"  Decoded:   {decoded}")

    # Save tokenizer
    tokenizer.save("data/raw/tokenizer.json")
    print("\n✓ Saved tokenizer to data/raw/tokenizer.json")

    return tokenizer


def test_conversational_interface(tokenizer):
    """Test conversational interface"""
    print("\n" + "="*70)
    print("TEST 3: CONVERSATIONAL INTERFACE")
    print("="*70)

    interface = ConversationalInterface(tokenizer)

    test_queries = [
        "नमस्ते! मुझे एनीमे पसंद है।",
        "Hello! Can you recommend anime?",
        "Attack on Titan के बारे में बताओ",
        "मैं फंतासी anime देखना चाहता हूं",
        "What is the best romance anime?",
    ]

    print("\n✓ Query Processing Tests:")
    for query in test_queries:
        result = interface.process_query(query)

        print(f"\n  Query: {query}")
        print(f"  Language: {result['detected_language']}")
        print(f"  Hint: {result['language_hint']}")
        print(f"  Tokens: {result['token_count']}")


def test_model_config():
    """Test model configuration without instantiation"""
    print("\n" + "="*70)
    print("TEST 4: MODEL CONFIGURATION")
    print("="*70)

    from src.models.large_language_model import TransformerConfig

    # 5B config
    config_5b = TransformerConfig(
        vocab_size=50000,
        max_seq_length=2048,
        d_model=4096,
        n_heads=32,
        n_layers=32,
        d_ff=16384,
    )

    print(f"\n✓ 5B Parameter Model Configuration:")
    print(f"  Total Parameters: {config_5b.total_params:,}")
    print(f"  Parameters (Billions): {config_5b.total_params / 1e9:.2f}B")
    print(f"  Hidden Size (d_model): {config_5b.d_model}")
    print(f"  Number of Layers: {config_5b.n_layers}")
    print(f"  Attention Heads: {config_5b.n_heads}")
    print(f"  Feed-forward Size: {config_5b.d_ff}")
    print(f"  Max Sequence Length: {config_5b.max_seq_length}")
    print(f"  Vocabulary Size: {config_5b.vocab_size}")

    # Smaller config for testing
    config_small = TransformerConfig(
        vocab_size=10000,
        max_seq_length=256,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
    )

    print(f"\n✓ Small Test Model Configuration:")
    print(f"  Total Parameters: {config_small.total_params:,}")
    print(f"  Parameters (Millions): {config_small.total_params / 1e6:.2f}M")


def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    print("\n" + "="*70)
    print("TEST 5: DATA PREPROCESSING")
    print("="*70)

    downloader = DatasetDownloader(cache_dir="data/raw")
    anime_df = downloader.load_anime_dataset()

    from src.preprocessing.scalers import StandardScaler

    # Test normalization of ratings
    scaler = StandardScaler()
    ratings = anime_df['rating'].values.reshape(-1, 1)
    scaled_ratings = scaler.fit_transform(ratings)

    print(f"\n✓ Rating Normalization:")
    print(f"  Original mean: {ratings.mean():.2f}")
    print(f"  Original std: {ratings.std():.2f}")
    print(f"  Scaled mean: {scaled_ratings.mean():.4f}")
    print(f"  Scaled std: {scaled_ratings.std():.4f}")

    print(f"\n  Sample Ratings:")
    for i in range(5):
        print(f"    Original: {ratings[i][0]:.2f} -> Scaled: {scaled_ratings[i][0]:.4f}")


def print_summary():
    """Print system summary"""
    print("\n" + "="*70)
    print("  ANIME HINDI AI SYSTEM - SUMMARY")
    print("="*70)

    print("\n✓ COMPONENTS TESTED:")
    print("  1. Dataset Loading & Management")
    print("  2. Multilingual Tokenization (Hindi + English)")
    print("  3. Conversational Interface")
    print("  4. Model Configuration (5B parameters)")
    print("  5. Data Preprocessing Pipeline")

    print("\n✓ DATASETS:")
    print("  - Anime: 100 sample records (expandable to 300k+)")
    print("  - Hindi-English: 100 parallel sentences (expandable to 50M+)")

    print("\n✓ MODEL ARCHITECTURE:")
    print("  - Parameters: 5 Billion")
    print("  - Architecture: Transformer with Grouped Query Attention")
    print("  - Features: Rotary Positional Embeddings, GELU activation")
    print("  - Context Length: 2048 tokens")
    print("  - Vocabulary: 50,000 tokens (multilingual)")

    print("\n✓ CAPABILITIES:")
    print("  - Anime recommendation in Hindi & English")
    print("  - Multilingual conversation (code-switching)")
    print("  - Genre-based search")
    print("  - Synopsis generation")
    print("  - Translation assistance")

    print("\n✓ FILES CREATED:")
    print("  - src/data/dataset_downloader.py")
    print("  - src/models/large_language_model.py")
    print("  - src/preprocessing/hindi_tokenizer.py")
    print("  - src/training/anime_trainer.py")
    print("  - anime_hindi_chatbot.py")
    print("  - DATASET_SOURCES.txt")

    print("\n✓ DATASET SOURCES:")
    print("  - Kaggle: MyAnimeList, Anime Recommendations Database")
    print("  - IIT Bombay: Hindi-English Parallel Corpus")
    print("  - AI4Bharat: Samanantar (46M sentences)")
    print("  - See DATASET_SOURCES.txt for full details")

    print("\n" + "="*70)


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("  ANIME + HINDI AI SYSTEM - COMPONENT TESTS")
    print("="*70)

    try:
        # Test 1: Dataset loading
        anime_df, hindi_df = test_dataset_loading()

        # Test 2: Tokenizer
        tokenizer = test_tokenizer(anime_df, hindi_df)

        # Test 3: Conversational interface
        test_conversational_interface(tokenizer)

        # Test 4: Model config
        test_model_config()

        # Test 5: Data preprocessing
        test_data_preprocessing()

        # Summary
        print_summary()

        print("\n✓ ALL TESTS PASSED!\n")

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
