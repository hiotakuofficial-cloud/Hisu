"""
Hindi Tokenizer Module
Handles tokenization for Hindi and English text (multilingual support)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter


class MultilingualTokenizer:
    """Tokenizer supporting both Hindi (Devanagari) and English text"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<SEP>': 4,  # Separator
            '<MASK>': 5,  # For masked language modeling
        }

        # Hindi-specific tokens
        self.hindi_special = {
            '<HINDI>': 6,
            '<ENGLISH>': 7,
            '<ANIME>': 8,
        }

        # Initialize with special tokens
        self.vocab.update(self.special_tokens)
        self.vocab.update(self.hindi_special)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        self.next_token_id = max(self.vocab.values()) + 1

    def _tokenize_devanagari(self, text: str) -> List[str]:
        """Tokenize Devanagari (Hindi) script text"""
        # Devanagari Unicode range: \u0900-\u097F
        tokens = []
        current_word = []

        for char in text:
            if '\u0900' <= char <= '\u097F':
                # Devanagari character
                current_word.append(char)
            elif char.isspace():
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)
            else:
                # Punctuation or other
                if current_word:
                    tokens.append(''.join(current_word))
                    current_word = []
                tokens.append(char)

        if current_word:
            tokens.append(''.join(current_word))

        return tokens

    def _tokenize_english(self, text: str) -> List[str]:
        """Tokenize English text using simple word-level tokenization"""
        # Split on whitespace and punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text.lower())
        return tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize multilingual text (auto-detects Hindi/English)"""
        tokens = []
        current_segment = []
        current_lang = None

        for char in text:
            if '\u0900' <= char <= '\u097F':
                # Hindi character
                if current_lang == 'english' and current_segment:
                    tokens.extend(self._tokenize_english(''.join(current_segment)))
                    current_segment = []
                current_lang = 'hindi'
                current_segment.append(char)
            elif char.isalpha():
                # English character
                if current_lang == 'hindi' and current_segment:
                    tokens.extend(self._tokenize_devanagari(''.join(current_segment)))
                    current_segment = []
                current_lang = 'english'
                current_segment.append(char)
            else:
                # Whitespace or punctuation
                if current_segment:
                    if current_lang == 'hindi':
                        tokens.extend(self._tokenize_devanagari(''.join(current_segment)))
                    else:
                        tokens.extend(self._tokenize_english(''.join(current_segment)))
                    current_segment = []

                if not char.isspace():
                    tokens.append(char)
                current_lang = None

        # Process remaining segment
        if current_segment:
            if current_lang == 'hindi':
                tokens.extend(self._tokenize_devanagari(''.join(current_segment)))
            else:
                tokens.extend(self._tokenize_english(''.join(current_segment)))

        return tokens

    def build_vocab(self, texts: List[str], min_frequency: int = 2):
        """Build vocabulary from training texts"""
        print(f"Building vocabulary from {len(texts)} texts...")

        # Tokenize all texts and count frequencies
        token_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counter.update(tokens)

        print(f"Found {len(token_counter)} unique tokens")

        # Sort by frequency and take top tokens
        most_common = token_counter.most_common(self.vocab_size - self.next_token_id)

        # Add to vocabulary
        for token, freq in most_common:
            if freq >= min_frequency and token not in self.vocab:
                self.vocab[token] = self.next_token_id
                self.inverse_vocab[self.next_token_id] = token
                self.next_token_id += 1

        print(f"✓ Built vocabulary with {len(self.vocab)} tokens")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        token_ids = []

        if add_special_tokens:
            token_ids.append(self.special_tokens['<BOS>'])

        for token in tokens:
            token_id = self.vocab.get(token, self.special_tokens['<UNK>'])
            token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.special_tokens['<EOS>'])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []

        for token_id in token_ids:
            if skip_special_tokens and token_id in self.special_tokens.values():
                continue

            token = self.inverse_vocab.get(token_id, '<UNK>')
            tokens.append(token)

        # Join tokens
        text = ''.join(tokens)

        # Clean up spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, padding: bool = True) -> List[List[int]]:
        """Encode batch of texts with optional padding"""
        encoded = [self.encode(text) for text in texts]

        if max_length is None and padding:
            max_length = max(len(seq) for seq in encoded)

        if padding and max_length:
            for i in range(len(encoded)):
                if len(encoded[i]) < max_length:
                    encoded[i] = encoded[i] + [self.special_tokens['<PAD>']] * (max_length - len(encoded[i]))
                elif len(encoded[i]) > max_length:
                    encoded[i] = encoded[i][:max_length]

        return encoded

    def save(self, filepath: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'hindi_special': self.hindi_special,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved tokenizer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MultilingualTokenizer':
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = {k: int(v) for k, v in data['vocab'].items()}
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.next_token_id = max(tokenizer.vocab.values()) + 1

        print(f"✓ Loaded tokenizer from {filepath}")
        return tokenizer

    def get_vocab_size(self) -> int:
        """Get actual vocabulary size"""
        return len(self.vocab)

    def translate_prompt(self, text: str, source_lang: str = 'auto') -> Tuple[str, str]:
        """
        Provide translation hints for Hindi-English text
        Returns: (detected_language, translation_hint)
        """
        # Simple language detection
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return 'unknown', text

        hindi_ratio = hindi_chars / total_chars

        if hindi_ratio > 0.5:
            detected = 'hindi'
            hint = "Detected Hindi text"
        elif hindi_ratio > 0:
            detected = 'mixed'
            hint = "Detected mixed Hindi-English text"
        else:
            detected = 'english'
            hint = "Detected English text"

        return detected, hint


class ConversationalInterface:
    """Interface for Hindi-English conversational AI"""

    def __init__(self, tokenizer: MultilingualTokenizer):
        self.tokenizer = tokenizer
        self.conversation_history = []

        # Common anime-related phrases in Hindi
        self.hindi_phrases = {
            'greeting': [
                'नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?',
                'स्वागत है! एनीमे के बारे में क्या जानना चाहते हैं?',
            ],
            'recommendation': [
                'मैं आपको यह एनीमे देखने की सलाह देता हूं:',
                'यह एनीमे आपको पसंद आ सकती है:',
            ],
            'description': [
                'यह एनीमे के बारे में है:',
                'कहानी का सारांश:',
            ],
        }

    def format_response(self, text: str, include_hindi: bool = True) -> str:
        """Format response with both English and Hindi"""
        if include_hindi:
            return f"{text}\n\n(Hindi: {self._quick_translate(text)})"
        return text

    def _quick_translate(self, text: str) -> str:
        """Provide basic translation hints"""
        # This is a placeholder - in production, use proper translation model
        common_translations = {
            'hello': 'नमस्ते',
            'anime': 'एनीमे',
            'recommend': 'सिफारिश करना',
            'watch': 'देखना',
            'favorite': 'पसंदीदा',
            'character': 'चरित्र',
            'story': 'कहानी',
        }

        for eng, hindi in common_translations.items():
            text = text.replace(eng, f"{eng}/{hindi}")

        return text

    def process_query(self, query: str) -> Dict:
        """Process user query and return structured response"""
        lang, hint = self.tokenizer.translate_prompt(query)

        tokens = self.tokenizer.encode(query)

        response = {
            'detected_language': lang,
            'language_hint': hint,
            'tokens': tokens,
            'token_count': len(tokens),
            'query': query,
        }

        return response


def create_anime_hindi_tokenizer(texts: List[str]) -> MultilingualTokenizer:
    """Create tokenizer specifically for anime and Hindi content"""
    tokenizer = MultilingualTokenizer(vocab_size=50000)
    tokenizer.build_vocab(texts, min_frequency=2)
    return tokenizer
