"""Text preprocessing and tokenization"""

import numpy as np
import re
from typing import List, Dict, Optional


class TextProcessor:
    """Text preprocessing and feature extraction"""

    def __init__(self, max_vocab_size: int = 10000, max_length: int = 512):
        self.max_vocab_size = max_vocab_size
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.word_counts = {}
        self.is_fitted = False

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = self.clean_text(text)
        return text.split()

    def fit(self, texts: List[str]) -> 'TextProcessor':
        """Build vocabulary from texts"""
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        idx = len(self.vocab)
        for word, count in sorted_words[:self.max_vocab_size - len(self.vocab)]:
            self.vocab[word] = idx
            idx += 1

        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to sequences of indices"""
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Call fit() first.")

        sequences = []

        for text in texts:
            tokens = self.tokenize(text)
            indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

            if len(indices) < self.max_length:
                indices += [self.vocab['<PAD>']] * (self.max_length - len(indices))
            else:
                indices = indices[:self.max_length]

            sequences.append(indices)

        return np.array(sequences)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform texts"""
        return self.fit(texts).transform(texts)

    def inverse_transform(self, sequences: np.ndarray) -> List[str]:
        """Transform sequences back to texts"""
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Call fit() first.")

        idx_to_word = {idx: word for word, idx in self.vocab.items()}
        texts = []

        for sequence in sequences:
            words = [idx_to_word.get(idx, '<UNK>') for idx in sequence if idx != self.vocab['<PAD>']]
            texts.append(' '.join(words))

        return texts

    def tfidf_vectorize(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF vectors"""
        n_docs = len(texts)

        doc_frequencies = {}
        term_frequencies = []

        for text in texts:
            tokens = self.tokenize(text)
            token_counts = {}

            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            term_frequencies.append(token_counts)

            for token in set(tokens):
                doc_frequencies[token] = doc_frequencies.get(token, 0) + 1

        vocab_list = sorted(doc_frequencies.keys())
        vocab_idx = {word: idx for idx, word in enumerate(vocab_list)}

        tfidf_matrix = np.zeros((n_docs, len(vocab_list)))

        for doc_idx, tf_dict in enumerate(term_frequencies):
            for word, count in tf_dict.items():
                if word in vocab_idx:
                    tf = count / len(tf_dict)
                    idf = np.log(n_docs / (doc_frequencies[word] + 1))
                    tfidf_matrix[doc_idx, vocab_idx[word]] = tf * idf

        return tfidf_matrix

    def n_gram_features(self, texts: List[str], n: int = 2) -> List[List[str]]:
        """Extract n-gram features"""
        n_grams = []

        for text in texts:
            tokens = self.tokenize(text)
            doc_n_grams = []

            for i in range(len(tokens) - n + 1):
                n_gram = ' '.join(tokens[i:i + n])
                doc_n_grams.append(n_gram)

            n_grams.append(doc_n_grams)

        return n_grams
