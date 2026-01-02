"""
Anime + Hindi Language Model Training Pipeline
Specialized trainer for multilingual anime understanding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path


class AnimeLanguageTrainer:
    """Trainer for anime and Hindi language model"""

    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        max_seq_length: int = 512,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.current_step = 0
        self.training_history = {
            'loss': [],
            'learning_rate': [],
            'perplexity': [],
        }

    def prepare_anime_dataset(self, anime_df, hindi_df) -> List[Dict]:
        """Prepare training data from anime and Hindi datasets"""
        training_examples = []

        # Create anime description examples
        for _, row in anime_df.iterrows():
            # English description
            example_en = {
                'text': f"Anime: {row['title']}. Genre: {row['genre']}. Synopsis: {row['synopsis']}",
                'language': 'english',
                'task': 'description'
            }
            training_examples.append(example_en)

            # Hindi description
            example_hi = {
                'text': f"एनीमे: {row['title_hindi']}. शैली: {row['genre']}. सारांश: {row['synopsis_hindi']}",
                'language': 'hindi',
                'task': 'description'
            }
            training_examples.append(example_hi)

            # Bilingual example
            example_bi = {
                'text': f"{row['title']} ({row['title_hindi']}): {row['synopsis']} | {row['synopsis_hindi']}",
                'language': 'mixed',
                'task': 'translation'
            }
            training_examples.append(example_bi)

        # Add conversational examples from Hindi dataset
        for _, row in hindi_df.iterrows():
            example = {
                'text': f"English: {row['english']} | Hindi: {row['hindi']}",
                'language': 'mixed',
                'task': 'conversation'
            }
            training_examples.append(example)

        print(f"✓ Prepared {len(training_examples)} training examples")
        return training_examples

    def create_training_batch(self, examples: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Create a batch of training data"""
        texts = [ex['text'] for ex in examples]

        # Tokenize and pad
        encoded = self.tokenizer.encode_batch(texts, max_length=self.max_seq_length, padding=True)

        # Convert to numpy arrays
        input_ids = np.array(encoded)

        # Create labels (shift input_ids by 1 for language modeling)
        labels = np.roll(input_ids, -1, axis=1)
        labels[:, -1] = self.tokenizer.special_tokens['<PAD>']

        return input_ids, labels

    def compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Compute cross-entropy
        # Simplified - in practice use proper cross-entropy with numerical stability
        loss = 0.0
        valid_tokens = 0

        for i in range(len(labels_flat)):
            if labels_flat[i] != self.tokenizer.special_tokens['<PAD>']:
                # Get prediction for this token
                logit = logits_flat[i]
                true_label = labels_flat[i]

                # Apply softmax
                exp_logits = np.exp(logit - np.max(logit))
                probs = exp_logits / np.sum(exp_logits)

                # Compute negative log likelihood
                loss += -np.log(probs[true_label] + 1e-10)
                valid_tokens += 1

        return loss / valid_tokens if valid_tokens > 0 else 0.0

    def train_step(self, input_ids: np.ndarray, labels: np.ndarray) -> Dict:
        """Single training step"""
        # Forward pass
        logits = self.model.forward(input_ids)

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Compute perplexity
        perplexity = np.exp(loss)

        # Learning rate schedule with warmup
        if self.current_step < self.warmup_steps:
            lr = self.learning_rate * (self.current_step / self.warmup_steps)
        else:
            lr = self.learning_rate

        self.current_step += 1

        return {
            'loss': loss,
            'perplexity': perplexity,
            'learning_rate': lr,
        }

    def train(
        self,
        training_examples: List[Dict],
        num_epochs: int = 3,
        save_dir: str = "models/checkpoints",
        log_interval: int = 10,
    ):
        """Train the model"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"{'='*60}")
        print(f"Model Parameters: {self.model.config.total_params:,} (~{self.model.config.total_params/1e9:.2f}B)")
        print(f"Training Examples: {len(training_examples)}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Max Sequence Length: {self.max_seq_length}")
        print(f"Number of Epochs: {num_epochs}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Warmup Steps: {self.warmup_steps}")
        print(f"{'='*60}\n")

        total_steps = (len(training_examples) // self.batch_size) * num_epochs
        print(f"Total Training Steps: {total_steps}\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            epoch_loss = 0.0
            epoch_steps = 0

            # Shuffle examples
            np.random.shuffle(training_examples)

            # Create batches
            for i in range(0, len(training_examples), self.batch_size):
                batch_examples = training_examples[i:i + self.batch_size]

                if len(batch_examples) < self.batch_size:
                    continue

                # Create batch
                input_ids, labels = self.create_training_batch(batch_examples)

                # Training step
                metrics = self.train_step(input_ids, labels)

                epoch_loss += metrics['loss']
                epoch_steps += 1

                # Log progress
                if epoch_steps % log_interval == 0:
                    avg_loss = epoch_loss / epoch_steps
                    elapsed = time.time() - start_time
                    print(f"  Step {epoch_steps}: Loss={avg_loss:.4f}, "
                          f"Perplexity={metrics['perplexity']:.2f}, "
                          f"LR={metrics['learning_rate']:.6f}, "
                          f"Time={elapsed:.1f}s")

                # Update history
                self.training_history['loss'].append(metrics['loss'])
                self.training_history['learning_rate'].append(metrics['learning_rate'])
                self.training_history['perplexity'].append(metrics['perplexity'])

            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_steps
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Average Perplexity: {np.exp(avg_epoch_loss):.2f}")
            print()

            # Save checkpoint
            checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.npy"
            print(f"  Saved checkpoint: {checkpoint_path}\n")

        total_time = time.time() - start_time
        print(f"{'='*60}")
        print(f"Training Complete!")
        print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Final Loss: {self.training_history['loss'][-1]:.4f}")
        print(f"Final Perplexity: {self.training_history['perplexity'][-1]:.2f}")
        print(f"{'='*60}\n")

    def evaluate(self, test_examples: List[Dict]) -> Dict:
        """Evaluate model on test set"""
        print("\nEvaluating model...")

        total_loss = 0.0
        total_batches = 0

        for i in range(0, len(test_examples), self.batch_size):
            batch_examples = test_examples[i:i + self.batch_size]

            if len(batch_examples) < self.batch_size:
                continue

            # Create batch
            input_ids, labels = self.create_training_batch(batch_examples)

            # Forward pass
            logits = self.model.forward(input_ids)

            # Compute loss
            loss = self.compute_loss(logits, labels)

            total_loss += loss
            total_batches += 1

        avg_loss = total_loss / total_batches
        perplexity = np.exp(avg_loss)

        print(f"✓ Evaluation Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
        }

    def generate_response(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7
    ) -> str:
        """Generate response from prompt"""
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = np.array([input_ids])

        # Generate
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature
        )

        # Decode
        response = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        return response

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        if not self.training_history['loss']:
            return {}

        return {
            'total_steps': len(self.training_history['loss']),
            'final_loss': self.training_history['loss'][-1],
            'best_loss': min(self.training_history['loss']),
            'final_perplexity': self.training_history['perplexity'][-1],
            'best_perplexity': min(self.training_history['perplexity']),
        }


class DatasetSplitter:
    """Split dataset into train/validation/test sets"""

    @staticmethod
    def split(examples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split examples into train/val/test"""
        np.random.shuffle(examples)

        n = len(examples)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)

        train = examples[:train_size]
        val = examples[train_size:train_size + val_size]
        test = examples[train_size + val_size:]

        print(f"Dataset split:")
        print(f"  Train: {len(train)} examples ({len(train)/n*100:.1f}%)")
        print(f"  Validation: {len(val)} examples ({len(val)/n*100:.1f}%)")
        print(f"  Test: {len(test)} examples ({len(test)/n*100:.1f}%)")

        return train, val, test
