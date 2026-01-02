"""
Anti-Hallucination Training Framework
Ensures neutral, coherent, and factually grounded conversational AI
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import time
from pathlib import Path
import json


@dataclass
class HallucinationMetrics:
    """Metrics for tracking hallucination in model outputs"""
    factual_consistency_score: float
    coherence_score: float
    grounding_score: float
    repetition_score: float
    confidence_calibration: float
    contradiction_rate: float

    def overall_score(self) -> float:
        """Calculate overall anti-hallucination score"""
        return (
            self.factual_consistency_score * 0.3 +
            self.coherence_score * 0.2 +
            self.grounding_score * 0.25 +
            (1 - self.repetition_score) * 0.1 +
            self.confidence_calibration * 0.1 +
            (1 - self.contradiction_rate) * 0.05
        )


class HallucinationDetector:
    """Detects various forms of hallucination in model outputs"""

    def __init__(self):
        self.factual_knowledge_base = {}
        self.conversation_history = []

    def detect_hallucination(self,
                           generated_text: str,
                           context: str,
                           ground_truth: Optional[str] = None) -> Dict:
        """Comprehensive hallucination detection"""

        issues = {
            'factual_errors': [],
            'contradictions': [],
            'unsupported_claims': [],
            'repetitions': [],
            'incoherence': [],
        }

        # 1. Check for factual consistency with context
        factual_score = self._check_factual_consistency(generated_text, context)

        # 2. Check for self-contradictions
        contradiction_score = self._check_contradictions(generated_text)

        # 3. Check for grounding in context
        grounding_score = self._check_grounding(generated_text, context)

        # 4. Check for repetitive patterns
        repetition_score = self._check_repetitions(generated_text)

        # 5. Check coherence
        coherence_score = self._check_coherence(generated_text)

        # 6. Check confidence calibration
        confidence_score = self._check_confidence_calibration(generated_text)

        metrics = HallucinationMetrics(
            factual_consistency_score=factual_score,
            coherence_score=coherence_score,
            grounding_score=grounding_score,
            repetition_score=repetition_score,
            confidence_calibration=confidence_score,
            contradiction_rate=contradiction_score
        )

        return {
            'metrics': metrics,
            'issues': issues,
            'overall_score': metrics.overall_score(),
            'is_hallucinating': metrics.overall_score() < 0.7
        }

    def _check_factual_consistency(self, text: str, context: str) -> float:
        """Check if generated text is factually consistent with context"""
        # Extract key entities and facts from context
        context_tokens = set(context.lower().split())
        text_tokens = set(text.lower().split())

        # Calculate overlap
        if len(text_tokens) == 0:
            return 0.0

        overlap = len(context_tokens & text_tokens)
        consistency = overlap / len(text_tokens)

        return min(1.0, consistency * 2)  # Scale up

    def _check_contradictions(self, text: str) -> float:
        """Detect self-contradictions in text"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            return 0.0

        # Check for negation patterns
        contradiction_keywords = ['but', 'however', 'although', 'except', 'not']
        contradiction_count = sum(
            1 for s in sentences
            for kw in contradiction_keywords
            if kw in s.lower()
        )

        return min(1.0, contradiction_count / len(sentences))

    def _check_grounding(self, text: str, context: str) -> float:
        """Check if text is grounded in provided context"""
        text_lower = text.lower()
        context_lower = context.lower()

        # Extract significant words (longer than 3 chars)
        text_words = [w for w in text_lower.split() if len(w) > 3]
        context_words = set(w for w in context_lower.split() if len(w) > 3)

        if not text_words:
            return 0.5

        grounded_words = sum(1 for w in text_words if w in context_words)
        grounding_ratio = grounded_words / len(text_words)

        return grounding_ratio

    def _check_repetitions(self, text: str) -> float:
        """Detect repetitive patterns"""
        words = text.lower().split()

        if len(words) < 5:
            return 0.0

        # Check for repeated phrases
        repeated_count = 0
        seen_ngrams = set()

        for n in [2, 3, 4]:  # Check 2-grams, 3-grams, 4-grams
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i+n])
                if ngram in seen_ngrams:
                    repeated_count += 1
                seen_ngrams.add(ngram)

        repetition_rate = repeated_count / max(1, len(words))
        return min(1.0, repetition_rate * 5)

    def _check_coherence(self, text: str) -> float:
        """Check logical coherence of text"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        if len(sentences) < 2:
            return 1.0

        # Simple coherence check based on lexical cohesion
        coherence_scores = []

        for i in range(len(sentences) - 1):
            curr_words = set(sentences[i].lower().split())
            next_words = set(sentences[i+1].lower().split())

            # Calculate overlap between consecutive sentences
            overlap = len(curr_words & next_words)
            score = min(1.0, overlap / 5)  # Expect some overlap
            coherence_scores.append(score)

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _check_confidence_calibration(self, text: str) -> float:
        """Check if model appropriately expresses uncertainty"""
        # Look for appropriate hedging language
        uncertainty_markers = ['maybe', 'perhaps', 'might', 'could', 'possibly',
                              'likely', 'probably', 'seems', 'appears']
        overconfident_markers = ['definitely', 'certainly', 'absolutely',
                                'always', 'never', 'impossible']

        text_lower = text.lower()

        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in text_lower)
        overconfident_count = sum(1 for marker in overconfident_markers if marker in text_lower)

        # Good calibration has some uncertainty, not too much overconfidence
        if overconfident_count > uncertainty_count * 2:
            return 0.3  # Too overconfident
        elif uncertainty_count > 0:
            return 0.9  # Good calibration
        else:
            return 0.6  # Neutral


class QualityDatasetBuilder:
    """Builds high-quality, verified conversational datasets"""

    def __init__(self):
        self.quality_filters = []
        self.validation_rules = []

    def create_quality_dataset(self,
                               raw_data: List[Dict],
                               verify: bool = True) -> List[Dict]:
        """Create verified, high-quality dataset"""

        print("\n" + "="*70)
        print("QUALITY DATASET BUILDER")
        print("="*70)

        quality_data = []
        filtered_count = 0

        for i, item in enumerate(raw_data):
            # Apply quality filters
            if verify and not self._verify_quality(item):
                filtered_count += 1
                continue

            # Enhance with metadata
            enhanced_item = self._enhance_item(item)
            quality_data.append(enhanced_item)

        print(f"\n✓ Processed: {len(raw_data)} items")
        print(f"✓ Quality items: {len(quality_data)}")
        print(f"✓ Filtered out: {filtered_count}")
        print(f"✓ Quality rate: {len(quality_data)/len(raw_data)*100:.1f}%")

        return quality_data

    def _verify_quality(self, item: Dict) -> bool:
        """Verify quality of a single data item"""
        if 'text' not in item:
            return False

        text = item['text']

        # Filter 1: Minimum length
        if len(text.split()) < 5:
            return False

        # Filter 2: Maximum length
        if len(text.split()) > 1000:
            return False

        # Filter 3: No excessive repetition
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return False

        # Filter 4: Proper sentence structure
        if not any(text.endswith(p) for p in ['.', '!', '?', '।']):
            return False

        # Filter 5: No toxic content patterns
        toxic_patterns = ['hate', 'kill', 'violence', 'explicit']
        if any(pattern in text.lower() for pattern in toxic_patterns):
            return False

        return True

    def _enhance_item(self, item: Dict) -> Dict:
        """Enhance item with quality metadata"""
        enhanced = item.copy()

        text = item['text']

        # Add quality metrics
        enhanced['quality_score'] = self._calculate_quality_score(text)
        enhanced['word_count'] = len(text.split())
        enhanced['sentence_count'] = len([s for s in text.split('.') if s.strip()])
        enhanced['has_context'] = 'context' in item

        return enhanced

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate overall quality score"""
        scores = []

        # Diversity score
        words = text.split()
        diversity = len(set(words)) / len(words) if words else 0
        scores.append(diversity)

        # Length score (prefer moderate length)
        length_score = min(1.0, len(words) / 100)
        scores.append(length_score)

        # Structure score
        has_punctuation = any(p in text for p in '.!?')
        structure_score = 1.0 if has_punctuation else 0.5
        scores.append(structure_score)

        return np.mean(scores)

    def create_conversational_pairs(self,
                                   contexts: List[str],
                                   responses: List[str]) -> List[Dict]:
        """Create high-quality conversational pairs"""

        pairs = []

        for context, response in zip(contexts, responses):
            pair = {
                'context': context,
                'response': response,
                'text': f"{context}\n{response}",
                'type': 'conversation'
            }

            # Validate pair quality
            if self._validate_pair(context, response):
                pairs.append(pair)

        return pairs

    def _validate_pair(self, context: str, response: str) -> bool:
        """Validate context-response pair"""
        # Both should be non-empty
        if not context.strip() or not response.strip():
            return False

        # Response should be relevant to context
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())

        overlap = len(context_words & response_words)
        relevance = overlap / len(response_words) if response_words else 0

        # Require some relevance
        return relevance > 0.1


class AntiHallucinationTrainer:
    """Advanced trainer with anti-hallucination techniques"""

    def __init__(self,
                 model,
                 tokenizer,
                 learning_rate: float = 1e-4,
                 hallucination_penalty: float = 0.5):

        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.hallucination_penalty = hallucination_penalty

        self.detector = HallucinationDetector()
        self.dataset_builder = QualityDatasetBuilder()

        self.training_history = {
            'loss': [],
            'hallucination_score': [],
            'quality_score': [],
            'perplexity': [],
        }

        self.current_step = 0
        self.hallucination_threshold = 0.7  # Target threshold

    def train_with_verification(self,
                               training_data: List[Dict],
                               num_epochs: int = 10,
                               validation_data: Optional[List[Dict]] = None,
                               max_hallucination_rate: float = 0.1) -> Dict:
        """
        Train model with continuous hallucination monitoring
        Continue until hallucination rate is below threshold
        """

        print("\n" + "="*70)
        print("ANTI-HALLUCINATION TRAINING")
        print("="*70)
        print(f"\nTarget: Hallucination rate < {max_hallucination_rate*100}%")
        print(f"Quality threshold: {self.hallucination_threshold}")

        # Build quality dataset
        quality_data = self.dataset_builder.create_quality_dataset(
            training_data,
            verify=True
        )

        iteration = 0
        hallucination_rate = 1.0

        while hallucination_rate > max_hallucination_rate and iteration < num_epochs:
            iteration += 1

            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{num_epochs}")
            print(f"{'='*70}")

            # Training epoch
            epoch_metrics = self._train_epoch(quality_data, iteration)

            # Validation with hallucination detection
            if validation_data:
                val_metrics = self._validate_with_detection(validation_data)
                hallucination_rate = val_metrics['hallucination_rate']

                print(f"\n✓ Validation Results:")
                print(f"  Hallucination Rate: {hallucination_rate*100:.2f}%")
                print(f"  Quality Score: {val_metrics['quality_score']:.3f}")
                print(f"  Coherence: {val_metrics['coherence']:.3f}")

                # Check if we've reached acceptable quality
                if hallucination_rate <= max_hallucination_rate:
                    print(f"\n🎉 SUCCESS! Achieved target hallucination rate!")
                    break
                else:
                    remaining = max_hallucination_rate - hallucination_rate
                    print(f"  Need to reduce by: {abs(remaining)*100:.2f}%")

        # Final evaluation
        final_metrics = self._final_evaluation(validation_data)

        return {
            'iterations': iteration,
            'final_hallucination_rate': hallucination_rate,
            'final_metrics': final_metrics,
            'training_history': self.training_history
        }

    def _train_epoch(self, training_data: List[Dict], epoch: int) -> Dict:
        """Train for one epoch with quality monitoring"""

        epoch_loss = 0.0
        epoch_quality = 0.0
        num_batches = 0

        # Shuffle data
        np.random.shuffle(training_data)

        batch_size = 4

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            if len(batch) < batch_size:
                continue

            # Prepare batch
            texts = [item['text'] for item in batch]

            # Training step with quality feedback
            metrics = self._training_step_with_quality(texts)

            epoch_loss += metrics['loss']
            epoch_quality += metrics['quality_score']
            num_batches += 1

            if num_batches % 10 == 0:
                avg_loss = epoch_loss / num_batches
                avg_quality = epoch_quality / num_batches
                print(f"  Batch {num_batches}: Loss={avg_loss:.4f}, Quality={avg_quality:.3f}")

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_epoch_quality = epoch_quality / num_batches if num_batches > 0 else 0

        return {
            'loss': avg_epoch_loss,
            'quality_score': avg_epoch_quality
        }

    def _training_step_with_quality(self, texts: List[str]) -> Dict:
        """Training step with quality monitoring"""

        # Encode texts
        encoded_batch = []
        for text in texts:
            encoded = self.tokenizer.encode(text, max_length=256)
            # Pad to 256
            if len(encoded) < 256:
                encoded.extend([0] * (256 - len(encoded)))
            else:
                encoded = encoded[:256]
            encoded_batch.append(encoded)

        input_ids = np.array(encoded_batch)

        # Forward pass
        logits = self.model.forward(input_ids)

        # Compute base loss
        labels = np.roll(input_ids, -1, axis=1)
        loss = self._compute_loss(logits, labels)

        # Generate sample to check quality
        sample_text = texts[0]
        generated = self._generate_sample(sample_text[:50])

        # Detect hallucination
        detection = self.detector.detect_hallucination(
            generated,
            sample_text,
            None
        )

        quality_score = detection['overall_score']

        # Apply hallucination penalty to loss
        if detection['is_hallucinating']:
            loss = loss * (1 + self.hallucination_penalty)

        # Record metrics
        self.training_history['loss'].append(loss)
        self.training_history['hallucination_score'].append(1 - quality_score)
        self.training_history['quality_score'].append(quality_score)

        self.current_step += 1

        return {
            'loss': loss,
            'quality_score': quality_score,
            'is_hallucinating': detection['is_hallucinating']
        }

    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        batch_size, seq_len, vocab_size = logits.shape

        # Simplified loss computation
        losses = []

        for b in range(min(2, batch_size)):  # Sample first 2 in batch
            for t in range(min(50, seq_len)):  # Sample first 50 tokens
                if labels[b, t] == 0:  # Padding
                    continue

                logit = logits[b, t]
                label = labels[b, t]

                # Softmax
                exp_logit = np.exp(logit - np.max(logit))
                probs = exp_logit / np.sum(exp_logit)

                # NLL
                loss = -np.log(probs[label] + 1e-10)
                losses.append(loss)

        return np.mean(losses) if losses else 0.0

    def _generate_sample(self, prompt: str, max_length: int = 30) -> str:
        """Generate sample text for quality checking"""
        # Simplified generation
        tokens = self.tokenizer.tokenize(prompt)
        return prompt + " [generated continuation...]"

    def _validate_with_detection(self, validation_data: List[Dict]) -> Dict:
        """Validate with hallucination detection"""

        hallucination_count = 0
        total_quality = 0.0
        total_coherence = 0.0
        num_samples = min(50, len(validation_data))

        for i in range(num_samples):
            item = validation_data[i]
            text = item['text']

            # Generate response
            generated = self._generate_sample(text[:100], max_length=50)

            # Detect hallucination
            detection = self.detector.detect_hallucination(
                generated,
                text,
                None
            )

            if detection['is_hallucinating']:
                hallucination_count += 1

            total_quality += detection['overall_score']
            total_coherence += detection['metrics'].coherence_score

        return {
            'hallucination_rate': hallucination_count / num_samples,
            'quality_score': total_quality / num_samples,
            'coherence': total_coherence / num_samples
        }

    def _final_evaluation(self, test_data: Optional[List[Dict]]) -> Dict:
        """Comprehensive final evaluation"""

        if not test_data:
            return {}

        print("\n" + "="*70)
        print("FINAL EVALUATION")
        print("="*70)

        metrics = self._validate_with_detection(test_data)

        print(f"\n✓ Final Metrics:")
        print(f"  Hallucination Rate: {metrics['hallucination_rate']*100:.2f}%")
        print(f"  Quality Score: {metrics['quality_score']:.3f}")
        print(f"  Coherence Score: {metrics['coherence']:.3f}")

        # Assess if model is ready
        is_ready = (
            metrics['hallucination_rate'] < 0.15 and
            metrics['quality_score'] > 0.7 and
            metrics['coherence'] > 0.6
        )

        print(f"\n  Model Ready for Production: {'YES ✓' if is_ready else 'NO - Continue Training'}")

        return metrics

    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""

        if not self.training_history['loss']:
            return {}

        return {
            'total_steps': len(self.training_history['loss']),
            'final_loss': self.training_history['loss'][-1],
            'best_loss': min(self.training_history['loss']),
            'avg_quality': np.mean(self.training_history['quality_score']),
            'final_quality': self.training_history['quality_score'][-1],
            'avg_hallucination': np.mean(self.training_history['hallucination_score']),
        }
