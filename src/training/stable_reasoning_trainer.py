"""
Stable Reasoning Training Pipeline
Ensures coherent, non-hallucinating conversational AI with iterative optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
import json


class ReasoningStabilityValidator:
    """Validates reasoning stability and consistency"""

    def __init__(self):
        self.stability_threshold = 0.8
        self.consistency_checks = []

    def validate_reasoning(self,
                          model_outputs: List[str],
                          ground_truths: List[str],
                          contexts: List[str]) -> Dict:
        """Comprehensive reasoning validation"""

        metrics = {
            'consistency_score': 0.0,
            'stability_score': 0.0,
            'logical_coherence': 0.0,
            'factual_accuracy': 0.0,
            'response_quality': 0.0,
        }

        if not model_outputs:
            return metrics

        # 1. Consistency check
        metrics['consistency_score'] = self._check_consistency(
            model_outputs,
            contexts
        )

        # 2. Stability check (repeated queries should have stable answers)
        metrics['stability_score'] = self._check_stability(
            model_outputs
        )

        # 3. Logical coherence
        metrics['logical_coherence'] = self._check_logical_coherence(
            model_outputs,
            contexts
        )

        # 4. Factual accuracy (if ground truth available)
        if ground_truths:
            metrics['factual_accuracy'] = self._check_factual_accuracy(
                model_outputs,
                ground_truths
            )

        # 5. Response quality
        metrics['response_quality'] = self._check_response_quality(
            model_outputs
        )

        # Overall score
        metrics['overall'] = np.mean([
            metrics['consistency_score'],
            metrics['stability_score'],
            metrics['logical_coherence'],
            metrics['factual_accuracy'],
            metrics['response_quality']
        ])

        return metrics

    def _check_consistency(self,
                          outputs: List[str],
                          contexts: List[str]) -> float:
        """Check if outputs are consistent with contexts"""

        scores = []

        for output, context in zip(outputs, contexts):
            output_words = set(output.lower().split())
            context_words = set(context.lower().split())

            # Check overlap
            if len(output_words) == 0:
                scores.append(0.0)
                continue

            overlap = len(output_words & context_words)
            consistency = min(1.0, overlap / len(output_words) * 2)
            scores.append(consistency)

        return np.mean(scores) if scores else 0.0

    def _check_stability(self, outputs: List[str]) -> float:
        """Check stability across multiple outputs"""

        if len(outputs) < 2:
            return 1.0

        # Compare consecutive outputs for stability
        stability_scores = []

        for i in range(len(outputs) - 1):
            words1 = set(outputs[i].lower().split())
            words2 = set(outputs[i+1].lower().split())

            if not words1 or not words2:
                continue

            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)

            similarity = intersection / union if union > 0 else 0
            stability_scores.append(similarity)

        return np.mean(stability_scores) if stability_scores else 0.5

    def _check_logical_coherence(self,
                                outputs: List[str],
                                contexts: List[str]) -> float:
        """Check logical coherence of outputs"""

        coherence_scores = []

        for output in outputs:
            sentences = [s.strip() for s in output.split('.') if s.strip()]

            if len(sentences) < 2:
                coherence_scores.append(1.0)
                continue

            # Check sentence-to-sentence coherence
            sentence_scores = []

            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i+1].lower().split())

                overlap = len(words1 & words2)
                score = min(1.0, overlap / 3)  # Expect some overlap
                sentence_scores.append(score)

            coherence_scores.append(np.mean(sentence_scores))

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def _check_factual_accuracy(self,
                               outputs: List[str],
                               ground_truths: List[str]) -> float:
        """Check factual accuracy against ground truth"""

        accuracy_scores = []

        for output, truth in zip(outputs, ground_truths):
            output_words = set(output.lower().split())
            truth_words = set(truth.lower().split())

            if not truth_words:
                continue

            # Calculate overlap with ground truth
            overlap = len(output_words & truth_words)
            accuracy = overlap / len(truth_words)
            accuracy_scores.append(min(1.0, accuracy))

        return np.mean(accuracy_scores) if accuracy_scores else 0.0

    def _check_response_quality(self, outputs: List[str]) -> float:
        """Check overall response quality"""

        quality_scores = []

        for output in outputs:
            words = output.split()

            # Length quality (prefer moderate length)
            length_score = min(1.0, len(words) / 50)

            # Diversity quality
            unique_ratio = len(set(words)) / len(words) if words else 0
            diversity_score = unique_ratio

            # Structure quality (has punctuation)
            has_structure = any(p in output for p in '.!?')
            structure_score = 1.0 if has_structure else 0.5

            # Combine
            quality = np.mean([length_score, diversity_score, structure_score])
            quality_scores.append(quality)

        return np.mean(quality_scores) if quality_scores else 0.0


class IterativeTrainingOptimizer:
    """Iterative optimization until stable reasoning is achieved"""

    def __init__(self,
                 target_quality: float = 0.85,
                 max_iterations: int = 20,
                 improvement_threshold: float = 0.01):

        self.target_quality = target_quality
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

        self.iteration_history = []

    def optimize_until_stable(self,
                             trainer,
                             training_data: List[Dict],
                             validation_data: List[Dict]) -> Dict:
        """
        Iteratively train until stable reasoning is achieved
        Returns when quality threshold is met or max iterations reached
        """

        print("\n" + "="*70)
        print("ITERATIVE OPTIMIZATION FOR STABLE REASONING")
        print("="*70)
        print(f"\n🎯 Target Quality: {self.target_quality:.2%}")
        print(f"📊 Max Iterations: {self.max_iterations}")
        print(f"📈 Improvement Threshold: {self.improvement_threshold:.2%}")

        current_quality = 0.0
        iteration = 0
        no_improvement_count = 0

        while iteration < self.max_iterations:
            iteration += 1

            print(f"\n{'='*70}")
            print(f"OPTIMIZATION ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*70}")

            # Training phase
            print("\n📚 Training phase...")
            train_metrics = trainer._train_epoch(training_data, iteration)

            # Validation phase
            print("\n✅ Validation phase...")
            val_metrics = trainer._validate_with_detection(validation_data)

            # Calculate quality
            current_quality = val_metrics['quality_score']
            improvement = current_quality - (
                self.iteration_history[-1]['quality']
                if self.iteration_history else 0.0
            )

            # Record iteration
            iteration_data = {
                'iteration': iteration,
                'quality': current_quality,
                'improvement': improvement,
                'train_loss': train_metrics['loss'],
                'hallucination_rate': val_metrics['hallucination_rate'],
                'coherence': val_metrics['coherence'],
            }
            self.iteration_history.append(iteration_data)

            # Display progress
            print(f"\n📊 Iteration {iteration} Results:")
            print(f"   Quality Score: {current_quality:.3f} (target: {self.target_quality:.3f})")
            print(f"   Improvement: {improvement:+.4f}")
            print(f"   Hallucination Rate: {val_metrics['hallucination_rate']*100:.2f}%")
            print(f"   Coherence: {val_metrics['coherence']:.3f}")

            # Check if target reached
            if current_quality >= self.target_quality:
                print(f"\n✅ SUCCESS! Target quality reached in {iteration} iterations!")
                break

            # Check for improvement
            if improvement < self.improvement_threshold:
                no_improvement_count += 1
                print(f"   ⚠️ Low improvement ({no_improvement_count}/3)")

                if no_improvement_count >= 3:
                    print(f"\n⚠️ Stopping: No significant improvement for 3 iterations")
                    break
            else:
                no_improvement_count = 0

            # Progress indicator
            progress = (current_quality / self.target_quality) * 100
            print(f"   Progress: {progress:.1f}% to target")

        # Final summary
        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"   Total Iterations: {iteration}")
        print(f"   Final Quality: {current_quality:.3f}")
        print(f"   Target Quality: {self.target_quality:.3f}")
        print(f"   Status: {'✅ TARGET REACHED' if current_quality >= self.target_quality else '⚠️ PARTIAL SUCCESS'}")

        return {
            'total_iterations': iteration,
            'final_quality': current_quality,
            'target_reached': current_quality >= self.target_quality,
            'iteration_history': self.iteration_history
        }

    def get_optimization_summary(self) -> Dict:
        """Get optimization summary"""

        if not self.iteration_history:
            return {}

        qualities = [it['quality'] for it in self.iteration_history]
        improvements = [it['improvement'] for it in self.iteration_history]

        return {
            'total_iterations': len(self.iteration_history),
            'initial_quality': qualities[0],
            'final_quality': qualities[-1],
            'best_quality': max(qualities),
            'total_improvement': qualities[-1] - qualities[0],
            'avg_improvement_per_iter': np.mean(improvements[1:]) if len(improvements) > 1 else 0,
        }


class StableReasoningTrainer:
    """
    Complete training pipeline for stable, non-hallucinating conversational AI
    Combines anti-hallucination training with iterative optimization
    """

    def __init__(self,
                 model,
                 tokenizer,
                 target_quality: float = 0.85,
                 max_hallucination_rate: float = 0.10):

        self.model = model
        self.tokenizer = tokenizer
        self.target_quality = target_quality
        self.max_hallucination_rate = max_hallucination_rate

        # Initialize components
        from .anti_hallucination_trainer import (
            AntiHallucinationTrainer,
            HallucinationDetector,
            QualityDatasetBuilder
        )

        self.base_trainer = AntiHallucinationTrainer(
            model=model,
            tokenizer=tokenizer,
            hallucination_penalty=0.5
        )

        self.validator = ReasoningStabilityValidator()
        self.optimizer = IterativeTrainingOptimizer(
            target_quality=target_quality,
            max_iterations=20
        )

        self.training_log = []

    def train_to_stable_reasoning(self,
                                 training_data: List[Dict],
                                 validation_data: List[Dict],
                                 test_data: Optional[List[Dict]] = None) -> Dict:
        """
        Main training function - trains until stable reasoning is achieved
        """

        print("\n" + "="*70)
        print("STABLE REASONING TRAINING PIPELINE")
        print("="*70)
        print(f"\n📋 Configuration:")
        print(f"   Target Quality: {self.target_quality:.2%}")
        print(f"   Max Hallucination Rate: {self.max_hallucination_rate:.2%}")
        print(f"   Training Examples: {len(training_data)}")
        print(f"   Validation Examples: {len(validation_data)}")
        if test_data:
            print(f"   Test Examples: {len(test_data)}")

        start_time = time.time()

        # Phase 1: Anti-hallucination training
        print(f"\n{'='*70}")
        print("PHASE 1: ANTI-HALLUCINATION TRAINING")
        print(f"{'='*70}")

        phase1_results = self.base_trainer.train_with_verification(
            training_data=training_data,
            num_epochs=10,
            validation_data=validation_data,
            max_hallucination_rate=self.max_hallucination_rate
        )

        print(f"\n✓ Phase 1 Complete:")
        print(f"   Iterations: {phase1_results['iterations']}")
        print(f"   Hallucination Rate: {phase1_results['final_hallucination_rate']*100:.2f}%")

        # Phase 2: Iterative optimization for stable reasoning
        print(f"\n{'='*70}")
        print("PHASE 2: ITERATIVE REASONING OPTIMIZATION")
        print(f"{'='*70}")

        phase2_results = self.optimizer.optimize_until_stable(
            trainer=self.base_trainer,
            training_data=training_data,
            validation_data=validation_data
        )

        print(f"\n✓ Phase 2 Complete:")
        print(f"   Iterations: {phase2_results['total_iterations']}")
        print(f"   Final Quality: {phase2_results['final_quality']:.3f}")
        print(f"   Target Reached: {'Yes ✓' if phase2_results['target_reached'] else 'No'}")

        # Phase 3: Final evaluation
        print(f"\n{'='*70}")
        print("PHASE 3: FINAL EVALUATION")
        print(f"{'='*70}")

        if test_data:
            final_metrics = self._final_evaluation(test_data)
        else:
            final_metrics = phase2_results

        # Training complete
        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"   Total Time: {total_time/60:.1f} minutes")
        print(f"   Phase 1 Iterations: {phase1_results['iterations']}")
        print(f"   Phase 2 Iterations: {phase2_results['total_iterations']}")
        print(f"   Final Hallucination Rate: {phase1_results['final_hallucination_rate']*100:.2f}%")
        print(f"   Final Quality Score: {phase2_results['final_quality']:.3f}")

        # Determine if model is ready
        is_ready = (
            phase1_results['final_hallucination_rate'] <= self.max_hallucination_rate and
            phase2_results['final_quality'] >= self.target_quality
        )

        print(f"\n   MODEL STATUS: {'✅ READY FOR DEPLOYMENT' if is_ready else '⚠️ NEEDS MORE TRAINING'}")
        print(f"{'='*70}\n")

        return {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'final_metrics': final_metrics,
            'total_time': total_time,
            'model_ready': is_ready
        }

    def _final_evaluation(self, test_data: List[Dict]) -> Dict:
        """Comprehensive final evaluation"""

        print("\n📊 Running final evaluation on test set...")

        # Get model outputs
        outputs = []
        contexts = []
        ground_truths = []

        num_samples = min(50, len(test_data))

        for i in range(num_samples):
            item = test_data[i]
            context = item.get('input', item.get('text', ''))
            ground_truth = item.get('output', '')

            # Generate output (simplified)
            output = self._generate_output(context)

            outputs.append(output)
            contexts.append(context)
            ground_truths.append(ground_truth)

        # Validate reasoning
        reasoning_metrics = self.validator.validate_reasoning(
            outputs,
            ground_truths,
            contexts
        )

        # Hallucination check
        hallucination_metrics = self.base_trainer._validate_with_detection(test_data)

        # Combine metrics
        final_metrics = {
            **reasoning_metrics,
            'hallucination_rate': hallucination_metrics['hallucination_rate'],
            'is_production_ready': (
                reasoning_metrics['overall'] >= 0.8 and
                hallucination_metrics['hallucination_rate'] <= 0.15
            )
        }

        print(f"\n✓ Final Evaluation Results:")
        print(f"   Reasoning Quality: {reasoning_metrics['overall']:.3f}")
        print(f"   Consistency: {reasoning_metrics['consistency_score']:.3f}")
        print(f"   Stability: {reasoning_metrics['stability_score']:.3f}")
        print(f"   Logical Coherence: {reasoning_metrics['logical_coherence']:.3f}")
        print(f"   Factual Accuracy: {reasoning_metrics['factual_accuracy']:.3f}")
        print(f"   Hallucination Rate: {hallucination_metrics['hallucination_rate']*100:.2f}%")
        print(f"   Production Ready: {'YES ✓' if final_metrics['is_production_ready'] else 'NO'}")

        return final_metrics

    def _generate_output(self, context: str) -> str:
        """Generate output from context (simplified)"""
        # In real implementation, this would use the model
        return f"Response to: {context}"

    def save_training_results(self, results: Dict, output_path: str):
        """Save training results to file"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n✓ Training results saved to: {output_path}")

    def get_training_summary(self) -> str:
        """Get human-readable training summary"""

        optimizer_summary = self.optimizer.get_optimization_summary()
        trainer_summary = self.base_trainer.get_training_summary()

        summary = f"""
TRAINING SUMMARY
{'='*70}

Anti-Hallucination Training:
  Total Steps: {trainer_summary.get('total_steps', 0)}
  Final Loss: {trainer_summary.get('final_loss', 0):.4f}
  Average Quality: {trainer_summary.get('avg_quality', 0):.3f}
  Average Hallucination: {trainer_summary.get('avg_hallucination', 0):.3f}

Iterative Optimization:
  Total Iterations: {optimizer_summary.get('total_iterations', 0)}
  Initial Quality: {optimizer_summary.get('initial_quality', 0):.3f}
  Final Quality: {optimizer_summary.get('final_quality', 0):.3f}
  Total Improvement: {optimizer_summary.get('total_improvement', 0):.3f}

{'='*70}
"""

        return summary
