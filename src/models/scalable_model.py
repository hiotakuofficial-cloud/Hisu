"""
Scalable Language Model Architecture
Supports scaling from 5B to 10B+ parameters
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelScale:
    """Predefined model scale configurations"""

    @staticmethod
    def get_5b_config():
        """5 Billion parameter configuration"""
        return {
            'vocab_size': 50000,
            'max_seq_length': 2048,
            'd_model': 4096,
            'n_heads': 32,
            'n_layers': 32,
            'd_ff': 16384,
            'dropout': 0.1,
        }

    @staticmethod
    def get_7b_config():
        """7 Billion parameter configuration"""
        return {
            'vocab_size': 50000,
            'max_seq_length': 2048,
            'd_model': 4096,
            'n_heads': 32,
            'n_layers': 40,
            'd_ff': 16384,
            'dropout': 0.1,
        }

    @staticmethod
    def get_10b_config():
        """10 Billion parameter configuration"""
        return {
            'vocab_size': 50000,
            'max_seq_length': 4096,
            'd_model': 5120,
            'n_heads': 40,
            'n_layers': 48,
            'd_ff': 20480,
            'dropout': 0.1,
        }

    @staticmethod
    def get_custom_config(target_params: float):
        """Generate config for target parameter count (in billions)"""

        if target_params <= 5.5:
            return ModelScale.get_5b_config()
        elif target_params <= 8:
            return ModelScale.get_7b_config()
        else:
            return ModelScale.get_10b_config()


class ScalableTransformerConfig:
    """Flexible transformer configuration with parameter calculation"""

    def __init__(self, **kwargs):
        # Set defaults
        self.vocab_size = kwargs.get('vocab_size', 50000)
        self.max_seq_length = kwargs.get('max_seq_length', 2048)
        self.d_model = kwargs.get('d_model', 4096)
        self.n_heads = kwargs.get('n_heads', 32)
        self.n_layers = kwargs.get('n_layers', 32)
        self.d_ff = kwargs.get('d_ff', 16384)
        self.dropout = kwargs.get('dropout', 0.1)

        # Advanced features
        self.use_rotary_embeddings = kwargs.get('use_rotary_embeddings', True)
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.use_grouped_query_attention = kwargs.get('use_grouped_query_attention', True)
        self.gqa_num_kv_heads = kwargs.get('gqa_num_kv_heads', 8)
        self.use_layer_scaling = kwargs.get('use_layer_scaling', True)
        self.use_gradient_checkpointing = kwargs.get('use_gradient_checkpointing', True)

        # Calculate total parameters
        self.total_params = self._calculate_parameters()
        self.trainable_params = self.total_params  # All params trainable

    def _calculate_parameters(self) -> int:
        """Calculate total model parameters"""

        # Token embeddings
        token_emb = self.vocab_size * self.d_model

        # Position embeddings (if not rotary)
        pos_emb = 0 if self.use_rotary_embeddings else (self.max_seq_length * self.d_model)

        # Per layer calculations
        # Attention: Q, K, V, O projections
        if self.use_grouped_query_attention:
            kv_heads = self.gqa_num_kv_heads
            d_head = self.d_model // self.n_heads
            attn_params = (
                self.d_model * self.d_model +  # Q projection
                self.d_model * (kv_heads * d_head) +  # K projection
                self.d_model * (kv_heads * d_head) +  # V projection
                self.d_model * self.d_model  # O projection
            )
        else:
            attn_params = 4 * self.d_model * self.d_model

        # Feed-forward network
        ffn_params = (
            self.d_model * self.d_ff +  # Up projection
            self.d_ff * self.d_model  # Down projection
        )

        # Layer normalization (2 per layer: pre-attn, pre-ffn)
        ln_params = 4 * self.d_model  # gamma and beta for each

        # Total per layer
        params_per_layer = attn_params + ffn_params + ln_params

        # All layers
        total_layer_params = params_per_layer * self.n_layers

        # Final layer norm
        final_ln = 2 * self.d_model

        # Output projection (unembedding)
        output_proj = self.d_model * self.vocab_size

        # Total
        total = token_emb + pos_emb + total_layer_params + final_ln + output_proj

        return total

    def get_size_in_gb(self, precision: str = 'fp32') -> float:
        """Estimate model size in GB"""

        bytes_per_param = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
            'int8': 1,
        }

        bytes_total = self.total_params * bytes_per_param.get(precision, 4)
        gb = bytes_total / (1024 ** 3)

        return gb

    def display_info(self):
        """Display model configuration information"""

        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        print(f"\n📊 Model Scale:")
        print(f"   Total Parameters: {self.total_params:,}")
        print(f"   Parameters (Billions): {self.total_params/1e9:.2f}B")
        print(f"   Model Size (FP32): {self.get_size_in_gb('fp32'):.2f} GB")
        print(f"   Model Size (FP16): {self.get_size_in_gb('fp16'):.2f} GB")

        print(f"\n🏗️ Architecture:")
        print(f"   Hidden Size (d_model): {self.d_model}")
        print(f"   Layers: {self.n_layers}")
        print(f"   Attention Heads: {self.n_heads}")
        print(f"   Feed-Forward Size: {self.d_ff}")
        print(f"   Vocabulary Size: {self.vocab_size:,}")
        print(f"   Max Sequence Length: {self.max_seq_length:,}")

        print(f"\n⚙️ Features:")
        print(f"   Rotary Embeddings: {'✓' if self.use_rotary_embeddings else '✗'}")
        print(f"   Flash Attention: {'✓' if self.use_flash_attention else '✗'}")
        print(f"   Grouped Query Attention: {'✓' if self.use_grouped_query_attention else '✗'}")
        if self.use_grouped_query_attention:
            print(f"   GQA KV Heads: {self.gqa_num_kv_heads}")
        print(f"   Layer Scaling: {'✓' if self.use_layer_scaling else '✗'}")
        print(f"   Gradient Checkpointing: {'✓' if self.use_gradient_checkpointing else '✗'}")

        print(f"\n💾 Memory Estimates:")
        print(f"   Parameters Only (FP32): {self.get_size_in_gb('fp32'):.2f} GB")
        print(f"   Parameters Only (FP16): {self.get_size_in_gb('fp16'):.2f} GB")
        print(f"   Training (with optimizer): ~{self.get_size_in_gb('fp32') * 4:.2f} GB")

        print("="*70 + "\n")


class ModelScaler:
    """Utilities for scaling models"""

    @staticmethod
    def calculate_params_for_config(config: Dict) -> int:
        """Calculate parameters for a given config"""
        temp_config = ScalableTransformerConfig(**config)
        return temp_config.total_params

    @staticmethod
    def find_config_for_target_params(target_params: float,
                                     base_config: Optional[Dict] = None) -> Dict:
        """
        Find configuration that matches target parameter count
        target_params: in billions (e.g., 10.0 for 10B)
        """

        if base_config is None:
            base_config = ModelScale.get_5b_config()

        target_params_exact = int(target_params * 1e9)

        print(f"\n🎯 Finding configuration for ~{target_params:.1f}B parameters...")

        # Try different layer counts
        best_config = base_config.copy()
        best_diff = float('inf')

        for n_layers in range(20, 80, 4):
            for d_model in [3072, 4096, 5120, 6144]:
                for d_ff_multiplier in [3, 4, 5]:
                    config = base_config.copy()
                    config['n_layers'] = n_layers
                    config['d_model'] = d_model
                    config['d_ff'] = d_model * d_ff_multiplier
                    config['n_heads'] = d_model // 128  # Maintain head size ~128

                    params = ModelScaler.calculate_params_for_config(config)
                    diff = abs(params - target_params_exact)

                    if diff < best_diff:
                        best_diff = diff
                        best_config = config

                        if diff / target_params_exact < 0.05:  # Within 5%
                            print(f"✓ Found config with {params/1e9:.2f}B params")
                            return best_config

        actual_params = ModelScaler.calculate_params_for_config(best_config)
        print(f"✓ Best match: {actual_params/1e9:.2f}B params (target: {target_params:.1f}B)")

        return best_config

    @staticmethod
    def create_scaled_model(target_params: float,
                          base_config: Optional[Dict] = None):
        """Create model scaled to target parameter count"""

        config_dict = ModelScaler.find_config_for_target_params(
            target_params,
            base_config
        )

        config = ScalableTransformerConfig(**config_dict)
        config.display_info()

        return config


class ProgressiveScaler:
    """Progressive model scaling strategy"""

    def __init__(self, start_params: float = 1.0, end_params: float = 10.0, stages: int = 5):
        self.start_params = start_params
        self.end_params = end_params
        self.stages = stages

        self.schedule = self._create_scaling_schedule()

    def _create_scaling_schedule(self) -> List[float]:
        """Create progressive scaling schedule"""

        # Logarithmic scaling for smoother progression
        schedule = np.logspace(
            np.log10(self.start_params),
            np.log10(self.end_params),
            self.stages
        )

        return schedule.tolist()

    def display_schedule(self):
        """Display scaling schedule"""

        print("\n" + "="*70)
        print("PROGRESSIVE SCALING SCHEDULE")
        print("="*70)

        for i, params in enumerate(self.schedule, 1):
            config = ModelScaler.find_config_for_target_params(params)
            actual_params = ModelScaler.calculate_params_for_config(config)

            print(f"\nStage {i}/{self.stages}:")
            print(f"  Target: {params:.2f}B parameters")
            print(f"  Actual: {actual_params/1e9:.2f}B parameters")
            print(f"  Layers: {config['n_layers']}")
            print(f"  Hidden: {config['d_model']}")

        print("\n" + "="*70 + "\n")

    def get_stage_config(self, stage: int) -> Dict:
        """Get configuration for specific stage"""

        if stage < 1 or stage > self.stages:
            raise ValueError(f"Stage must be between 1 and {self.stages}")

        target_params = self.schedule[stage - 1]
        config = ModelScaler.find_config_for_target_params(target_params)

        return config


def create_10b_model():
    """Create 10B parameter model"""

    print("\n" + "="*70)
    print("CREATING 10B PARAMETER MODEL")
    print("="*70)

    config_dict = ModelScale.get_10b_config()
    config = ScalableTransformerConfig(**config_dict)

    config.display_info()

    return config


def create_custom_model(target_params: float):
    """Create model with custom parameter count"""

    print(f"\n🎯 Target: {target_params:.1f}B parameters")

    config = ModelScaler.create_scaled_model(target_params)

    return config


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SCALABLE MODEL ARCHITECTURE DEMO")
    print("="*70)

    # Show predefined configurations
    print("\n📋 Predefined Configurations:")

    for name, config_func in [('5B', ModelScale.get_5b_config),
                              ('7B', ModelScale.get_7b_config),
                              ('10B', ModelScale.get_10b_config)]:
        config_dict = config_func()
        config = ScalableTransformerConfig(**config_dict)
        print(f"\n{name} Model: {config.total_params/1e9:.2f}B parameters")

    # Show progressive scaling
    print("\n" + "="*70)
    print("PROGRESSIVE SCALING EXAMPLE")
    print("="*70)

    scaler = ProgressiveScaler(start_params=2.0, end_params=10.0, stages=5)
    scaler.display_schedule()
