"""
Large Language Model Architecture
5B parameter transformer model for anime and Hindi language understanding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class TransformerConfig:
    """Configuration for 5B parameter transformer model"""

    def __init__(
        self,
        vocab_size: int = 50000,
        max_seq_length: int = 2048,
        d_model: int = 4096,
        n_heads: int = 32,
        n_layers: int = 32,
        d_ff: int = 16384,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_rotary_embeddings: bool = True,
        use_flash_attention: bool = True,
        use_grouped_query_attention: bool = True,
        gqa_num_kv_heads: int = 8,
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_rotary_embeddings = use_rotary_embeddings
        self.use_flash_attention = use_flash_attention
        self.use_grouped_query_attention = use_grouped_query_attention
        self.gqa_num_kv_heads = gqa_num_kv_heads

        # Calculate parameters
        self.total_params = self._calculate_parameters()

    def _calculate_parameters(self) -> int:
        """Calculate approximate total parameters"""
        # Embedding layer
        embedding_params = self.vocab_size * self.d_model

        # Per-layer parameters
        attention_params = (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            4 * self.d_model  # Biases
        )

        ffn_params = (
            2 * self.d_model * self.d_ff +  # Up and down projections
            self.d_ff + self.d_model  # Biases
        )

        layer_norm_params = 4 * self.d_model  # 2 layer norms per layer

        params_per_layer = attention_params + ffn_params + layer_norm_params
        total_layer_params = params_per_layer * self.n_layers

        # Output layer
        output_params = self.d_model * self.vocab_size

        total = embedding_params + total_layer_params + output_params
        return total


class RotaryPositionalEmbedding:
    """Rotary Position Embedding (RoPE) for better positional encoding"""

    def __init__(self, dim: int, max_seq_length: int = 2048, base: int = 10000):
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        self.inv_freq = self._compute_inv_freq()

    def _compute_inv_freq(self) -> np.ndarray:
        """Compute inverse frequencies for rotary embeddings"""
        return 1.0 / (self.base ** (np.arange(0, self.dim, 2) / self.dim))

    def apply(self, x: np.ndarray, position: int) -> np.ndarray:
        """Apply rotary embeddings to input tensor"""
        # Simplified implementation
        seq_len = x.shape[1]
        positions = np.arange(seq_len)
        angles = np.outer(positions, self.inv_freq)
        cos = np.cos(angles)
        sin = np.sin(angles)

        # Apply rotation (simplified)
        return x * cos[position] + x * sin[position]


class MultiHeadAttention:
    """Multi-Head Attention with optional Grouped Query Attention"""

    def __init__(self, config: TransformerConfig):
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.use_gqa = config.use_grouped_query_attention
        self.num_kv_heads = config.gqa_num_kv_heads if self.use_gqa else config.n_heads

        # Initialize weights
        self.W_q = np.random.randn(config.d_model, config.d_model) * 0.02
        self.W_k = np.random.randn(config.d_model, self.num_kv_heads * self.d_head) * 0.02
        self.W_v = np.random.randn(config.d_model, self.num_kv_heads * self.d_head) * 0.02
        self.W_o = np.random.randn(config.d_model, config.d_model) * 0.02

        self.dropout = config.dropout

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through multi-head attention"""
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        K = K.reshape(batch_size, seq_len, self.num_kv_heads, self.d_head)
        V = V.reshape(batch_size, seq_len, self.num_kv_heads, self.d_head)

        # If using GQA, repeat K and V
        if self.use_gqa:
            repeat_factor = self.n_heads // self.num_kv_heads
            K = np.repeat(K, repeat_factor, axis=2)
            V = np.repeat(V, repeat_factor, axis=2)

        # Compute attention scores (simplified)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_head)

        if mask is not None:
            scores = scores + mask

        # Apply softmax
        attention_weights = self._softmax(scores)

        # Apply attention to values
        output = np.matmul(attention_weights, V)

        # Reshape and project
        output = output.reshape(batch_size, seq_len, self.d_model)
        output = output @ self.W_o

        return output

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FeedForward:
    """Feed-forward network with GELU activation"""

    def __init__(self, config: TransformerConfig):
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        # Initialize weights
        self.W1 = np.random.randn(config.d_model, config.d_ff) * 0.02
        self.W2 = np.random.randn(config.d_ff, config.d_model) * 0.02
        self.b1 = np.zeros(config.d_ff)
        self.b2 = np.zeros(config.d_model)

        self.dropout = config.dropout

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through feed-forward network"""
        # First linear layer + GELU
        hidden = x @ self.W1 + self.b1
        hidden = self._gelu(hidden)

        # Second linear layer
        output = hidden @ self.W2 + self.b2

        return output

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class TransformerBlock:
    """Single transformer block with attention and feed-forward"""

    def __init__(self, config: TransformerConfig):
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

        # Layer normalization parameters
        self.ln1_gamma = np.ones(config.d_model)
        self.ln1_beta = np.zeros(config.d_model)
        self.ln2_gamma = np.ones(config.d_model)
        self.ln2_beta = np.zeros(config.d_model)

        self.dropout = config.dropout

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through transformer block"""
        # Pre-norm architecture
        # Attention block with residual
        normed = self._layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attention_out = self.attention.forward(normed, mask)
        x = x + attention_out

        # Feed-forward block with residual
        normed = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ff_out = self.feed_forward.forward(normed)
        x = x + ff_out

        return x

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta


class LargeLanguageModel:
    """5B parameter transformer model for anime and Hindi language tasks"""

    def __init__(self, config: Optional[TransformerConfig] = None):
        if config is None:
            config = TransformerConfig()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model

        # Token and position embeddings
        self.token_embedding = np.random.randn(config.vocab_size, config.d_model) * 0.02

        if config.use_rotary_embeddings:
            self.pos_embedding = RotaryPositionalEmbedding(config.d_model, config.max_seq_length)
        else:
            self.pos_embedding = np.random.randn(config.max_seq_length, config.d_model) * 0.02

        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.n_layers)]

        # Final layer norm
        self.final_ln_gamma = np.ones(config.d_model)
        self.final_ln_beta = np.zeros(config.d_model)

        # Output projection
        self.output_projection = np.random.randn(config.d_model, config.vocab_size) * 0.02

        print(f"✓ Initialized LLM with {config.total_params:,} parameters (~{config.total_params/1e9:.2f}B)")

    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass through the model"""
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding[input_ids]

        # Add positional embeddings
        if isinstance(self.pos_embedding, RotaryPositionalEmbedding):
            for i in range(seq_len):
                x[:, i, :] = self.pos_embedding.apply(x[:, i:i+1, :], i)
        else:
            x = x + self.pos_embedding[:seq_len]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)

        # Final layer norm
        x = self._layer_norm(x, self.final_ln_gamma, self.final_ln_beta)

        # Project to vocabulary
        logits = x @ self.output_projection

        return logits

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> np.ndarray:
        """Generate text continuation from prompt"""
        generated = prompt_ids.copy()

        for _ in range(max_length):
            # Get logits for next token
            logits = self.forward(generated)
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < np.partition(next_token_logits, -top_k)[-top_k]
                next_token_logits[indices_to_remove] = -np.inf

            # Apply softmax
            probs = self._softmax(next_token_logits)

            # Sample next token
            next_token = np.random.choice(len(probs), p=probs)

            # Append to generated sequence
            generated = np.concatenate([generated, [[next_token]]], axis=1)

            # Stop if max length reached
            if generated.shape[1] >= self.config.max_seq_length:
                break

        return generated

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * normalized + beta

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def get_model_size(self) -> Dict:
        """Get model size information"""
        return {
            'total_parameters': self.config.total_params,
            'parameters_billions': self.config.total_params / 1e9,
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_heads': self.config.n_heads,
            'd_ff': self.config.d_ff,
            'vocab_size': self.config.vocab_size,
            'max_seq_length': self.config.max_seq_length,
        }


def create_5b_model() -> LargeLanguageModel:
    """Create a 5B parameter model configuration"""
    config = TransformerConfig(
        vocab_size=50000,
        max_seq_length=2048,
        d_model=4096,
        n_heads=32,
        n_layers=32,
        d_ff=16384,
        dropout=0.1,
        use_rotary_embeddings=True,
        use_flash_attention=True,
        use_grouped_query_attention=True,
        gqa_num_kv_heads=8,
    )

    model = LargeLanguageModel(config)
    return model
