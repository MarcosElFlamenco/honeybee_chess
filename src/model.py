"""
Chess Transformer Model for the Chess Challenge.

This module provides a simple GPT-style transformer architecture
designed to fit within the 1M parameter constraint.

Key components:
- ChessConfig: Configuration class for model hyperparameters
- ChessForCausalLM: The main model class for next-move prediction
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ChessConfig(PretrainedConfig):
    """
    Configuration class for the Chess Transformer model.
    
    This configuration is designed for a ~1M parameter model.
    Students can adjust these values to explore different architectures.
    
    Parameter budget breakdown (with default values):
    - Embeddings (vocab): 1200 x 128 = 153,600
    - Position Embeddings: 256 x 128 = 32,768
    - Transformer Layers: 6 x ~120,000 = ~720,000
    - LM Head (with weight tying): 0 (shared with embeddings)
    - Total: ~906,000 parameters
    
    Attributes:
        vocab_size: Size of the vocabulary (number of unique moves).
        n_embd: Embedding dimension (d_model).
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        n_ctx: Maximum sequence length (context window).
        n_inner: Feed-forward inner dimension (default: 3 * n_embd).
        dropout: Dropout probability.
        layer_norm_epsilon: Epsilon for layer normalization.
        tie_weights: Whether to tie embedding and output weights.
        rms_Norm: Whether to use RMSNorm instead of LayerNorm.
    
    """
    model_type = "chess_transformer"
    
    def __init__(
        self,
        vocab_size: int = 1200,
        n_embd: int = 128,
        n_layer: int = 6,
        n_head: int = 4,
        n_ctx: int = 256,
        n_inner: Optional[int] = None,
        group_size: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        tie_weights: bool = True,
        rms_Norm: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.group_size = group_size
        self.n_inner = n_inner if n_inner is not None else 3 * n_embd  # Reduced from 4x to 3x
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.tie_weights = tie_weights
        self.rms_Norm = rms_Norm
        # Inform HF base class about tying behavior
        self.tie_word_embeddings = bool(tie_weights)


class GroupedQueryAttention(nn.Module):

    def __init__(self, config: ChessConfig):
        super().__init__()

        assert config.n_head % config.group_size == 0, "n_head must be divisible by group_size"
        print(f"Using Grouped Query Attention with group_size={config.group_size}")        
        self.n_head = config.n_head          # Total Query heads
        self.group_size = config.group_size
        self.n_kv_head = self.n_head // config.group_size  # Number of KV heads
        
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Q projection stays the same, but K and V projections are smaller
        # Total output: n_embd (for Q) + 2 * (n_kv_head * head_dim) (for K and V)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.n_kv_head * self.head_dim)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.n_ctx, config.n_ctx))
                             .view(1, 1, config.n_ctx, config.n_ctx), persistent=False)


    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # 1. Project Q, K, V
        q = self.q_proj(x)  # (B, T, n_head * head_dim)
        kv = self.kv_proj(x) # (B, T, 2 * n_kv_head * head_dim)
        k, v = kv.split(self.n_kv_head * self.head_dim, dim=2)
        
        # 2. Reshape Q normally
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. Reshape K, V and REPEAT them to match Q
        k = k.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        # Repeat KV heads 'group_size' times to match n_head
        # We use .repeat_interleave to ensure head 0 of KV is used by the first 'group_size' Q heads
        k = k.repeat_interleave(self.group_size, dim=1) # (B, n_head, T, head_dim)
        v = v.repeat_interleave(self.group_size, dim=1) # (B, n_head, T, head_dim)
        
        # 4. Standard Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))
        
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)
        
        # 5. Recombine
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.c_proj(attn_output)
    


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    This is a standard scaled dot-product attention implementation
    with causal masking for autoregressive generation.
    """
    
    def __init__(self, config: ChessConfig):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        
        print(f"Using Regular Attention with group_size={config.group_size}")        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Combined QKV projection for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask (will be created on first forward pass)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(
                1, 1, config.n_ctx, config.n_ctx
            ),
            persistent=False,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embd
        )
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        
        return attn_output


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) module.
    
    Standard two-layer MLP with GELU activation.
    """
    
    def __init__(self, config: ChessConfig):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single transformer block with attention and feed-forward layers.
    
    Uses pre-normalization (LayerNorm before attention/FFN) for better
    training stability.
    """
    
    def __init__(self, config: ChessConfig,group_size: int = None):
        super().__init__()
#nn.modules.normalization.RMSNorm
        if config.rms_Norm == True:
            print(f"using RMSNorm")
            self.ln_1 = nn.RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        else:
            print(f"using LayerNorm")
            self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        if config.group_size is not None:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = MultiHeadAttention(config)
        if config.rms_Norm == True:
            self.ln_2 = nn.RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        else:
            self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        # Pre-norm FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class ChessForCausalLM(PreTrainedModel):
    """
    Chess Transformer for Causal Language Modeling (next-move prediction).
    
    This model is designed to predict the next chess move given a sequence
    of previous moves. It uses a GPT-style architecture with:
    - Token embeddings for chess moves
    - Learned positional embeddings
    - Stacked transformer blocks
    - Linear head for next-token prediction
    
    The model supports weight tying between the embedding layer and the
    output projection to save parameters.
    
    Example:
        >>> config = ChessConfig(vocab_size=1200, n_embd=128, n_layer=6)
        >>> model = ChessForCausalLM(config)
        >>> inputs = {"input_ids": torch.tensor([[1, 42, 87]])}
        >>> outputs = model(**inputs)
        >>> next_move_logits = outputs.logits[:, -1, :]
    """
    
    config_class = ChessConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    # Suppress missing-key warning for tied lm_head when loading
    keys_to_ignore_on_load_missing = ["lm_head.weight"]
    
    def __init__(self, config: ChessConfig):
        super().__init__(config)
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])


        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Declare tied weights for proper serialization
        if config.tie_weights:
            self._tied_weights_keys = ["lm_head.weight"]
        
        # Initialize weights
        self.post_init()
        
        # Tie weights if configured
        if config.tie_weights:
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Module:
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.wte = new_embeddings
        if getattr(self.config, "tie_weights", False):
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings

    def tie_weights(self):
        # Use HF helper to tie or clone depending on config
        if getattr(self.config, "tie_weights", False) or getattr(self.config, "tie_word_embeddings", False):
            self._tie_or_clone_weights(self.lm_head, self.wte)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            position_ids: Position IDs of shape (batch_size, seq_len).
            labels: Labels for language modeling loss.
            return_dict: Whether to return a ModelOutput object.
        
        Returns:
            CausalLMOutputWithPast containing loss (if labels provided) and logits.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(token_embeds + position_embeds)
        
        # Pass through transformer blocks
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross-entropy
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    @torch.no_grad()
    def generate_move(
        self,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> int:
        """
        Generate the next move given a sequence of moves.
        
        Applies structural constraints enforcing:
        ColoredPiece [SOURCE] source [DEST] dest [modifiers]*
        
        Args:
            input_ids: Token IDs of shape (1, seq_len).
            temperature: Sampling temperature (1.0 = no change).
            top_k: If set, only sample from top k tokens.
            top_p: If set, use nucleus sampling with this threshold.
        
        Returns:
            The token ID of the predicted next move.
        """
        self.eval()
        
        # Get logits for the last position
        outputs = self(input_ids)
        logits = outputs.logits[:, -1, :].clone() / temperature
        
        # Apply structural constraints (hardcoded to ChessTokenizer structure)
        from src.tokenizer import ChessLogitsProcessor
        processor = ChessLogitsProcessor()
        logits = processor.constrain_logits(input_ids, logits)
        
        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) 
        return next_token.item()




# Register the model with Auto classes for easy loading
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("chess_transformer", ChessConfig)
AutoModelForCausalLM.register(ChessConfig, ChessForCausalLM)
