"""
TRM-style model for the Chess Challenge.

This implements a weight-shared recurrent transformer (Tiny Recursive Model style)
for causal language modeling under the 1M parameter constraint.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ChessTRMConfig(PretrainedConfig):
    model_type = "chess_trm"

    def __init__(
        self,
        vocab_size: int = 1200,
        n_embd: int = 128,
        n_layer: int = 2,
        n_head: int = 4,
        n_ctx: int = 256,
        n_inner: Optional[int] = None,
        n_cycles: int = 8,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        tie_weights: bool = True,
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
        self.vocab_size = int(vocab_size)
        self.n_embd = int(n_embd)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_ctx = int(n_ctx)
        self.n_inner = int(n_inner) if n_inner is not None else int(3 * n_embd)
        self.n_cycles = int(n_cycles)
        self.dropout = float(dropout)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.tie_weights = bool(tie_weights)
        self.tie_word_embeddings = bool(tie_weights)


class _TRMMultiHeadAttention(nn.Module):
    def __init__(self, config: ChessTRMConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})")
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_ctx, config.n_ctx)).view(1, 1, config.n_ctx, config.n_ctx),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))

        if attention_mask is not None:
            expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(expanded == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        attn_output = self.c_proj(attn_output)
        return attn_output


class _TRMFeedForward(nn.Module):
    def __init__(self, config: ChessTRMConfig):
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


class _TRMBlock(nn.Module):
    def __init__(self, config: ChessTRMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = _TRMMultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = _TRMFeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ChessTRMForCausalLM(PreTrainedModel):
    config_class = ChessTRMConfig
    base_model_prefix = "trm"
    supports_gradient_checkpointing = True
    keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: ChessTRMConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_ctx, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([_TRMBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self._tied_weights_keys = ["lm_head.weight"]

        self.post_init()
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
        if getattr(self.config, "tie_weights", False) or getattr(self.config, "tie_word_embeddings", False):
            self._tie_or_clone_weights(self.lm_head, self.wte)

    def _init_weights(self, module: nn.Module):
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        if seq_len > self.config.n_ctx:
            raise ValueError(f"seq_len ({seq_len}) exceeds n_ctx ({self.config.n_ctx})")

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        token_embeds = self.wte(input_ids)
        pos_embeds = self.wpe(position_ids)
        input_injection = token_embeds + pos_embeds

        hidden_states = self.drop(input_injection)

        for _ in range(self.config.n_cycles):
            hidden_states = hidden_states + input_injection
            for block in self.blocks:
                hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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


AutoConfig.register("chess_trm", ChessTRMConfig)
AutoModelForCausalLM.register(ChessTRMConfig, ChessTRMForCausalLM)
