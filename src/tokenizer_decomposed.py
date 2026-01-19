"""
Decomposed Chess Tokenizer.

This tokenizer decomposes each move into 3-4 tokens:
- color+piece token (e.g., "WP", "BN")
- from-square token with suffix "_f" (e.g., "e2_f")
- to-square token with suffix "_t" (e.g., "e4_t")
- optional promotion token (one of "q", "r", "b", "n")

This avoids UNKs for rare moves and makes legality learning easier because the model
always emits explicit squares.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class ChessDecomposedTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    _MOVE_RE = re.compile(r"^[WB][PNBRQK][a-h][1-8][a-h][1-8].*$")

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)

        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            self._vocab = self._create_full_vocab()

        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )

    @staticmethod
    def _create_full_vocab() -> Dict[str, int]:
        special_tokens = [
            ChessDecomposedTokenizer.PAD_TOKEN,
            ChessDecomposedTokenizer.BOS_TOKEN,
            ChessDecomposedTokenizer.EOS_TOKEN,
            ChessDecomposedTokenizer.UNK_TOKEN,
        ]

        pieces = ["P", "N", "B", "R", "Q", "K"]
        colors = ["W", "B"]
        piece_tokens = [f"{c}{p}" for c in colors for p in pieces]

        files = "abcdefgh"
        ranks = "12345678"
        squares = [f"{f}{r}" for f in files for r in ranks]
        from_tokens = [f"{sq}_f" for sq in squares]
        to_tokens = [f"{sq}_t" for sq in squares]

        promo_tokens = ["q", "r", "b", "n"]

        tokens = special_tokens + piece_tokens + from_tokens + to_tokens + promo_tokens
        return {tok: idx for idx, tok in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def _tokenize(self, text: str) -> List[str]:
        raw = text.strip()
        if not raw:
            return []

        parts = raw.split()
        out: List[str] = []

        for part in parts:
            if part in {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}:
                out.append(part)
                continue

            if not self._MOVE_RE.match(part):
                out.append(self.UNK_TOKEN)
                continue

            color = part[0]
            piece = part[1]
            from_sq = part[2:4]
            to_sq = part[4:6]
            out.append(f"{color}{piece}")
            out.append(f"{from_sq}_f")
            out.append(f"{to_sq}_t")

            if "=" in part:
                promo_idx = part.find("=")
                if promo_idx != -1 and promo_idx + 1 < len(part):
                    promo = part[promo_idx + 1].lower()
                    if promo in {"q", "r", "b", "n"}:
                        out.append(promo)

        return out

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return " ".join(t for t in tokens if t not in special)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)

        return (vocab_file,)
