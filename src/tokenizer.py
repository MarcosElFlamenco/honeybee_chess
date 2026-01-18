"""
Custom Chess Tokenizer for the Chess Challenge.

This tokenizer treats each move as a single token using the extended UCI notation
from the Lichess dataset (e.g., WPe2e4, BNg8f6).

The dataset format uses:
- W/B prefix for White/Black
- Piece letter: P=Pawn, N=Knight, B=Bishop, R=Rook, Q=Queen, K=King
- Source and destination squares (e.g., e2e4)
- Special suffixes: (x)=capture, (+)=check, (+*)=checkmate, (o)/(O)=castling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class FrequencyChessTokenizer(PreTrainedTokenizer):
    """
    A frequency-based tokenizer for chess moves using extended UCI notation.
    
    This tokenizer maps each possible chess move to a unique token ID.
    The vocabulary is built from the training dataset to ensure all moves
    encountered during training have a corresponding token.
    
    Only includes moves that appear at least `min_frequency` times in the dataset.
    Rare moves become [UNK] tokens.
    
    Example:
        >>> tokenizer = FrequencyChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, 42, 87, 2]  # [BOS, e2e4, e7e5, EOS]
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initialize the chess tokenizer.
        
        Args:
            vocab_file: Path to a JSON file containing the vocabulary mapping.
            vocab: Dictionary mapping tokens to IDs (alternative to vocab_file).
            **kwargs: Additional arguments passed to PreTrainedTokenizer.
        """
        # Initialize special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        # Remove any duplicate special-token entries passed through kwargs
        # to avoid "multiple values for keyword" errors when loading from disk.
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)
        
        # Load or create vocabulary
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            # Create a minimal vocabulary with just special tokens
            # The full vocabulary should be built from the dataset
            self._vocab = self._create_default_vocab()
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """
        Create a minimal default vocabulary with just special tokens.
        
        For the full vocabulary, use `build_vocab_from_dataset()`.
        This minimal vocab is just a placeholder - you should build from data.
        """
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        return vocab
    
    @classmethod
    def build_vocab_from_iterator(
        cls,
        iterator,
        min_frequency: int = 1,
    ) -> "FrequencyChessTokenizer":
        """
        Build a tokenizer vocabulary from an iterator of game strings.
        
        Args:
            iterator: An iterator yielding game strings (space-separated moves).
            min_frequency: Minimum frequency for a token to be included.
        
        Returns:
            A FrequencyChessTokenizer with the built vocabulary.
        """
        from collections import Counter
        
        token_counts = Counter()
        
        for game in iterator:
            moves = game.strip().split()
            token_counts.update(moves)
        
        # Filter by frequency
        tokens = [
            token for token, count in token_counts.items()
            if count >= min_frequency
        ]
        
        # Sort for reproducibility
        tokens = sorted(tokens)
        
        # Build vocabulary
        special_tokens = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens + tokens)}
        
        return cls(vocab=vocab)
    
    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        min_frequency: int = 500,
        max_samples: Optional[int] = 100000,
    ) -> "FrequencyChessTokenizer":
        """
        Build a tokenizer vocabulary from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Dataset split to use.
            column: Column containing the game strings.
            min_frequency: Minimum frequency for a token to be included (default: 500).
            max_samples: Maximum number of samples to process (default: 100k).
        
        Returns:
            A FrequencyChessTokenizer with the built vocabulary.
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def game_iterator():
            for example in dataset:
                yield example[column]
        
        return cls.build_vocab_from_iterator(game_iterator(), min_frequency=min_frequency)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into a list of tokens.
        
        Args:
            text: A string of space-separated moves.
        
        Returns:
            List of move tokens.
        """
        return text.strip().split()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens back to a string."""
        # Filter out special tokens for cleaner output
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return " ".join(t for t in tokens if t not in special)
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory to save the vocabulary.
            filename_prefix: Optional prefix for the filename.
        
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)


def count_vocab_from_dataset(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = 10000,
) -> Dict[str, int]:
    """
    Count token frequencies in a dataset (useful for vocabulary analysis).
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        split: Dataset split to use.
        column: Column containing the game strings.
        max_samples: Maximum number of samples to process.
    
    Returns:
        Dictionary mapping tokens to their frequencies.
    """
    from collections import Counter
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    token_counts = Counter()
    
    for example in dataset:
        moves = example[column].strip().split()
        token_counts.update(moves)
    
    return dict(token_counts)


class ChessTokenizer(FrequencyChessTokenizer):
    """
    A compositional tokenizer for chess moves using split color/piece tokens.
    
    This tokenizer breaks each move into 6 core components with explicit structure:
    1. Color: W or B (makes turn information explicit!)
    2. Piece: P, N, B, R, Q, K
    3. SOURCE marker: [SOURCE]
    4. Source square: a1, a2, ..., h8
    5. DEST marker: [DEST]
    6. Destination square: a1, a2, ..., h8
    
    Optional modifier tokens for captures, checks, checkmate, and castling.
    
    Example:
        >>> tokenizer = ChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, W_id, P_id, SRC_id, e2_id, DST_id, e4_id, B_id, P_id, SRC_id, e7_id, DST_id, e5_id, 2]
    
    Vocabulary:
    - Colors (2): W, B [makes turn alternation explicit]
    - Pieces (6): P, N, B, R, Q, K
    - Position markers (2): [SOURCE], [DEST]
    - Squares (64): a1-h8
    - Modifiers (5): [CAPTURE], [CHECK], [CHECKMATE], [CASTLING_KS], [CASTLING_QS]
    - Special (4): [PAD], [BOS], [EOS], [UNK]
    Total: ~83 tokens (deterministic, 4 fewer than before)
    
    Key advantage: Color is now EXPLICIT, making turn alternation obvious to the model!
    """
    
    # Color tokens (split for explicit turn information)
    COLORS = ['W', 'B']
    
    # Piece tokens
    PIECES = ['P', 'N', 'B', 'R', 'Q', 'K']
    
    # Position markers
    POSITION_MARKERS = ['[SOURCE]', '[DEST]']
    
    # Board squares (standard chess notation)
    SQUARES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]
    
    # Move modifiers
    MODIFIERS = ['[CAPTURE]', '[CHECK]', '[CHECKMATE]', '[CASTLING_KS]', '[CASTLING_QS]']
    
    def __init__(self, **kwargs):
        """
        Initialize the compositional chess tokenizer.
        
        Vocabulary is built deterministically from pieces and squares.
        No vocab_file or dataset scanning needed.
        """
        # Remove vocab-related kwargs to avoid conflicts
        kwargs.pop("vocab_file", None)
        kwargs.pop("vocab", None)
        
        # Build deterministic vocabulary
        vocab = self._build_deterministic_vocab()
        
        # Initialize parent with the built vocab
        super().__init__(vocab=vocab, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        """
        Return the vocabulary size.
        
        Tokens: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3, W=4, B=5, P-K=6-11, 
                 [SOURCE]=12, [DEST]=13, squares=14-77, modifiers=78-82
        
        Total: 83 tokens (indices 0-82)
        """
        return 4 + 2 + 6 + 2 + 64 + 5  # special + colors + pieces + markers + squares + modifiers
    
    def _build_deterministic_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary deterministically from colored pieces, squares, and modifiers.
        
        Returns:
            Dictionary mapping token strings to IDs.
        """
        vocab = {}
        idx = 0
        
        # Special tokens first (matching parent class order)
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            vocab[token] = idx
            idx += 1
        
        # Color tokens (W, B)
        for color in self.COLORS:
            vocab[color] = idx
            idx += 1
        
        # Piece tokens (P, N, B, R, Q, K)
        for piece in self.PIECES:
            vocab[piece] = idx
            idx += 1
        
        # Position marker tokens
        for marker in self.POSITION_MARKERS:
            vocab[marker] = idx
            idx += 1
        
        # Square tokens
        for square in self.SQUARES:
            vocab[square] = idx
            idx += 1
        
        # Modifier tokens
        for modifier in self.MODIFIERS:
            vocab[modifier] = idx
            idx += 1
        
        return vocab
    
    def _parse_move(self, move_str: str) -> Dict:
        """
        Parse a move string in extended UCI notation.
        
        Args:
            move_str: Move string like "WPe2e4" or "BNg8f6(x)" or "We1g1(o)"
        
        Returns:
            Dictionary with keys: piece, color, src, dest, modifiers
        """
        import re
        
        # Pattern: [WB][PNBRQK]<square><square>(<modifiers>)
        pattern = r'([WB])([PNBRQK])([a-h][1-8])([a-h][1-8])((?:\([^)]*\))?)'
        match = re.match(pattern, move_str.strip())
        
        if not match:
            raise ValueError(f"Invalid move format: {move_str}")
        
        color, piece, src, dest, modifier_str = match.groups()
        
        # Parse modifiers
        modifiers = []
        if modifier_str:
            # Remove parentheses and split by lowercase letters/symbols
            mod_content = modifier_str.strip('()')
            
            if 'x' in mod_content:
                modifiers.append('[CAPTURE]')
            if '+*' in mod_content:
                modifiers.append('[CHECKMATE]')
            elif '+' in mod_content:
                modifiers.append('[CHECK]')
            if 'o' in mod_content or 'O' in mod_content:
                # Determine kingside vs queenside based on destination
                if dest == 'g1' or dest == 'g8':
                    modifiers.append('[CASTLING_KS]')
                elif dest == 'c1' or dest == 'c8':
                    modifiers.append('[CASTLING_QS]')
        
        return {
            'piece': piece,
            'color': color,
            'src': src,
            'dest': dest,
            'modifiers': modifiers,
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into component tokens with positional markers.
        
        Each move becomes: [ColoredPiece, [SOURCE], source, [DEST], dest, *modifiers]
        
        Args:
            text: String of space-separated moves (e.g., "WPe2e4 BPe7e5")
        
        Returns:
            List of component tokens with structure markers.
        """
        move_strings = text.strip().split()
        tokens = []
        
        for move_str in move_strings:
            parsed = self._parse_move(move_str)
            
            # Add color and piece as SEPARATE tokens (now explicit!)
            tokens.append(parsed['color'])  # W or B
            tokens.append(parsed['piece'])  # P, N, B, R, Q, K
            
            # Add positional markers and squares
            tokens.append('[SOURCE]')
            tokens.append(parsed['src'])
            tokens.append('[DEST]')
            tokens.append(parsed['dest'])
            
            # Add modifier tokens if any
            tokens.extend(parsed['modifiers'])
        
        return tokens
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstruct moves from component tokens with positional markers.
        
        Expects structure: Color, Piece, [SOURCE], source, [DEST], dest, *modifiers
        
        Args:
            tokens: List of component tokens
        
        Returns:
            Space-separated move string.
        """
        moves = []
        token_idx = 0
        
        while token_idx < len(tokens):
            token = tokens[token_idx]
            
            # Skip special tokens
            special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
            if token in special:
                token_idx += 1
                continue
            
            # Expect: Color token (W or B)
            if token not in self.COLORS:
                break
            
            color = token
            
            # Expect: Piece token (P, N, B, R, Q, K)
            if token_idx + 1 >= len(tokens) or tokens[token_idx + 1] not in self.PIECES:
                break
            
            piece = tokens[token_idx + 1]
            colored_piece = color + piece
            
            # Expect: [SOURCE] marker
            if token_idx + 2 >= len(tokens) or tokens[token_idx + 2] != '[SOURCE]':
                break
            
            # Expect: source square
            if token_idx + 3 >= len(tokens):
                break
            src = tokens[token_idx + 3]
            if src not in self.SQUARES:
                break
            
            # Expect: [DEST] marker
            if token_idx + 4 >= len(tokens) or tokens[token_idx + 4] != '[DEST]':
                break
            
            # Expect: dest square
            if token_idx + 5 >= len(tokens):
                break
            dest = tokens[token_idx + 5]
            if dest not in self.SQUARES:
                break
            
            # Build move string
            move_str = f"{color}{piece}{src}{dest}"
            
            # Collect modifiers (next tokens until we hit another color token or end)
            token_idx += 6
            modifiers_list = []
            
            while token_idx < len(tokens) and tokens[token_idx] in self.MODIFIERS:
                modifier = tokens[token_idx]
                modifiers_list.append(modifier)
                token_idx += 1
            
            # Append modifier suffixes
            if modifiers_list:
                modifier_str = ""
                if '[CAPTURE]' in modifiers_list:
                    modifier_str += "x"
                if '[CHECKMATE]' in modifiers_list:
                    modifier_str += "+*"
                elif '[CHECK]' in modifiers_list:
                    modifier_str += "+"
                if '[CASTLING_KS]' in modifiers_list:
                    modifier_str += "o"
                elif '[CASTLING_QS]' in modifiers_list:
                    modifier_str += "o"
                
                move_str += f"({modifier_str})"
            
            moves.append(move_str)
        
        return " ".join(moves)
    
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """
        Decode token IDs back to string representation.
        
        Properly handles individual tokens by converting each ID to its token string.
        For single tokens or incomplete move sequences, returns the raw token strings.
        For complete move sequences, reconstructs the move format.
        
        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            String representation of the tokens
        """
        # Convert tensor to list if needed
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Handle 2D tensor/list (batch)
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids]
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, int):
                token = self._convert_id_to_token(token_id)
            else:
                token = str(token_id)
            
            tokens.append(token)
        
        # Try to reconstruct moves from tokens
        # If successful, return the reconstructed moves
        reconstructed = self._try_reconstruct_moves(tokens, skip_special_tokens)
        if reconstructed is not None:
            return reconstructed
        
        # Fallback: return tokens joined with spaces, filtering special tokens if requested
        if skip_special_tokens:
            special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
            tokens = [t for t in tokens if t not in special]
        
        return " ".join(tokens)
    
    def _try_reconstruct_moves(self, tokens: List[str], skip_special_tokens: bool = False) -> Optional[str]:
        """
        Try to reconstruct complete moves from tokens.
        
        Returns the reconstructed move string if tokens form valid move(s),
        None if tokens don't form a complete move structure.
        
        Args:
            tokens: List of token strings
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Reconstructed move string or None
        """
        moves = []
        token_idx = 0
        found_moves = False
        
        while token_idx < len(tokens):
            token = tokens[token_idx]
            
            # Skip special tokens
            special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
            if token in special:
                token_idx += 1
                continue
            
            # Check if this starts a move (color token)
            if token not in self.COLORS:
                # No more complete moves
                break
            
            color = token
            
            # Need at least 6 more tokens for a complete move
            if token_idx + 5 >= len(tokens):
                break
            
            # Expect: Piece token (P, N, B, R, Q, K)
            if tokens[token_idx + 1] not in self.PIECES:
                break
            
            piece = tokens[token_idx + 1]
            
            # Expect: [SOURCE] marker
            if tokens[token_idx + 2] != '[SOURCE]':
                break
            
            # Expect: source square
            src = tokens[token_idx + 3]
            if src not in self.SQUARES:
                break
            
            # Expect: [DEST] marker
            if tokens[token_idx + 4] != '[DEST]':
                break
            
            # Expect: dest square
            dest = tokens[token_idx + 5]
            if dest not in self.SQUARES:
                break
            
            # Build move string
            move_str = f"{color}{piece}{src}{dest}"
            
            # Collect modifiers
            token_idx += 6
            modifiers_list = []
            
            while token_idx < len(tokens) and tokens[token_idx] in self.MODIFIERS:
                modifiers_list.append(tokens[token_idx])
                token_idx += 1
            
            # Append modifier suffixes
            if modifiers_list:
                modifier_str = ""
                if '[CAPTURE]' in modifiers_list:
                    modifier_str += "x"
                if '[CHECKMATE]' in modifiers_list:
                    modifier_str += "+*"
                elif '[CHECK]' in modifiers_list:
                    modifier_str += "+"
                if '[CASTLING_KS]' in modifiers_list:
                    modifier_str += "o"
                elif '[CASTLING_QS]' in modifiers_list:
                    modifier_str += "o"
                
                move_str += f"({modifier_str})"
            
            moves.append(move_str)
            found_moves = True
        
        if found_moves:
            return " ".join(moves)
        
        return None


class ChessLogitsProcessor:
    """
    Logits processor for enforcing chess move structure during generation.
    
    Enforces the token sequence pattern:
    Color Piece [SOURCE] source [DEST] dest [modifiers]*
    
    Uses a state machine with 7 states:
    - State 0: Expect color (W, B)
    - State 1: Expect piece (P, N, B, R, Q, K)
    - State 2: Expect [SOURCE] marker
    - State 3: Expect source square (a1-h8)
    - State 4: Expect [DEST] marker
    - State 5: Expect dest square (a1-h8)
    - State 6: Expect modifiers or next color token
    
    Token structure is hardcoded to match ChessTokenizer:
    - Colors: W, B (EXPLICIT for turn alternation)
    - Pieces: P, N, B, R, Q, K
    - Position markers: [SOURCE], [DEST]
    - Squares: a1-h8 (64 total)
    - Modifiers: [CAPTURE], [CHECK], [CHECKMATE], [CASTLING_KS], [CASTLING_QS]
    """
    
    # Token vocabulary indices (hardcoded to match ChessTokenizer vocab order)
    # Special tokens: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3
    # Colors (4-5)
    COLOR_IDS = {'W': 4, 'B': 5}
    # Pieces (6-11)
    PIECE_IDS = {'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11}
    # Position markers (12-13)
    POSITION_MARKER_IDS = {'[SOURCE]': 12, '[DEST]': 13}
    # Squares (14-77): a1=14, a2=15, ..., h8=77
    SQUARE_IDS = {f"{file}{rank}": 14 + (rank - 1) * 8 + ord(file) - ord('a')
                  for rank in range(1, 9) for file in "abcdefgh"}
    # Modifiers (78-82)
    MODIFIER_IDS = {
        '[CAPTURE]': 78, '[CHECK]': 79, '[CHECKMATE]': 80,
        '[CASTLING_KS]': 81, '[CASTLING_QS]': 82
    }
    
    def __init__(self):
        """
        Initialize the logits processor with hardcoded ChessTokenizer structure.
        """
        import torch
        self.torch = torch
        
        # Convert to sets for membership testing
        self.color_ids = set(self.COLOR_IDS.values())
        self.piece_ids = set(self.PIECE_IDS.values())
        self.square_ids = set(self.SQUARE_IDS.values())
        self.modifier_ids = set(self.MODIFIER_IDS.values())
    
    def _get_state(self, input_ids):
        """
        Determine current state in move sequence based on recent tokens.
        
        Returns state (0-6) indicating what token type is expected next.
        """
        if input_ids.numel() == 0:
            return 0  # Start: expect color
        
        # Get the sequence of tokens
        seq = input_ids[0].tolist()
        
        # Work backwards to find the last color token (marks start of move)
        last_move_idx = -1
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] in self.color_ids:
                last_move_idx = i
                break
        
        if last_move_idx == -1:
            return 0  # No color found, expect color
        
        # Count tokens since last color
        tokens_since_color = len(seq) - 1 - last_move_idx
        
        # Pattern: Color, Piece, [SOURCE], source, [DEST], dest, ...modifiers
        if tokens_since_color == 0:
            return 1  # Expect piece after color
        elif tokens_since_color == 1:
            # Should have: color, piece
            if seq[-1] in self.piece_ids:
                return 2  # Expect [SOURCE]
            else:
                return 1  # Unexpected, reset
        elif tokens_since_color == 2:
            # Should have: color, piece, [SOURCE]
            if (seq[-2] in self.piece_ids and 
                seq[-1] in [self.POSITION_MARKER_IDS['[SOURCE]']]):
                return 3  # Expect source square
            else:
                return 1  # Reset
        elif tokens_since_color == 3:
            # Should have: color, piece, [SOURCE], source
            if (seq[-3] in self.piece_ids and
                seq[-2] in [self.POSITION_MARKER_IDS['[SOURCE]']] and
                seq[-1] in self.square_ids):
                return 4  # Expect [DEST]
            else:
                return 1  # Reset
        elif tokens_since_color == 4:
            # Should have: color, piece, [SOURCE], source, [DEST]
            if (seq[-2] in self.square_ids and
                seq[-1] in [self.POSITION_MARKER_IDS['[DEST]']]):
                return 5  # Expect dest square
            else:
                return 1  # Reset
        elif tokens_since_color == 5:
            # Should have: color, piece, [SOURCE], source, [DEST], dest
            if seq[-1] in self.square_ids:
                return 6  # Expect modifiers or next color (move complete)
            else:
                return 1  # Reset
        else:
            # tokens_since_color >= 6: We're in modifiers or expecting next move
            # If last token is a modifier, still expect more modifiers or next color
            # If last token is not a modifier, we should expect next color
            if seq[-1] not in self.modifier_ids:
                return 0  # Expect next move (next color)
            else:
                return 6  # Could be more modifiers or next color
    
    def constrain_logits(self, input_ids, logits):
        """
        Mask invalid tokens in logits based on move structure.
        
        Sets logits to -inf for tokens that violate move structure.
        
        Args:
            input_ids: Model input token IDs of shape (batch_size, seq_len)
            logits: Model output logits of shape (batch_size, vocab_size)
        
        Returns:
            Modified logits with invalid tokens masked to -inf
        """
        state = self._get_state(input_ids)
        
        # Create a mask for valid tokens (all ones initially)
        valid_mask = self.torch.ones(logits.shape[-1], dtype=self.torch.bool)
        valid_mask[:] = False  # Start by forbidding all
        
        # Allow tokens based on current state
        if state == 0:
            # Expect color (W or B)
            for color_id in self.color_ids:
                valid_mask[color_id] = True
        
        elif state == 1:
            # Expect piece (P, N, B, R, Q, K)
            for piece_id in self.piece_ids:
                valid_mask[piece_id] = True
        
        elif state == 2:
            # Expect [SOURCE]
            valid_mask[self.POSITION_MARKER_IDS['[SOURCE]']] = True
        
        elif state == 3:
            # Expect source square
            for square_id in self.square_ids:
                valid_mask[square_id] = True
        
        elif state == 4:
            # Expect [DEST]
            valid_mask[self.POSITION_MARKER_IDS['[DEST]']] = True
        
        elif state == 5:
            # Expect dest square
            for square_id in self.square_ids:
                valid_mask[square_id] = True
        
        elif state == 6:
            # Expect modifiers or next color token
            # Allow: modifiers + colors + EOS
            for modifier_id in self.modifier_ids:
                valid_mask[modifier_id] = True
            for color_id in self.color_ids:
                valid_mask[color_id] = True
            valid_mask[2] = True  # Allow EOS to end sequence
        
        # Apply mask
        logits = logits.clone()
        logits[0, ~valid_mask] = float('-inf')
        
        return logits

