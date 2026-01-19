"""
Training script for the Chess Challenge.

This script provides a complete training pipeline using the Hugging Face Trainer.
Students can modify this script to experiment with different training strategies.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

# Suppress warnings from third-party libraries (multiprocess has Python 3.14 compat issues)
warnings.filterwarnings("ignore", message="'return' in a 'finally' block")

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import ChessDataCollator, create_train_val_datasets
from src.model import ChessConfig, ChessForCausalLM
from src.trm_model import ChessTRMConfig, ChessTRMForCausalLM
from src.tokenizer import ChessTokenizer
from src.tokenizer_decomposed import ChessDecomposedTokenizer
from src.utils import count_parameters, print_parameter_budget, print_trm_parameter_budget


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a chess-playing language model"
    )
    
    # Model arguments
    parser.add_argument(
        "--arch", type=str, default="gpt", choices=["gpt", "trm"],
        help="Model architecture: gpt (baseline) or trm (weight-shared recurrent transformer)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=1200,
        help="Vocabulary size (will be overridden by the built tokenizer)"
    )
    parser.add_argument(
        "--n_embd", type=int, default=128,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--n_layer", type=int, default=4,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=4,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--n_ctx", type=int, default=256,
        help="Maximum context length"
    )
    parser.add_argument(
        "--n_inner", type=int, default=None,
        help="Feed-forward inner dimension (default: 4 * n_embd)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability"
    )
    parser.add_argument(
        "--n_cycles", type=int, default=8,
        help="TRM only: number of recurrent refinement cycles"
    )
    parser.add_argument(
        "--no_tie_weights", action="store_true",
        help="Disable weight tying between embedding and output layers"
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name", type=str, default="dlouapre/lichess_2025-01_1M",
        help="Name of the dataset on Hugging Face Hub"
    )
    parser.add_argument(
        "--tokenizer_type", type=str, default="move", choices=["move", "decomposed"],
        help="Tokenizer type: move (one move token) or decomposed (piece/from/to tokens)"
    )
    parser.add_argument(
        "--tokenizer_min_frequency", type=int, default=500,
        help="Minimum move frequency to include in tokenizer vocab"
    )
    parser.add_argument(
        "--tokenizer_max_samples", type=int, default=100000,
        help="Maximum games to scan when building tokenizer vocab"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Maximum number of training samples"
    )
    parser.add_argument(
        "--val_samples", type=int, default=5000,
        help="Number of validation samples"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for model and logs"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=64,
        help="Evaluation batch size per device"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    # Logging arguments
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Logging frequency"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000,
        help="Checkpoint saving frequency"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("CHESS CHALLENGE - TRAINING")
    print("=" * 60)
    
    # Build tokenizer from dataset
    print("\nBuilding tokenizer from dataset...")
    if args.tokenizer_type == "decomposed":
        tokenizer = ChessDecomposedTokenizer()
    else:
        tokenizer = ChessTokenizer.build_vocab_from_dataset(
            dataset_name=args.dataset_name,
            min_frequency=args.tokenizer_min_frequency,
            max_samples=args.tokenizer_max_samples,
        )
    print(f"   Vocabulary size: {tokenizer.vocab_size}")
    
    # Use the vocab size from tokenizer (override args if provided)
    actual_vocab_size = tokenizer.vocab_size
    
    # Create model configuration
    print("\nCreating model configuration...")
    if args.arch == "trm":
        config = ChessTRMConfig(
            vocab_size=actual_vocab_size,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_ctx=args.n_ctx,
            n_inner=args.n_inner,
            n_cycles=args.n_cycles,
            dropout=args.dropout,
            tie_weights=not args.no_tie_weights,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        config = ChessConfig(
            vocab_size=actual_vocab_size,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_ctx=args.n_ctx,
            n_inner=args.n_inner,
            dropout=args.dropout,
            tie_weights=not args.no_tie_weights,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Print parameter budget
    if args.arch == "trm":
        print_trm_parameter_budget(config)
    else:
        print_parameter_budget(config)
    
    # Create model
    print("\nCreating model...")
    if args.arch == "trm":
        model = ChessTRMForCausalLM(config)
    else:
        model = ChessForCausalLM(config)
    n_params = count_parameters(model)
    print(f"   Total parameters: {n_params:,}")
    
    if n_params > 1_000_000:
        print("WARNING: Model exceeds 1M parameter limit!")
    else:
        print("✓  Model is within 1M parameter limit")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        max_length=args.n_ctx,
        train_samples=args.max_train_samples,
        val_samples=args.val_samples,
    )
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    
    # Create data collator
    data_collator = ChessDataCollator(tokenizer, max_length=args.n_ctx)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to=["none"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("\nTraining complete!")
    print(f"   Model saved to: {args.output_dir}/final_model")


if __name__ == "__main__":
    main()