#!/bin/bash
#SBATCH -p mesonet 
#SBATCH --account=m25146
#SBATCH --job-name=train_all_long
#SBATCH --output=logs/all_long.out
#SBATCH --error=logs/all_long.out
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-user=oscar.garnier@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate virtual environment
source .venv/bin/activate

# Run the training script
python -m src.train \
    --n_embd 124 \
    --n_layer 7 \
    --n_head 4 \
    --n_ctx 256 \
    --dropout 0.1 \
    --rms_Norm \
    --dataset_name dlouapre/lichess_2025-01_1M \
    --val_samples 5000 \
    --output_dir ./outputs/all_long \
    --num_train_epochs 3 \
    --group_size 4 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --learning_rate 0.0005 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 1000
