#!/bin/bash
#SBATCH -p mesonet 
#SBATCH --account=m25146
#SBATCH --job-name=train_vanilla_honeybee
#SBATCH --output=logs/vanilla_honeybee.out
#SBATCH --error=logs/vanilla_honeybee.out
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
    --output_dir ./my_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32
