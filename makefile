
# Basic training
train:
	python -m src.train \
		--output_dir ./my_model \
		--num_train_epochs 3 \
		--per_device_train_batch_size 32 \
    	--n_embd 64 \
    	--n_layer 20 \

valid_moves:
	python -m src.evaluate \
		--model_path ./outputs/$(CURRENT_TRAIN)/final_model \
		--mode legal \
		--n_positions 500

CURRENT_TRAIN = vanilla

generate_script:
	python generate_script.py $(CURRENT_TRAIN)

run_slurm:
	sbatch scripts/$(CURRENT_TRAIN).sh

run_training: generate_script run_slurm

