
# Basic training
train:
	python -m src.train \
		--output_dir ./my_model \
		--num_train_epochs 3 \
		--per_device_train_batch_size 32 \
    	--n_embd 64 \
    	--n_layer 20 \


CURRENT_TRAIN = all_long

valid_moves:
	python -m src.evaluate \
		--model_path ./outputs/$(CURRENT_TRAIN)/final_model \
		--mode legal \
		--n_positions 500

generate_script:
	python generate_script.py $(CURRENT_TRAIN)

run_slurm:
	sbatch scripts/$(CURRENT_TRAIN).sh

run_training: generate_script run_slurm


local_test: generate_script
	bash scripts/$(CURRENT_TRAIN).sh

submit:
	python submit.py --model_path ./outputs/vanilla/final_model --model_name oscar_best_chess_model 
