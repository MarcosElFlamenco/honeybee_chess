
# Basic training
train:
	python -m src.train \
		--output_dir ./my_model \
		--num_train_epochs 3 \
		--per_device_train_batch_size 32

remotetrain:
	sbatch scripts/main_train.sh