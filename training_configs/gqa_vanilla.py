# Example config for the `vanilla` training run.
# These defaults mirror the defaults in `src/train.py`'s `parse_args()`.

vocab_size = 1200
n_embd = 128
n_layer = 4
n_head = 4
n_ctx = 256
n_inner = None
dropout = 0.1
no_tie_weights = False
group_size = 4

dataset_name = "dlouapre/lichess_2025-01_1M"
max_train_samples = None
val_samples = 5000

output_dir = "./output"
num_train_epochs = 3
per_device_train_batch_size = 32
per_device_eval_batch_size = 64
learning_rate = 5e-4
weight_decay = 0.01
warmup_ratio = 0.1
seed = 42

logging_steps = 100
eval_steps = 500
save_steps = 1000
