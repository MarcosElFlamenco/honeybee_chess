#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"  # we run from the Chess1MChallenge folder

uv sync -n  # we make sure deps are synced without using cache

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-200000}"
VAL_SAMPLES="${VAL_SAMPLES:-5000}"

EPOCHS_SHORT="${EPOCHS_SHORT:-1}"
EPOCHS_LONG="${EPOCHS_LONG:-2}"

EVAL_POSITIONS="${EVAL_POSITIONS:-500}"
DO_EVAL="${DO_EVAL:-1}"  # we set to 0 to skip evals

echo "max_train_samples=${MAX_TRAIN_SAMPLES}"
echo "val_samples=${VAL_SAMPLES}"
echo "epochs_short=${EPOCHS_SHORT}"
echo "epochs_long=${EPOCHS_LONG}"
echo "eval_positions=${EVAL_POSITIONS}"
echo "do_eval=${DO_EVAL}"

run_one () {
  local name="$1"
  local out_dir="$2"
  local n_ctx="$3"
  local n_embd="$4"
  local n_layer="$5"
  local n_head="$6"
  local n_cycles="$7"
  local bs="$8"
  local epochs="$9"

  echo ""
  echo "============================================================"
  echo "RUN: ${name}"
  echo "cfg: ctx=${n_ctx} d=${n_embd} layers=${n_layer} heads=${n_head} cycles=${n_cycles} bs=${bs} epochs=${epochs}"
  echo "out: ${out_dir}"
  echo "============================================================"

  uv run python -c "from src.trm_model import ChessTRMConfig, ChessTRMForCausalLM; from src.utils import count_parameters; cfg=ChessTRMConfig(vocab_size=148,n_ctx=${n_ctx},n_embd=${n_embd},n_layer=${n_layer},n_head=${n_head},n_cycles=${n_cycles},tie_weights=True); m=ChessTRMForCausalLM(cfg); print('params', count_parameters(m, trainable_only=False))"

  uv run python -m src.train \
    --arch trm \
    --tokenizer_type decomposed \
    --output_dir "${out_dir}" \
    --num_train_epochs "${epochs}" \
    --per_device_train_batch_size "${bs}" \
    --per_device_eval_batch_size "${bs}" \
    --n_ctx "${n_ctx}" \
    --n_embd "${n_embd}" \
    --n_layer "${n_layer}" \
    --n_head "${n_head}" \
    --n_cycles "${n_cycles}" \
    --learning_rate 5e-4 \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --val_samples "${VAL_SAMPLES}"

  if [ "${DO_EVAL}" = "1" ]; then
    uv run python -m src.evaluate \
      --model_path "${out_dir}/final_model" \
      --mode legal \
      --n_positions "${EVAL_POSITIONS}" \
      --temperature 0.7 \
      --top_k 10 \
      --max_retries 3
  fi
}

# we keep vocab fixed at 148 via tokenizer_type=decomposed
# 50k-ish params (48,776)
run_one "TRM_50K"  "./runs/trm_50k"   256 52 1 4 6 128 "${EPOCHS_SHORT}"

# 100k-ish params (~101,516)
run_one "TRM_100K" "./runs/trm_100k"  256 82 1 2 6 96  "${EPOCHS_SHORT}"

# 250k params (243,000) - train longer
run_one "TRM_250K_LONG" "./runs/trm_250k_long" 256 100 2 5 6 64 "${EPOCHS_LONG}"

# 500k-ish params (~495,390) - train longer
run_one "TRM_500K_LONG" "./runs/trm_500k_long" 256 147 2 3 6 48 "${EPOCHS_LONG}"

# 750k-ish params (~748,470)
run_one "TRM_750K" "./runs/trm_750k"  256 183 2 3 6 32 "${EPOCHS_SHORT}"

# 999k-ish params (~996,100) - keep under 1M
run_one "TRM_999K" "./runs/trm_999k"  256 175 3 5 6 24 "${EPOCHS_SHORT}"

echo ""
echo "ALL RUNS COMPLETE"

