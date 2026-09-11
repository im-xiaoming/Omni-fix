#!/usr/bin/env bash
#
# OmniRetriever-7B training launcher.
#
# Required env:
#   WAVE_PATH        path to the WAVE-7B backbone (HuggingFace layout)
#   BEATS_PATH       path to the BEATs audio-encoder checkpoint (.pt)
#   DATA_PATH        path to the training manifest (.jsonl, one record per line)
#
# Optional env (sensible defaults):
#   OUTPUT_DIR       checkpoint output dir              (default: ./output/omniretriever_7b)
#   NUM_GPUS         GPUs to use                        (default: 4)
#   MASTER_PORT      deepspeed master port              (default: 29503)
#   EPOCHS           training epochs                    (default: 1)
#   BATCH_SIZE       per-device micro-batch             (default: 8)
#   GRAD_ACCUM       gradient accumulation steps        (default: 8)
#   LR               learning rate                      (default: 1e-5)
#   LORA_R           LoRA rank                          (default: 16)
#   LORA_ALPHA       LoRA alpha                         (default: 32)
#   VIDEO_BLACKLIST  optional path to a one-id-per-line blacklist file
#
# Media root dirs (optional – only needed when JSONL contains bare filenames):
#   VIDEO_ROOT       root directory for video files  (e.g. /data/videos)
#   AUDIO_ROOT       root directory for audio files  (e.g. /data/audio)
#   IMAGE_ROOT       root directory for image files  (e.g. /data/images)
#
# Example:
#   WAVE_PATH=/data/WAVE-7B \
#   BEATS_PATH=/data/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
#   DATA_PATH=/data/omniretriever_1m.jsonl \
#   bash training/train.sh
#
# Loss components:
#   L_A (pairwise InfoNCE)        — always on when train_classify=True
#   L_D (fusion-as-teacher)       — enabled by train_classify=True + classify_type=all_layer
#                                   (uses the joint forward's anchor as a stop-gradient teacher
#                                    for the single-modal embeddings; see model forward)
#   L_T (Tuple-InfoNCE)           — enabled by --use_tuple_infonce True
#
# To run the pairwise-only ablation pass USE_TUPLE_INFONCE=False below; for an
# L_A-only baseline, also set --classify_type single (single-stream classifier).

set -euo pipefail

# --- required env ---
: "${WAVE_PATH:?Set WAVE_PATH to the WAVE-7B backbone dir}"
: "${BEATS_PATH:?Set BEATS_PATH to the BEATs checkpoint (.pt)}"
: "${DATA_PATH:?Set DATA_PATH to the training manifest (.jsonl)}"

export BEATS_PATH    # picked up by qwenvl.train.train_qwen

# --- optional knobs ---
OUTPUT_DIR="${OUTPUT_DIR:-./output/omniretriever_7b}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_CKPT="${LORA_CKPT:-No}"
LORA_INIT_ONLY="${LORA_INIT_ONLY:-False}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
USE_TUPLE_INFONCE="${USE_TUPLE_INFONCE:-True}"

# Media root dirs — export so data loader picks them up
export VIDEO_ROOT="${VIDEO_ROOT:-}"
export AUDIO_ROOT="${AUDIO_ROOT:-}"
export IMAGE_ROOT="${IMAGE_ROOT:-}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

deepspeed --num_gpus="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/qwenvl/train/train_qwen.py" \
  --deepspeed "${REPO_ROOT}/configs/ds_zero0.json" \
  --model_name_or_path "${WAVE_PATH}" \
  --model_base         "${WAVE_PATH}" \
  --dataset_use        "${DATA_PATH}" \
  --bf16 True \
  --output_dir         "${OUTPUT_DIR}" \
  --num_train_epochs   "${EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate      "${LR}" \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 2048 \
  --dataloader_num_workers 4 \
  --train_classify True \
  --classify_type all_layer \
  --pred_embeds True \
  --use_beats True \
  --tune_beats_proj True \
  --fixed_audio_duration 8 \
  --video_max_frames 8 \
  --video_min_frames 8 \
  --max_pixels 50176 \
  --min_pixels 50176 \
  --use_lora True \
  --lora_r     "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --use_tuple_infonce  "${USE_TUPLE_INFONCE}" \
  --lora_ckpt "${LORA_CKPT}" \
  --lora_init_only "${LORA_INIT_ONLY}" \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 5 \
  --report_to none \
  "$@"
