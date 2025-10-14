#!/bin/bash

export MLP_WORKER_NUM=1
export MLP_WORKER_GPU=8
export MLP_ROLE_INDEX=0
export MLP_WORKER_0_HOST="localhost"
export MLP_WORKER_0_PORT="29500"

DISTRIBUTED_ARGS="
    --nnodes $MLP_WORKER_NUM \
    --nproc_per_node $MLP_WORKER_GPU \
    --node_rank $MLP_ROLE_INDEX \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
"

export WANDB_BASE_URL= #! Edit or delete
export WANDB_API_KEY= #! Edit or delete
export HF_HOME= #! Edit or delete
export CUTLASS_PATH= #! Edit or delete
wandb online

project_name=WANDB_PROJECT_NAME #! Edit
run_name=WANDB_RUN_NAME #! Edit
### Model settings
model_name=PATH_TO_BACKBONE_MODEL #! Edit, example: /model/llava-onevision-qwen2-7b-ov-hf
image_dir=PATH_TO_MMEB_IMAGE_PATH #! Edit, example: /data/MMEB_train
hard_negaitve_data_path=PATH_TO_DOWNLOAD_JSON_FILE.json #! Edit, Example: /data/train_data_qwen257B_scores.json
output_dir=output/${run_name}
### Image settings
image_resolution=mid_336
max_len=4096

### Training settings
lr=1e-4
max_steps=2000
warmup_steps=200
save_steps=50
per_device_train_batch_size=64 # total batch size = per_device_train_batch_size * MLP_WORKER_NUM * MLP_WORKER_GPU 

torchrun $DISTRIBUTED_ARGS train.py \
  --model_name $model_name \
  --output_dir $output_dir \
  --bf16 --pooling last \
  --lora \
  --image_dir $image_dir \
  --image_resolution $image_resolution \
  --max_len $max_len \
  --logging_steps 1 \
  --lr_scheduler_type linear \
  --learning_rate $lr \
  --warmup_steps $warmup_steps \
  --save_steps $save_steps \
  --normalize True \
  --temperature 0.02 \
  --per_device_train_batch_size $per_device_train_batch_size \
  --grad_cache True \
  --gc_q_chunk_size 2 \
  --gc_p_chunk_size 2 \
  --save_safetensors True \
  --remove_unused_columns False \
  --report_to wandb \
  --project_name $project_name \
  --num_train_epochs 1 \
  --max_steps $max_steps \
  --run_name $run_name \
  --save_total_limit 50 \
  --save_on_each_node True \
  --resume_from_checkpoint True \
  --hard_negaitve_data_path $hard_negaitve_data_path \
  --select_hard_negative_num_4_training 8 \
  --select_hard_negative_num 8