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
MODEL_ID=qwen2_5-vl-7b   
model_name=PATH_TO_BACKBONE_MODEL #! Edit, example: /model/Qwen2.5-VL-7B-Instruct
output_dir=output/${run_name}
dataset_name=PATH_TO_MMEB_IMAGE_PATH #! Edit, example: /data/MMEB_train
MMEB_train_data_path=PATH_TO_DOWNLOAD_JSON_FILE.json #! Edit, /data/train_data_qwen257B_scores.json

### Image settings
image_resolution=mid_336
max_len=4096


# LamRA training setting
TRAIN_VISION_ENCODER=False                              
USE_VISION_LORA=False                                   
TRAIN_VISION_PROJECTOR=False                           
USE_LORA=True                                           
Q_LORA=False                                           
LORA_R=128                                            
LORA_ALPHA=256                                   

DS_STAGE=zero2                                         
PER_DEVICE_BATCH_SIZE=1                               
GRAD_ACCUM=4                                            
NUM_EPOCHS=1                                           

LR=2e-5                                                 
MODEL_MAX_LEN=4096                      

torchrun $DISTRIBUTED_ARGS train/train_rerank.py \
    --model_id $MODEL_ID \
    --training_data_path $MMEB_train_data_path \
    --dataset_name $dataset_name \
    --output_dir $output_dir \
    --deepspeed ./ds_configs/${DS_STAGE}.json \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LEN \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --train_vision_encoder $TRAIN_VISION_ENCODER \
    --use_vision_lora $USE_VISION_LORA \
    --train_vision_projector $TRAIN_VISION_PROJECTOR \
    --use_lora $USE_LORA \
    --q_lora $Q_LORA \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --model_local_path $model_name \
    --use_flash_attn True \
    --report_to wandb \
    --run_name $run_name \
    --project_name $project_name \