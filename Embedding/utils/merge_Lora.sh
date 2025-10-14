export CUDA_HOME=/usr/local/cuda-12

model_name=BACKBONE_MODEL_PATH
lora_path=LORA_PATH

# Merge Embedding Model
python merge_Lora.py \
    --model_name $model_name \
    --checkpoint_path $lora_path \
	--lora

# Merge Rerank Model
python merge_Lora_rerank.py \
    --model_name $model_name \
    --checkpoint_path $lora_path \
	--lora