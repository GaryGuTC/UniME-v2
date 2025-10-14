# ======================
#  Configuration
# ======================
# Evaluation Parameters
BATCH_SIZE=4
IMAGE_RESOLUTION="high"
MASTER_PORT_BASE=22290  # Base port for torchrun
MAX_LEN=4096

# Dataset Subsets
SUBSET_LIST=(
  "ImageNet-1K N24News HatefulMemes VOC2007 SUN397 A-OKVQA MSCOCO"
  "Place365 ImageNet-A ImageNet-R ObjectNet Country211 OK-VQA"
  "DocVQA InfographicsVQA ChartQA NIGHTS FashionIQ"
  "ScienceQA Visual7W VizWiz GQA TextVQA VisDial"
  "CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t"
  "Wiki-SS-NQ"
  "WebQA OVEN EDIS"
  "RefCOCO-Matching Visual7W-Pointing RefCOCO"
)

RETRIEVAL_SUBSET_LIST=("flickr30k" "coco2014" "sharegpt4v" "Urban200K")
COMPOSITIONAL_SUBSET_LIST=("sugarcrepe")

# Model Checkpoints
file_name="UniME-V2_LLaVA_onevision_8B" #! Choose to Edit
lora_path="None"
EXPERIMENTS_LIST=(
  ${lora_path}
)

base_model_name=MODEL_PATH #! Edit
# Path Configuration
ENCODE_OUTPUT_PATH="evaluate_result/${file_name}"
MMEB_EVAL_DATA_PATH=MMEB_EVAL_PATH #! Edit, example: /data/MMEB_eval

# GPU Allocation
GPU_IDS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}
GPU_INDEX=0  # Tracks GPU assignment

MMEB_TEST=true
RETRIEVAL_TEST=true
COMPOSITIONAL=true

# ======================
#  Helper Functions
# ======================
print_success() {
    echo -e "\033[32m[SUCCESS] $1\033[0m"
}

print_error() {
    echo -e "\033[31m[ERROR] $1\033[0m"
}

print_warning() {
    echo -e "\033[33m[WARNING] $1\033[0m"
}
# ======================
#  Core Functions
# ======================
run_experiment() {
    local checkpoint_path=$1
    local gpu_id=$2
    local subset_id=$3

    local ckpt_name=$(basename "$checkpoint_path")
    local exp_dir=$(dirname "$checkpoint_path")
    local exp_basename=$(basename "$exp_dir")
    
    echo -e "\n=== Starting Evaluation ==="
    echo "GPU: $gpu_id | Subset: ${SUBSET_LIST[$subset_id]}"
    echo "Model: $exp_basename/$ckpt_name"

    # Construct evaluation command
    local cmd=(
        "CUDA_VISIBLE_DEVICES=$gpu_id"
        "torchrun --nproc_per_node=1"
        "--master_port=$((MASTER_PORT_BASE + gpu_id))"
        "--max_restarts=0"
        "eval.py"
        "--model_name $base_model_name"
        "--encode_output_path $ENCODE_OUTPUT_PATH/MMEB_eval"
        "--max_len $MAX_LEN"
        "--pooling eos"
        "--normalize True"
        "--dataset_name ${MMEB_EVAL_DATA_PATH}/MMEB-eval"
        "--subset_name ${SUBSET_LIST[$subset_id]}"
        "--dataset_split test"
        "--image_resolution $IMAGE_RESOLUTION"
        "--per_device_eval_batch_size $BATCH_SIZE"
        "--image_dir ${MMEB_EVAL_DATA_PATH}/images"
    )
    
    # Execute command
    if ! eval "${cmd[@]}"; then
        print_error "Evaluation failed on GPU $gpu_id"
        return 1
    fi
}

# ======================
#  Main Execution
# ======================
print_success "Environment validated"

if [ "$MMEB_TEST" = true ]; then
  for checkpoint_path in "${EXPERIMENTS_LIST[@]}"; do

      for subset_id in "${!SUBSET_LIST[@]}"; do
          current_gpu=${GPU_IDS[$GPU_INDEX]}
          
          echo -e "\n[$(date +'%T')] Assigning to GPU $current_gpu: Subset $((subset_id+1))/${#SUBSET_LIST[@]}"
          run_experiment "$checkpoint_path" $current_gpu $subset_id &
          
          Update GPU assignment
          GPU_INDEX=$(( (GPU_INDEX + 1) % NUM_GPUS ))
          
          # Wait when all GPUs are busy
          if [ $GPU_INDEX -eq 0 ]; then
              echo -e "\n[$(date +'%T')] Waiting for current batch..."
              wait
          fi
      done
  done
  # Final wait for remaining jobs
  wait
  python evaluation/eval_MMEB.py --checkpoint_path "$ENCODE_OUTPUT_PATH/MMEB_eval" --output_path "$ENCODE_OUTPUT_PATH/MMEB_eval_conclude"
  echo -e "\n\033[1;32mAll evaluations of MMEB completed!\033[0m"

fi


if [ "$RETRIEVAL_TEST" = true ]; then 
  for dataset in "${RETRIEVAL_SUBSET_LIST[@]}"; do
    echo -e "\n\033[1;34m##### Testing ${dataset} #####\033[0m"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        evaluation/eval_flickr_coco_sharegpt4v_urban1k.py \
        --model_name "$base_model_name" \
        --encode_output_path "$ENCODE_OUTPUT_PATH/${dataset}" \
        --max_len $MAX_LEN \
        --pooling eos \
        --normalize True \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --subset_name "$dataset" \
        --dataset_split test
  done
  wait
  echo -e "\n\033[1;32mAll evaluations of retrieval completed!\033[0m"
fi


if [ "$COMPOSITIONAL" = true ]; then 
  for dataset in "${COMPOSITIONAL_SUBSET_LIST[@]}"; do
    echo -e "\n\033[1;34m##### Testing ${dataset} #####\033[0m"
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
        evaluation/eval_sugar_crepe.py \
        --model_name "$base_model_name" \
        --encode_output_path "$ENCODE_OUTPUT_PATH/${dataset}" \
        --max_len $MAX_LEN \
        --pooling eos \
        --normalize True \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --subset_name "$dataset" \
        --dataset_split test
  done
  wait
  echo -e "\n\033[1;32mAll evaluations of compositional completed!\033[0m"
fi