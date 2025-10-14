#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import re
import json

from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import HfArgumentParser, AutoConfig, AutoProcessor

from src.utils import print_rank
from src.model_utils import get_backbone_name
from src.collator import EvalCollator_rerank_listwise
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.vlm_backbone.qwen2_5_vl_Rerank import Qwen2_5_VLForConditionalGeneration

from evaluation.utils.utils import gather_object, SequentialDistributedSampler
from evaluation.utils.flickr_coco_sharegpt4v_urban1K import EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K_rerank


def parse_answer_index(answer_text):
    match = re.search(r'\((\d+)\)', answer_text)
    if match:
        return int(match.group(1)) - 1
    return None

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True

def setup_distributed():
    """Initialize distributed training environment"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

class FeatureExtractor:
    def __init__(self, model, device, processor, dtype=torch.bfloat16):
        self.model = model.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.processor = processor
        
    @torch.no_grad()
    def extract_features(self, dataloader, desc="Extracting"):
        # features, names = [], []
        encoded_tensor = []
        for batch in tqdm(dataloader, desc=desc, disable=not is_rank_zero()):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                outputs = self.model.generate(**batch, max_new_tokens=128, output_scores=True, return_dict_in_generate=True, do_sample=False)
            generated_ids = outputs.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch['input_ids'], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            encoded_tensor += output_text
        return encoded_tensor
    
def main():
    # Initialize distributed training if available
    try:
        local_rank = setup_distributed()
        is_distributed = True
    except:
        local_rank = 0
        is_distributed = False
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    saved_image2text_score_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name[0]}_image2text_rerank.json")
    saved_text2image_score_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name[0]}_text2image_rerank.json")
    
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)

    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    
    # Set up model and data
    if model_backbone == "qwen2_5_vl":
        processor = AutoProcessor.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
    
    data_args.subset_name = data_args.subset_name[0]
    
    if is_rank_zero():
        print_rank(f'model_backbone: {model_backbone}')
    
    # Initialize datasets
    eval_img_dataset = EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K_rerank(
        modality="image",
        model_backbone=model_args.model_backbone,
        data_args=data_args
    )
    eval_txt_dataset = EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K_rerank(
        modality="text",
        model_backbone=model_args.model_backbone,
        data_args=data_args
    )
    
    eval_collator = EvalCollator_rerank_listwise(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )
    
    # Create data loaders
    eval_img_loader = DataLoader(
        eval_img_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        sampler=SequentialDistributedSampler(eval_img_dataset) if is_distributed else None,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=False if model_backbone == "llava_onevision" else True
    )
    
    eval_txt_loader = DataLoader(
        eval_txt_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=eval_collator,
        sampler=SequentialDistributedSampler(eval_txt_dataset) if is_distributed else None,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=False if model_backbone == "llava_onevision" else True
    )
    
    config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation = "flash_attention_2"
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
    )

    if model_args.lora:
        from peft import LoraConfig, PeftModel
        lora_config = LoraConfig.from_pretrained(model_args.checkpoint_path)
        lora_model = PeftModel.from_pretrained(model, model_args.checkpoint_path, config=lora_config)
        model = lora_model.merge_and_unload()
    
    model.to(local_rank)
    model.eval()
    
    extractor = FeatureExtractor(model, training_args.device, processor=processor)
    
    # Paths for cached features
    encode_img_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_image2text_rerank")
    encode_txt_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_text2image_rerank")
    
    # Extract or load image features
    if not os.path.exists(encode_img_path) or not os.path.exists(encode_txt_path):
        if is_rank_zero():
            os.makedirs(data_args.encode_output_path, exist_ok=True)
        
        if not os.path.exists(encode_img_path): 
            img_features = extractor.extract_features(eval_img_loader, f"{data_args.subset_name} Image Rerank")
            gather_object(encode_img_path, img_features, is_distributed, rerank=True)
        
                
        if not os.path.exists(encode_txt_path): 
            txt_features = extractor.extract_features(eval_txt_loader, f"{data_args.subset_name} Text Rerank")
            gather_object(encode_txt_path, txt_features, is_distributed, rerank=True)
            
    # Synchronize and load features
    if is_distributed: dist.barrier()

    if is_rank_zero():
        with open(encode_img_path, 'rb') as f: img_features = pickle.load(f)
        with open(encode_txt_path, 'rb') as f: txt_features = pickle.load(f)
        
        if not os.path.exists(saved_image2text_score_path):
            n_correct = 0
            for idx,score in enumerate(img_features):
                if data_args.subset_name == "coco2014" or data_args.subset_name == "flickr30k":
                    corresponding_idx = idx*5
                else:
                    corresponding_idx = idx
                select_idx = parse_answer_index(score)
                try:
                    temp_set = [eval_img_dataset.text_data[idx] for idx in range(corresponding_idx, corresponding_idx+5)]
                    if eval_img_dataset.eval_data[idx]["cand_data"][select_idx] in temp_set:
                        n_correct += 1
                except:
                    continue

            with open(saved_image2text_score_path, "w") as f:
                score_dict = {"acc": n_correct/len(eval_img_dataset.eval_data), "num_correct": n_correct, "num_pred": len(eval_img_dataset.eval_data)}
                json.dump(score_dict, f, indent=4)
            print(f"\033[91m Image2text accuracy: {n_correct/len(eval_img_dataset.eval_data)}\033[0m")
  
        if not os.path.exists(saved_text2image_score_path):
            n_correct = 0 # eval_img_dataset.eval_data | eval_txt_dataset.eval_data
            for idx,score in enumerate(txt_features):
                if data_args.subset_name == "coco2014" or data_args.subset_name == "flickr30k":
                    corresponding_idx = idx//5
                else:
                    corresponding_idx = idx
                select_idx = parse_answer_index(score)
                try:
                    if eval_txt_dataset.eval_data[idx]["cand_data"][select_idx] == eval_txt_dataset.image_data[corresponding_idx]:
                        n_correct += 1
                except:
                    continue

            with open(saved_text2image_score_path, "w") as f:
                score_dict = {"acc": n_correct/len(eval_txt_dataset.eval_data), "num_correct": n_correct, "num_pred": len(eval_txt_dataset.eval_data)}
                json.dump(score_dict, f, indent=4)
            print(f"\033[91m Text2image accuracy: {n_correct/len(eval_txt_dataset.eval_data)}\033[0m")
    
    if is_distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()