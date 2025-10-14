#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import HfArgumentParser, AutoConfig, AutoProcessor

from src.utils import print_rank
from src.collator import EvalCollator_rerank
from src.model_utils import get_backbone_name
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.vlm_backbone.qwen2_5_vl_Rerank import Qwen2_5_VLForConditionalGeneration

from evaluation.utils.utils import gather_object, SequentialDistributedSampler
from evaluation.utils.sugarcrepe import EvalDataset4sugarcrepe_rerank

    
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
        features = []
        for batch in tqdm(dataloader, desc=desc, disable=not is_rank_zero()):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                output = self.model.generate(**batch, max_new_tokens=128, output_scores=True, return_dict_in_generate=True, do_sample=False)
            logits = output.scores[0] 
            scores = []
            for idx in range(len(logits)):
                probs = (torch.nn.functional.softmax( 
                            torch.FloatTensor([      
                                logits[idx][self.processor.tokenizer("Yes").input_ids[0]], 
                                logits[idx][self.processor.tokenizer("No").input_ids[0]]   
                            ]),dim=0).detach().cpu().numpy()
                        )
                scores.append(probs[0])
            features += scores
        return features

def main():
    try:
        local_rank = setup_distributed()
        is_distributed = True
    except:
        local_rank = 0
        is_distributed = False
    
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.exists(f"{data_args.encode_output_path}/recall_results_rerank.txt"): 
        print_rank(f"{data_args.encode_output_path} bas been processed")
        return
    
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)

    data_args.subset_name = data_args.subset_name[0]
    
    if is_rank_zero():
        print_rank(f'model_backbone: {model_backbone}')

    # Load model
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
    
    compositioanl_names = ['replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att', 'add_obj', 'add_att']
    for composition_name in compositioanl_names:
        data_args.subset_name = composition_name
        
        eval_dataset = EvalDataset4sugarcrepe_rerank(
            model_backbone=model_args.model_backbone,
            data_args=data_args
        )
        
        eval_collator = EvalCollator_rerank(
            data_args=data_args,
            model_args=model_args,
            processor=processor,
        )
        
        eval_dataset_loader = DataLoader(
            eval_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            sampler=SequentialDistributedSampler(eval_dataset) if is_distributed else None,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=False if model_backbone == "llava_onevision" else True
        )
        
        extractor = FeatureExtractor(model, training_args.device, processor=processor)
        
        # Paths for cached features
        encode_txt_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_rerank")
        
        # Extract or load image features
        if not os.path.exists(encode_txt_path):
            if is_rank_zero(): os.makedirs(data_args.encode_output_path, exist_ok=True)
            
            if not os.path.exists(encode_txt_path): 
                txt_pos_features = extractor.extract_features(eval_dataset_loader, f"{data_args.subset_name} Rerank")
                gather_object(encode_txt_path, np.array(txt_pos_features), is_distributed)

    # Synchronize and load features
    if is_distributed: dist.barrier()

    if is_rank_zero():
        ans_dict = {}
        for composition_name in compositioanl_names:
            encode_txt_pos_path = os.path.join(data_args.encode_output_path, f"{composition_name}_rerank")
            
            with open(encode_txt_pos_path, 'rb') as f:
                txt_pos_features = pickle.load(f)
            ans = 0

            for i in range(0, len(txt_pos_features), 2):
                if txt_pos_features[i] > txt_pos_features[i+1]: ans+=1

            total_num = len(txt_pos_features)/2
            ans_dict[composition_name] = ans/total_num

        output_file=f"{data_args.encode_output_path}/recall_results_rerank.txt"
        ans_dict['replace_avg'] = (ans_dict['replace_obj'] + ans_dict['replace_att'] + ans_dict['replace_rel']) / 3
        ans_dict['swap_avg'] = (ans_dict['swap_obj'] + ans_dict['swap_att']) / 2
        ans_dict['add_avg'] = (ans_dict['add_obj'] + ans_dict['add_att']) / 2

        table_content = [
            "="*40,
            "| Metric       | score  | ",
            "|--------------|-----------|",
            f"| replace obj  |-- {ans_dict['replace_obj']:.3f} -- |",
            f"| replace att  |-- {ans_dict['replace_att']:.3f} -- |",
            f"| replace rel  |-- {ans_dict['replace_rel']:.3f} -- |",
            f"| swap    obj  |-- {ans_dict['swap_obj']:.3f} -- |",
            f"| swap    att  |-- {ans_dict['swap_att']:.3f} -- |",
            f"| add     obj  |-- {ans_dict['add_obj']:.3f} -- |",
            f"| add     att  |-- {ans_dict['add_att']:.3f} -- |",
            "|--------------|-----------|",
            f"| replace avg  |-- {ans_dict['replace_avg']:.3f} -- |",
            f"| swap    avg  |-- {ans_dict['swap_avg']:.3f} -- |",
            f"| add     avg  |-- {ans_dict['add_avg']:.3f} -- |",
            "="*40
        ]
        
        with open(output_file, "w") as f:
            f.write("\n".join(table_content))
        
        if is_rank_zero():
            print("\n".join(table_content))

        return ans_dict
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()