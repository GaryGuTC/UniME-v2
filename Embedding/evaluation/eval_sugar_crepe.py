#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from transformers import HfArgumentParser, AutoConfig

from src.utils import print_rank
from src.model import MMEBModel
from src.collator import EvalCollator
from src.model_utils import get_backbone_name, load_processor
from src.arguments import ModelArguments, DataArguments, TrainingArguments

from evaluation.utils.utils import gather_object, SequentialDistributedSampler
from evaluation.utils.sugarcrepe import EvalDataset4sugarcrepe


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
    def __init__(self, model, device, dtype=torch.bfloat16):
        self.model = model.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        
    @torch.no_grad()
    def extract_features(self, dataloader, desc="Extracting"):
        features = []
        for batch in tqdm(dataloader, desc=desc, disable=not is_rank_zero()):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = self.model(qry=batch)
            features.append(outputs["qry_reps"].cpu().float())
        return torch.cat(features)
    
    
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

    if os.path.exists(f"{data_args.encode_output_path}/recall_results.txt"): 
        print_rank(f"{data_args.encode_output_path} bas been processed")
        return
    
    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)

    # Set up model and data
    processor = load_processor(model_args)
    data_args.subset_name = data_args.subset_name[0]
    
    if is_rank_zero():
        print_rank(f'model_backbone: {model_backbone}')

    # Load model
    model = MMEBModel.load(model_args)
    model.to(local_rank)
    model.eval()
    
    compositioanl_names = ['replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att', 'add_obj', 'add_att']
    for composition_name in compositioanl_names:
        # Initialize datasets
        data_args.subset_name = composition_name
        eval_img_dataset = EvalDataset4sugarcrepe(
            modality="image",
            model_backbone=model_args.model_backbone,
            data_args=data_args
        )
        
        eval_txt_pos_dataset = EvalDataset4sugarcrepe(
            modality="text_pos",
            model_backbone=model_args.model_backbone,
            data_args=data_args
        )

        eval_txt_neg_dataset = EvalDataset4sugarcrepe(
            modality="text_neg",
            model_backbone=model_args.model_backbone,
            data_args=data_args
        )
        
        eval_collator = EvalCollator(
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
        
        eval_txt_pos_loader = DataLoader(
            eval_txt_pos_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            sampler=SequentialDistributedSampler(eval_txt_pos_dataset) if is_distributed else None,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=False if model_backbone == "llava_onevision" else True
        )

        eval_txt_neg_loader = DataLoader(
            eval_txt_neg_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            sampler=SequentialDistributedSampler(eval_txt_neg_dataset) if is_distributed else None,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=False if model_backbone == "llava_onevision" else True
        )
        
        extractor = FeatureExtractor(model, training_args.device)
        
        # Paths for cached features
        encode_img_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_image")
        encode_txt_pos_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_text_pos")
        encode_txt_neg_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_text_neg")
        
        # Extract or load image features
        if not os.path.exists(encode_img_path) or not os.path.exists(encode_txt_pos_path) or not os.path.exists(encode_txt_neg_path):
            if is_rank_zero():
                os.makedirs(data_args.encode_output_path, exist_ok=True)
            
            if not os.path.exists(encode_img_path): 
                img_features = extractor.extract_features(eval_img_loader, f"{data_args.subset_name} Image features")
                gather_object(encode_img_path, img_features.numpy(), is_distributed)
            if not os.path.exists(encode_txt_pos_path): 
                txt_pos_features = extractor.extract_features(eval_txt_pos_loader, f"{data_args.subset_name} Text features")
                gather_object(encode_txt_pos_path, txt_pos_features.numpy(), is_distributed)
            if not os.path.exists(encode_txt_neg_path):
                txt_neg_features = extractor.extract_features(eval_txt_neg_loader, f"{data_args.subset_name} Text features")
                gather_object(encode_txt_neg_path, txt_neg_features.numpy(), is_distributed)

    # Synchronize and load features
    if is_distributed: dist.barrier()

    if is_rank_zero():
        ans_dict = {}
        for composition_name in compositioanl_names:
            encode_img_path = os.path.join(data_args.encode_output_path, f"{composition_name}_image")
            encode_txt_pos_path = os.path.join(data_args.encode_output_path, f"{composition_name}_text_pos")
            encode_txt_neg_path = os.path.join(data_args.encode_output_path, f"{composition_name}_text_neg")
            
            with open(encode_img_path, 'rb') as f:
                img_features = pickle.load(f)
                img_features = torch.from_numpy(img_features)
            
            with open(encode_txt_pos_path, 'rb') as f:
                txt_pos_features = pickle.load(f)
                txt_pos_features = torch.from_numpy(txt_pos_features)
            
            with open(encode_txt_neg_path, 'rb') as f:
                txt_neg_features = pickle.load(f)
                txt_neg_features = torch.from_numpy(txt_neg_features)
            
            pos_score = txt_pos_features @ img_features.t()
            neg_score = txt_neg_features @ img_features.t()
            
            pos_score = pos_score.diagonal()
            neg_score = neg_score.diagonal()
        
            ans = sum([1 if a.item() > b.item() else 0 for a,b in zip(pos_score, neg_score)])
            ans_dict[composition_name] = ans/txt_pos_features.shape[0]

        output_file=f"{data_args.encode_output_path}/recall_results.txt"
        # 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att', 'add_obj', 'add_att'
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
        
        # Save to file
        with open(output_file, "w") as f:
            f.write("\n".join(table_content))
        
        # Print to console (only in rank 0)
        if is_rank_zero():
            print("\n".join(table_content))

        return ans_dict
        
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()