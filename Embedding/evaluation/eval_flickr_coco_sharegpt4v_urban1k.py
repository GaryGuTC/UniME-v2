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

from src.model import MMEBModel
from src.model_utils import get_backbone_name, load_processor
from src.utils import print_rank
from src.collator import EvalCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments

from evaluation.utils.utils import recall_at_k, batchify, recall_k_candiadates, gather_object, SequentialDistributedSampler
from evaluation.utils.flickr_coco_sharegpt4v_urban1K import EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K

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

def evaluate_recall(img_features, txt_features, texts_image_index, k_values=(1, 5, 10), output_file="recall_results.txt"):
    """Evaluate recall@k for both image-to-text and text-to-image retrieval"""
    img_features = torch.nn.functional.normalize(img_features, dim=1)
    txt_features = torch.nn.functional.normalize(txt_features, dim=1)
    
    def calculate_recall(query_features, target_features, texts_image_index):
        scores  = query_features @ target_features.t()
        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), texts_image_index] = True
        metrics = {}
        recall_k_list = k_values
        batch_size = 64
        device="cuda"
        for recall_k in recall_k_list:
            metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
            metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()
        
        top10_candidates_T2I = batchify(recall_k_candiadates, scores, positive_pairs, batch_size, device, k=10)
        top10_candidates_I2T = batchify(recall_k_candiadates, scores.T, positive_pairs.T, batch_size, device, k=10)
        
        print(f"top10_candidates_I2T: {top10_candidates_I2T.shape}")
        print(f"top10_candidates_T2I: {top10_candidates_T2I.shape}")
        saved_item = {
            "top10_candidates_I2T": top10_candidates_I2T,
            "top10_candidates_T2I": top10_candidates_T2I 
        }
        torch.save(saved_item, output_file.replace("recall_results.txt", "rerank_top10.pt"))
    
        return metrics
    
    # Image-to-Text & Text-to-Image evaluation
    metrics = calculate_recall(txt_features, img_features, texts_image_index)
    
    print(metrics)
    table_content = [
        "="*80,
        "| Metric       | Recall@1  | Recall@5  | Recall@10 |",
        "|--------------|-----------|-----------|-----------|",
        f"| Image->Text  |-- {metrics['image_retrieval_recall@1']:.3f} -- | -- {metrics['image_retrieval_recall@5']:.3f} -- | -- {metrics['image_retrieval_recall@10']:.3f} -- |",
        f"| Text->Image  |-- {metrics['text_retrieval_recall@1']:.3f} -- | -- {metrics['text_retrieval_recall@5']:.3f} -- | -- {metrics['text_retrieval_recall@10']:.3f} -- |",
        "="*80
    ]
    
    # Save to file
    with open(output_file, "w") as f:
        f.write("\n".join(table_content))
    
    # Print to console (only in rank 0)
    if is_rank_zero():
        print("\n".join(table_content))
    
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
    
    # Initialize datasets
    eval_img_dataset = EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K(
        modality="image",
        model_backbone=model_args.model_backbone,
        data_args=data_args
    )
    eval_txt_dataset = EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K(
        modality="text",
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
    
    # Load model
    model = MMEBModel.load(model_args)
    model.to(local_rank)
    model.eval()
    
    extractor = FeatureExtractor(model, training_args.device)
    
    # Paths for cached features
    encode_img_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_image")
    encode_txt_path = os.path.join(data_args.encode_output_path, f"{data_args.subset_name}_text")
    
    # Extract or load image features
    if not os.path.exists(encode_img_path) or not os.path.exists(encode_txt_path):
        if is_rank_zero(): os.makedirs(data_args.encode_output_path, exist_ok=True)
        
        if not os.path.exists(encode_img_path): 
            img_features = extractor.extract_features(eval_img_loader, f"{data_args.subset_name} Image features")
            gather_object(encode_img_path, img_features.numpy(), is_distributed)
        if not os.path.exists(encode_txt_path): 
            txt_features = extractor.extract_features(eval_txt_loader, f"{data_args.subset_name} Text features")
            gather_object(encode_txt_path, txt_features.numpy(), is_distributed)

    # Synchronize and load features
    if is_distributed: dist.barrier()

    if is_rank_zero():
        with open(encode_img_path, 'rb') as f:
            img_features = pickle.load(f)
            img_features = torch.from_numpy(img_features)
        
        with open(encode_txt_path, 'rb') as f:
            txt_features = pickle.load(f)
            txt_features = torch.from_numpy(txt_features)
        
        if data_args.subset_name == "coco2014" or data_args.subset_name == "flickr30k":
            texts_image_index = [i // 5 for i in range(img_features.shape[0]*5)]
        else:
            texts_image_index = [i for i in range(img_features.shape[0])]

        evaluate_recall(img_features, txt_features, texts_image_index=texts_image_index, k_values=(1, 5, 10), output_file=f"{data_args.encode_output_path}/recall_results.txt")
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()