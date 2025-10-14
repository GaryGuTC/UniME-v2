import json
import sys

import os
import re
import torch
import pickle
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoProcessor, AutoConfig

from src.utils import print_rank
from src.model_utils import get_backbone_name
from src.dataset import EvalDataset_rerank
from src.collator import EvalCollator_rerank_listwise
from src.vlm_backbone.qwen2_5_vl_Rerank import Qwen2_5_VLForConditionalGeneration
from src.arguments import ModelArguments, DataArguments, TrainingArguments

def parse_answer_index(answer_text):
    match = re.search(r'\((\d+)\)', answer_text)
    if match:
        return int(match.group(1)) - 1
    return None

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

def tensors_to_device(data, device, dtype):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            if key == 'pixel_values':
                data[key] = data[key].to(device).to(dtype)
            else:
                data[key] = data[key].to(device)
    return data 

def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.rerank_output_path, exist_ok=True)

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)

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

    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')

    config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation = "flash_attention_2"
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

    model = model.to(training_args.device, dtype=torch.bfloat16)
    model.eval()

    eval_collator = EvalCollator_rerank_listwise(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for subset_idx, subset in enumerate(data_args.subset_name):
        save_path_subset = os.path.join(data_args.rerank_output_path, f"{subset}_rerank_scores")
        if os.path.exists(save_path_subset): continue
        eval_qry_dataset = EvalDataset_rerank(
            data_args=data_args,
            model_args=model_args,
            subset=subset
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc=f"[{subset_idx+1}/{len(data_args.subset_name)}] Rerank Listwise {subset}"):
                batch = tensors_to_device(batch, training_args.device, model.dtype)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    outputs = model.generate(**batch, max_new_tokens=128, output_scores=True, return_dict_in_generate=True, do_sample=False)
                generated_ids = outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(batch['input_ids'], generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                encoded_tensor += output_text
                
        with open(save_path_subset, 'wb') as f:
            pickle.dump(encoded_tensor, f)

    for subset in tqdm(data_args.subset_name, desc=f"calculate score {subset}"):
        saved_subset_score_path = os.path.join(data_args.rerank_output_path, f"{subset}_rerank_scores_final.json")
        if os.path.exists(saved_subset_score_path): continue
        loaded_tensor = pickle.load((open(os.path.join(data_args.rerank_output_path, f"{subset}_rerank_scores"), 'rb')))
        subset_data = json.load(open(os.path.join(data_args.encode_output_path, f"{subset}_rerank_topk.json")))
        n_correct = 0
        for each,score in zip(subset_data, loaded_tensor):
            select_idx = parse_answer_index(score)
            try:
                if each["candidates_topk"][select_idx] == [each["tgt_text"], each["tgt_img_path"]]:
                    n_correct += 1
            except:
                continue

        with open(saved_subset_score_path, "w") as f:
            score_dict = {"acc": n_correct/len(subset_data), "num_correct": n_correct, "num_pred": len(subset_data)}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/len(subset_data)}\033[0m")

if __name__ == "__main__":
    main()
