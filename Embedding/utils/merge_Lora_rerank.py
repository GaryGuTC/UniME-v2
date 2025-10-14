import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil

import torch
from torch import nn

from peft import LoraConfig, PeftModel

from transformers.hf_argparser import HfArgumentParser
from transformers import AutoProcessor, PreTrainedModel, AutoModelForCausalLM, AutoConfig

from src.arguments import DataArguments, ModelArguments, TrainingArguments
from src.vlm_backbone.qwen2_5_vl_Rerank import Qwen2_5_VLForConditionalGeneration
from src.model_utils import get_backbone_name, print_master

class MMEBModel(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.TRANSFORMER_CLS = AutoModelForCausalLM

    @classmethod
    def load(cls, model_args: ModelArguments, **kwargs):
        # Loading the base model
        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        config.use_cache = False
        model_backbone = get_backbone_name(hf_config=config)
        setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_backbone}]')


        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2"
        config.vision_config._attn_implementation = "flash_attention_2"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
        )

        lora_config = LoraConfig.from_pretrained(checkpoint_path)
        lora_model = PeftModel.from_pretrained(model, checkpoint_path, config=lora_config)
        model = lora_model.merge_and_unload()
                
        model = cls(
            encoder=model,
        )

        return model


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    hf_config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    model = MMEBModel.load(model_args).to("cpu", dtype=torch.bfloat16)
    model.eval()

    # merged_model.save_pretrained
    saved_path = model_args.checkpoint_path+"-merged"
    os.makedirs(saved_path, exist_ok=True)
    model.encoder.save_pretrained(saved_path)
    processor.save_pretrained(saved_path)

    # copy the chat_template.json file
    source_chat_file = os.path.join(model_args.model_name, "chat_template.json")
    target_chat_file = os.path.join(saved_path, "chat_template.json")
    shutil.copy(source_chat_file, target_chat_file)

    print("\033[91m" + f"Merged model saved to {saved_path}" + "\033[0m")

if __name__ == "__main__":
    main()