# Adapted from Tevatron code
import logging
import sys
import torch
import wandb
import pathlib

import transformers
from transformers import HfArgumentParser

from src.utils import print_rank
from src.model import MMEBModel
from src.dataset import UniME_v2_dataset
from src.collator import UniME_v2_data_collector
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.trainer import GradCacheLateProcessTrainer
from src.model_utils import load_processor, get_backbone_name

logger = logging.getLogger(__name__)

def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
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

    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    model = MMEBModel.build(model_args, training_args)
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    processor = load_processor(model_args)
    setattr(model, 'processor', processor)
    
    train_dataset = UniME_v2_dataset(data_args, model_args)
    collator = UniME_v2_data_collector(data_args, model_args, processor)

    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=collator,
        max_length=data_args.max_len
    )
    train_dataset.trainer = trainer

    if ckpt_path:=list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        transformers.logging.set_verbosity_info()
        ckpt_path = sorted(ckpt_path, key=lambda x: int(x.stem.split('-')[-1]))
        trainer.train(resume_from_checkpoint = ckpt_path[-1])
    else:
        trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
