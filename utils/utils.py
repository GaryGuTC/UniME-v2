import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Embedding"))

import re
import torch
from PIL import Image

from transformers import AutoConfig, AutoProcessor

from src.model_utils import get_backbone_name, get_backbone_name, backbone2model, process_vlm_inputs_fns, vlm_image_tokens, PHI3V, process_vlm_rerank_inputs_fns
from src.vlm_backbone.qwen2_5_vl_Rerank import Qwen2_5_VLForConditionalGeneration

def parse_answer_index(answer_text):
    match = re.search(r'\((\d+)\)', answer_text)
    if match:
        return int(match.group(1)) - 1
    return None

def init_model_and_processor(model_name, device, embedding = True): 
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)    

    config.use_cache = False
    config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation = "flash_attention_2"
        
    if embedding:
        model = backbone2model[model_backbone].from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
        )
    
    model.eval()
    model.to(device)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.padding = True
    
    return model, processor

def prepare_stage_data(model_name, processor, txt, img, embedding = True): 
    
    
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=hf_config)
    
    if embedding:
        
        img_prompt = "<|image_1|> Find an image caption describing the given image."    
        text_prompt = ""
        img_prompt = img_prompt.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[model_backbone])
        image = Image.open(img).resize((1344, 1344))
        text = text_prompt + txt
        
        examples = {'text': [img_prompt], 'image': [image]}
        inputs_image = process_vlm_inputs_fns[model_backbone](examples,
                                                              processor = processor,
                                                              max_length = 4096)

        examples = {'text': [text], 'image': [None]}
        inputs_txt = process_vlm_inputs_fns[model_backbone](examples,
                                                            processor = processor,
                                                            max_length = 4096)
        
        return inputs_image, inputs_txt
    else:
        
        img_prompt = "<|image_1|> Find an image caption describing the given image."    
        text_prompt = ""
        img_prompt = img_prompt.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[model_backbone])
        qry_image = Image.open(img).resize((1344, 1344))
        texts = [text_prompt+each for each in txt]
        cand_images = [None for _ in range(len(texts))]
        
        examples = {'qry_text': [img_prompt],
                    'qry_image': [qry_image],
                    'cand_text': [texts],
                    'cand_image': [cand_images]}
        inputs = process_vlm_rerank_inputs_fns[model_backbone](examples,
                                                                processor = processor,
                                                                max_length = 4096,
                                                                pairwise_listwise="listwise")
        return inputs
    
    