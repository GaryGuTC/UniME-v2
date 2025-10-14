import torch
import logging

from src.utils import print_master
from typing import Dict, Sequence

from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from src.vlm_backbone.qwen2_5_vl_Rerank.LamRA_qwen2_5_vision_process import process_vision_info_customed
from transformers import LlavaOnevisionForConditionalGeneration

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
LLAVA_OV = 'llava_onevision'
MODEL2BACKBONE = {  # keys are from hf_config.model_type
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
    'llava_onevision': LLAVA_OV,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
    LLAVA_OV: "<image>"
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
    LLAVA_OV: LlavaOnevisionForConditionalGeneration,
}

def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    print_master('Loading processor')
    model_name = model_args.processor_name if model_args.processor_name else model_args.model_name
    if model_args.model_backbone == PHI3V:
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True
        )
    elif model_args.model_backbone == QWEN2_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    elif model_args.model_backbone == QWEN2_5_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28
        )
    elif model_args.model_backbone == LLAVA_OV:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops,
        )
        processor.tokenizer.padding_side = "left"
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # dummy image inputs based on the first valid data point
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
        # make the batch full tensors
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
        image_sizes = torch.cat(image_sizes, dim=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs

def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.stack(pixel_values, dim=0)
        # image_grid_thw = np.concatenate(image_grid_thw, axis=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_grid_thw'] = [None] * input_ids.shape[0]
    return inputs

#########################################################################
######################## Rereank functions start ########################
#########################################################################

def construct_rerank_messages(qry_text=None, qey_image=None, cand_text=None, cand_image=None):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query and a candidate. Please evaluate whether the candidate meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, respond with 'No'."}
                ]
            }
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidate:'}]

        if qey_image != None:
            query.append({'type': 'image', 'image': qey_image})
        if qry_text != None:
            query.append({'type': 'text', 'text': qry_text})
        if cand_image != None:
            cand.append({'type': 'image', 'image': cand_image})
        if cand_text != None:
            cand.append({'type': 'text', 'text': cand_text})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

def construct_rerank_messages_multi_candidates(qry_text=None, qey_image=None, cand_text=None, cand_image=None):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "I will provide you with a query followed by multiple candidates in the format: (1) cand1 (2) cand2, etc. Each candidate is independent of the others. Evaluate each candidate against the query, and respond with the number corresponding to the candidate that best meets the requirements of the query."}
                ]
            },
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidates:'}]

        if qey_image != None:
            query.append({'type': 'image', 'image': qey_image})
        if qry_text != None:
            query.append({'type': 'text', 'text': qry_text})
        for i,(txt,img) in enumerate(zip(cand_text, cand_image)):
            cand.append({'type': 'text', 'text': f'({i + 1}) '})
            txt = txt.replace("<|image_pad|>\n", "").replace("<|image_pad|>", "")
            if img!=None:
                cand.append({'type': 'image', 'image': img})
            if txt!=None:
                cand.append({'type': 'text', 'text': txt})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

def construct_rerank_messages_multi_candidates_training_listwise(qry_text=None, qey_image=None, cand_text=None, cand_image=None):
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Given a query and 5 candidate options (listed as (1) candidate1, (2) candidate2, (3) candidate3, (4) candidate4,(5) candidate5.), evaluate the relevance of each candidate to the query and return their indices in order of descending relevance (most relevant first). Return only the indices of the candidates in sorted order as a list, like [3,1,4,2,5]."}
                ]
            },
        ]
        query = [{'type': 'text', 'text': 'Query:'}]
        cand = [{'type': 'text', 'text': 'Candidates:'}]

        if qey_image != None:
            query.append({'type': 'image', 'image': qey_image})
        if qry_text != None:
            query.append({'type': 'text', 'text': qry_text})
        for i,(txt,img) in enumerate(zip(cand_text, cand_image)):
            cand.append({'type': 'text', 'text': f'({i + 1}) '})
            txt = txt.replace("<|image_pad|>\n", "").replace("<|image_pad|>", "")
            if img!=None:
                cand.append({'type': 'image', 'image': img})
            if txt!=None:
                cand.append({'type': 'text', 'text': txt})

        for item in query:
            message[0]['content'].append(item)

        for item in cand:
            message[0]['content'].append(item)

        return message

def appply_chat_template(image=None, text=None):
    if image != None:
        conversation_image = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text.replace("<image>\n", "").replace("<image>", "").strip()},
                    ],
            }]
    else:
        conversation_image = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text.replace("<image>\n", "").replace("<image>", "").strip()},
                    ],
            }]
    return conversation_image

def Llava_OV_process(model_inputs: dict, processor, max_length=None):
    texts, images = model_inputs['text'], model_inputs['image']
    coversations = [appply_chat_template(image, text) for text, image in zip(texts, images)]
    inputs = processor.apply_chat_template(
        coversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    return inputs

def Qwen2_5_VL_rerank_process_fn(model_inputs: Sequence[Dict], processor, max_length=None, pairwise_listwise="pairwise", rerank_train_liswise=False) -> Dict[str, torch.Tensor]:
    IGNORE_TOKEN_ID = -100 # processor.tokenizer.ignore_token_id
    PAD_TOKEN_ID = processor.tokenizer.pad_token_id

    messages = []
    qry_texts, qey_images, cand_texts, cand_images = \
            model_inputs['qry_text'], model_inputs['qey_image'], model_inputs['cand_text'], model_inputs['cand_image']
    
    for qry_text, qey_image, cand_text, cand_image  in zip(qry_texts, qey_images, cand_texts, cand_images):
        qry_text = qry_text.replace("<|image_pad|>\n", "").replace("<|image_pad|>", "")
        if pairwise_listwise=="pairwise":
            cand_text = cand_text.replace("<|image_pad|>\n", "").replace("<|image_pad|>", "")
            messages.append(construct_rerank_messages(qry_text, qey_image, cand_text, cand_image))
        else:
            if rerank_train_liswise:
                messages.append(construct_rerank_messages_multi_candidates_training_listwise(qry_text, qey_image, cand_text, cand_image))
            else:    
                messages.append(construct_rerank_messages_multi_candidates(qry_text, qey_image, cand_text, cand_image))

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    ########################################################
    image_inputs, video_inputs = process_vision_info_customed(messages) #! 这边会resize image
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs['input_ids']
    labels = input_ids.clone()
    labels[labels == PAD_TOKEN_ID] = IGNORE_TOKEN_ID

    if 'attention_mask' in inputs:
        attention_mask = inputs['attention_mask']
    else:
        attention_mask = None 
    if 'pixel_values' in inputs:
        pixel_values = inputs['pixel_values']
    else:
        pixel_values = None 
    if 'image_grid_thw' in inputs:
        image_grid_thw = inputs['image_grid_thw']
    else:
        image_grid_thw = None 
    
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        labels=labels,
    ) 

process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
    LLAVA_OV: Llava_OV_process,
}

process_vlm_rerank_inputs_fns = {
    PHI3V: None,
    LLAVA_NEXT: None,
    QWEN2_VL: None,
    QWEN2_5_VL: Qwen2_5_VL_rerank_process_fn,
}
