import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.distributed as dist
from torch.utils.data import Dataset
from datasets import load_dataset

from src.model_utils import PHI3V, vlm_image_tokens
from src.dataset import process_image

from evaluation.utils.sharegpt4v import share4v_val_dataset_4_LLM
from evaluation.utils.urban1k import urban1k_dataset_4_LLM
from evaluation.utils.utils import dataset_2_HFdataset
from evaluation.utils.data_path import coco as COCO
from evaluation.utils.data_path import flickr30K as FLICKR30K

class EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K(Dataset):
    def __init__(self, modality, model_backbone, data_args):
        self.model_backbone = model_backbone
        self.data_args = data_args
        self.modality = modality
        if data_args.subset_name == "coco2014":
            self.raw_data = load_dataset(COCO["data_path"], split='test')
            self.raw_data = self.raw_data.map(lambda x: {'text': x['text'][:5]}, num_proc=16)
        elif data_args.subset_name == "flickr30k":
            self.raw_data = load_dataset(FLICKR30K["data_path"], split='test')
        elif data_args.subset_name == "sharegpt4v":
            self.raw_data = share4v_val_dataset_4_LLM()
            self.raw_data = dataset_2_HFdataset(self.raw_data)
        elif data_args.subset_name == "Urban200K":
            self.raw_data = urban1k_dataset_4_LLM()
            self.raw_data = dataset_2_HFdataset(self.raw_data) 

        if modality == "image":
            self.eval_data = self.get_image_data()
        else:
            self.eval_data = self.get_text_data()

    def __len__(self):
        return len(self.eval_data)

    def __getitem__(self, idx):
        text, image = self.eval_data[idx]
        if self.model_backbone != PHI3V:
            text = text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            if self.data_args.image_resolution:
                image = process_image(image, self.data_args.image_resolution)
        return text, image

    def get_image_data(self):
        eval_data = []
        inst = "<|image_1|> Find an image caption describing the given image." 

        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
        return eval_data

    def get_text_data(self):
        eval_data = []
        inst = ""
        if self.data_args.subset_name == "coco2014" or self.data_args.subset_name == "flickr30k":
            for row in self.raw_data:
                for caption in row["text"]: eval_data.append((inst + caption, None))
        else:
            for row in self.raw_data: eval_data.append((inst + row["text"], None))
        return eval_data
    
class EvalDataset4Flickr_COCO_ShareGPT4V_Urban1K_rerank(Dataset):
    def __init__(self, modality, model_backbone, data_args):
        self.model_backbone = model_backbone
        self.data_args = data_args
        self.modality = modality
        if data_args.subset_name == "coco2014":
            self.raw_data = load_dataset(COCO["data_path"], split='test')
            self.raw_data = self.raw_data.map(lambda x: {'text': x['text'][:5]}, num_proc=16)
        elif data_args.subset_name == "flickr30k":
            self.raw_data = load_dataset(FLICKR30K["data_path"], split='test')
        elif data_args.subset_name == "sharegpt4v":
            self.raw_data = share4v_val_dataset_4_LLM()
            self.raw_data = dataset_2_HFdataset(self.raw_data)
        elif data_args.subset_name == "Urban200K":
            self.raw_data = urban1k_dataset_4_LLM()
            self.raw_data = dataset_2_HFdataset(self.raw_data)
        self.rerank_candidates = torch.load(f"{data_args.encode_output_path}/rerank_top10.pt")
        self.topk_4_rerank=10
        """
        saved_item = {
            "top10_candidates_I2T": top10_candidates_I2T, # 1000,10 | 5000, 10 | 1000, 10 | 1000, 10
            "top10_candidates_T2I": top10_candidates_T2I  # 5000,10 | 25000, 10 | 1000, 10 | 1000, 10
        }
        """
        self.image_data = self.get_image_data()
        self.text_data = self.get_text_data()
        if modality == "image":
            self.eval_data = self.get_image_data_4_rerank_list()
        else:
            self.eval_data = self.get_text_data_4_rerank_list()

    def __len__(self):
        return len(self.eval_data)
        
    def __getitem__(self, idx):
        '''
        examples = {'qry_text': [e[0] for e in examples],
                    'qey_image': [e[1] for e in examples],
                    'cand_text': [[each[0] for each in e[2]] for e in examples],
                    'cand_image': [[each[1] for each in e[2]] for e in examples]}
        '''
        qry_text, qry_img, cand_data = self.eval_data[idx]["qry_text"], self.eval_data[idx]["qry_image"], self.eval_data[idx]["cand_data"]
        if self.model_backbone != PHI3V:
            qry_text = qry_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone])
            new_cand_data = []
            for cand_text, cand_image in cand_data:
                new_cand_data.append([cand_text.replace(vlm_image_tokens[PHI3V], vlm_image_tokens[self.model_backbone]), process_image(cand_image, self.data_args.image_resolution)])
        
        return qry_text, process_image(qry_img, self.data_args.image_resolution), new_cand_data

    def get_image_data(self):
        eval_data = []
        inst = "<|image_1|> Find an image caption describing the given image." 
        for row in self.raw_data:
            eval_data.append((inst, row["image"]))
        return eval_data

    def get_text_data(self):
        eval_data = []
        inst = ""
        if self.data_args.subset_name == "coco2014" or self.data_args.subset_name == "flickr30k":
            for row in self.raw_data:
                for caption in row["text"]: eval_data.append((inst + caption, None))
        else:
            for row in self.raw_data: eval_data.append((inst + row["text"], None))
        return eval_data

    def get_image_data_4_rerank_list(self):
        eval_data = []
        for row1, row2 in zip(self.image_data, self.rerank_candidates["top10_candidates_I2T"]):
            candidate_data = [self.text_data[idx] for idx in row2[:self.topk_4_rerank]]
            eval_data.append({'qry_text': row1[0],
                                'qry_image': row1[1],
                                'cand_data': candidate_data})
        return eval_data

    def get_text_data_4_rerank_list(self):
        eval_data = []
        for row1, row2 in zip(self.text_data, self.rerank_candidates["top10_candidates_T2I"]):
            candidate_data = [self.image_data[idx] for idx in row2[:self.topk_4_rerank]]
            eval_data.append({'qry_text': row1[0],
                                'qry_image': row1[1],
                                'cand_data': candidate_data})
        return eval_data